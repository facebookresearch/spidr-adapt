# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Training loop."""

import dataclasses
import json
import logging
import os
from contextlib import ExitStack
from pathlib import Path

import torch
import wandb
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from spidr_adapt.checkpoint import Checkpointer, remove_param_group
from spidr_adapt.config import Config, ResumeConfig, read_config
from spidr_adapt.data import BatchType, InterleaveSLDatasetLoader
from spidr_adapt.environment import setup_training
from spidr_adapt.models import LossWeightDefinition, build_model
from spidr_adapt.optimizer import build_optimizer
from spidr_adapt.slurm import launch_with_submitit, validation_job_config
from spidr_adapt.tools import AverageMeters, profiler_context
from spidr_adapt.validate import validate_existing_checkpoint

logger = logging.getLogger()


def init_wandb(cfg: Config) -> str | None:
    logger.debug("Initializing wandb")
    cfg.run.dir.mkdir(parents=True, exist_ok=True)
    (cfg.run.dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=4))
    run = wandb.init(
        project=cfg.run.wandb_project,
        name=cfg.run.wandb_name,
        mode=cfg.run.wandb_mode,
        dir=Path(cfg.run.workdir).resolve(),
        config=dataclasses.asdict(cfg),
    )
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    for prefix in cfg.validation:
        wandb.define_metric(f"{prefix}/step")
        wandb.define_metric(f"{prefix}/*", step_metric=f"{prefix}/step")
    return run.id


def launch_validation(cfg: Config, resume: ResumeConfig) -> None:
    if resume.checkpoint is None:
        logger.info("No checkpoint found, skipping validation at step %s", resume.step)
        return
    if cfg.run.slurm_validation is None and "SLURM_JOB_ID" not in os.environ:
        logger.info("Not training in a Slurm job and no slurm_validation config provided, skipping validation")
        return
    logger.info("Launching validation at step %s", resume.step)
    launch_with_submitit(
        f"val-{resume.step}",
        [(validate_existing_checkpoint, (cfg, resume))],
        cfg.run.dir / "validation",
        cfg.run.slurm_validation or validation_job_config(),
        copy_code=False,
    )


def get_loss_weights_and_batch_type(
    step: int,
    supervised_every_step: int | None = None,
) -> tuple[LossWeightDefinition, BatchType]:
    """Determine the loss weights and data type based on the current step."""
    if supervised_every_step and step % supervised_every_step == 0:
        loss_weights = LossWeightDefinition(ssl=0.0, supervised=1.0)
        batch_type = BatchType.SUPERVISED
    else:
        loss_weights = LossWeightDefinition(ssl=1.0, supervised=0.0)
        batch_type = BatchType.SSL
    return loss_weights, batch_type


def train(cfg: Config) -> None:  # noqa: PLR0914, PLR0915, C901
    with ExitStack() as stack:
        logger.info("Starting job")
        setup_training(cfg.run.random_seed, use_deterministic=cfg.run.use_deterministic)
        stack.callback(dist.destroy_process_group)
        global_rank, world_size = dist.get_rank(), dist.get_world_size()
        is_main = global_rank == 0
        if is_main:
            init_wandb(cfg)
            stack.callback(wandb.finish)

        logger.info("Building model, optimizer, and dataloaders")
        device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[cfg.optimizer.dtype]
        model = build_model(cfg=cfg.model, model_type=cfg.run.model_type, checkpoint=cfg.run.init_ckpt)
        model = model.to(device).train()
        optimizer, scaler, scheduler = build_optimizer(model, cfg.optimizer)
        dist.barrier(device_ids=[device.index])
        ckpt = Checkpointer(cfg.run.dir, cfg.run.save_interval, cfg.run.keep_latest)
        ckpt.init_state(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        resuming = ckpt.load_existing_run()
        step, epoch, supervised_epoch = int(ckpt.step), int(ckpt.epoch), int(ckpt.supervised_epoch)
        loader = InterleaveSLDatasetLoader(cfg, epoch, supervised_epoch=supervised_epoch)
        if not resuming and is_main and ckpt.save(step, epoch, supervised_epoch=supervised_epoch):
            launch_validation(cfg, ResumeConfig(step=step, checkpoint=ckpt.last, results=ckpt.metrics))
        ddp_model = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True)

        logger.info("Starting training loop")
        meters = AverageMeters(
            ["loss", "supervised_loss", "ssl_loss", "grad_norm", "batch_size", "target_ppl", "pred_ppl"], device=device
        )
        profiler = stack.enter_context(profiler_context(cfg.run.dir / "trace.html" if is_main else None))
        pbar = stack.enter_context(tqdm(total=cfg.optimizer.max_steps, initial=step, disable=not is_main))
        while step < cfg.optimizer.max_steps:
            loss_weights, batch_type = get_loss_weights_and_batch_type(step, cfg.model.supervised_every_step)
            batch = loader.load_batch_data(batch_type)
            if step >= cfg.optimizer.max_steps:
                break
            if step == cfg.model.freeze_step and len(optimizer.param_groups) > 1:
                remove_param_group(optimizer, 1)

            with torch.autocast("cuda", dtype, cfg.optimizer.mixed_precision):
                ssl_loss, supervised_loss, outputs = ddp_model(batch.to(device), loss_weights=loss_weights)
            num_frames = torch.tensor(ssl_loss.size(0), dtype=torch.long, device=device)
            dist.all_reduce(num_frames)
            ssl_loss = ssl_loss.sum() * world_size / num_frames
            loss = loss_weights.supervised * supervised_loss + loss_weights.ssl * ssl_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(ddp_model.parameters(), cfg.optimizer.max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            lr = scheduler.get_last_lr()[0]
            scheduler.step()
            step += 1
            ema_decay = model.update_ema(step)

            meters.update(loss=loss.detach(), supervised_loss=supervised_loss.detach(), ssl_loss=ssl_loss.detach())
            meters.update(batch_size=batch.waveforms.size(0), grad_norm=grad_norm)
            meters.update(target_ppl=outputs["target_ppl"], pred_ppl=outputs["pred_ppl"])
            pbar.update()
            if is_main and step % cfg.run.log_interval == 0:
                infos = meters.pop() | {
                    "lr": lr,
                    "ema_decay": ema_decay * 1000,
                    "step": step,
                    "epoch": epoch,
                    "supervised_epoch": supervised_epoch,
                }
                wandb.log({f"train/{key}": value for key, value in infos.items()})
                pbar.set_postfix(loss=infos["loss"], target_ppl=infos["target_ppl"], pred_ppl=infos["pred_ppl"])
            if is_main and ckpt.save(step, epoch, supervised_epoch=supervised_epoch):
                launch_validation(cfg, ResumeConfig(step=step, checkpoint=ckpt.last, results=ckpt.metrics))
                for val_metric in ckpt.find_new_metrics():
                    wandb.log(val_metric)
            profiler.step()

        if is_main and ckpt.save_final(step, epoch, supervised_epoch):
            launch_validation(cfg, ResumeConfig(step=step, checkpoint=ckpt.last, results=ckpt.metrics))
        logger.info("Training finished")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    cfg = read_config(parser.parse_args().config)
    train(cfg)
