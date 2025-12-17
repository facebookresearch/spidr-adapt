# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Training loop."""

import logging
import os
from contextlib import ExitStack
from dataclasses import asdict

import torch
import wandb
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from spidr_adapt.checkpoint import Checkpointer, remove_param_group
from spidr_adapt.config import SAMPLE_RATE, Config, DataConfig, MetaUpdateConfig, ResumeConfig, read_config
from spidr_adapt.data import BatchType, InterleaveSLDatasetLoader, get_number_of_data_chunks
from spidr_adapt.environment import pg_ddp, pg_metalearning, setup_training
from spidr_adapt.metalearning import META_UPDATER_CLASSES
from spidr_adapt.models import LossWeightDefinition, build_model
from spidr_adapt.optimizer import build_optimizer
from spidr_adapt.tools import AverageMeters, profiler_context
from spidr_adapt.train import init_wandb, launch_validation

logger = logging.getLogger()


def get_num_tasks(data_cfg: DataConfig) -> int:
    """Get the number of tasks in the inner loop.

    Number of tasks = number of data chunks across all languages.
    """
    ssl_data_cfg = data_cfg if isinstance(data_cfg, DataConfig) else data_cfg[BatchType.SSL.name.lower()]
    return get_number_of_data_chunks(ssl_data_cfg.manifest, SAMPLE_RATE, ssl_data_cfg.lang_task_chunk_duration)


def get_meta_train_dataloader(
    cfg: Config,
    step: int,
    epoch: int,
    supervised_epoch: int,
    metalearning_size: int,
    metalearning_rank: int,
    num_tasks: int | None,
) -> torch.utils.data.DataLoader:
    """Build the dataloader.

    Adds logic for reshuffling data within tasks once cycled through all tasks by changing seed.
    """
    ssl_data_cfg = cfg.data if isinstance(cfg.data, DataConfig) else cfg.data[BatchType.SSL.name.lower()]
    supervised_data_cfg = None if isinstance(cfg.data, DataConfig) else cfg.data.get(BatchType.SUPERVISED.name.lower())
    num_reset = step // cfg.meta_update.task_interval
    if num_reset == 0:
        return InterleaveSLDatasetLoader(cfg, epoch, supervised_epoch=supervised_epoch)
    # create new ssl data config to change seed
    ssl_data_cfg_json = asdict(ssl_data_cfg)
    ssl_data_cfg_json["random_seed"] = int(
        (num_reset * metalearning_size + metalearning_rank) // num_tasks
    )  # random seed = number of times we have iterated through the tasks
    logger.info("Resetting dataloader with seed %s", ssl_data_cfg_json["random_seed"])
    ssl_data_cfg_seeded = DataConfig(**ssl_data_cfg_json)
    cfg = Config(
        run=cfg.run,
        data={"ssl": ssl_data_cfg_seeded, "supervised": supervised_data_cfg},
        model=cfg.model,
        optimizer=cfg.optimizer,
        masking=cfg.masking,
        validation=cfg.validation,
        meta_update=cfg.meta_update,
    )
    return InterleaveSLDatasetLoader(cfg, epoch, supervised_epoch=supervised_epoch)


def should_reset_data_loader(
    step: int, inner_step: int, metalearning_size: int, metalearning_rank: int, num_tasks: int
) -> bool:
    """Reset dataloader to reshuffle tasks after each run through all tasks."""
    if not num_tasks:
        return False
    num_reset = step // inner_step
    batch_language_index = (num_reset * metalearning_size + metalearning_rank) % num_tasks
    return 0 <= batch_language_index < metalearning_size


def get_meta_loss_weights_and_batch_type(
    step: int, meta_update_cfg: MetaUpdateConfig
) -> tuple[LossWeightDefinition, BatchType]:
    """Determine the loss weights and data type based on the current step."""
    if meta_update_cfg.method == "reptile":
        loss_weights = LossWeightDefinition(ssl=1.0, supervised=0.0)
        batch_type = BatchType.SSL
    elif meta_update_cfg.method == "foblo":
        if (step % meta_update_cfg.task_interval) < meta_update_cfg.inner_step:
            loss_weights = LossWeightDefinition(ssl=1.0, supervised=0.0)
            batch_type = BatchType.SSL
        else:
            loss_weights = LossWeightDefinition(ssl=0.0, supervised=1.0)
            batch_type = BatchType.SUPERVISED
    else:
        raise ValueError(f"Unknown meta_update method: {meta_update_cfg.method}")
    return loss_weights, batch_type


def meta_train(cfg: Config) -> None:  # noqa: C901, PLR0912, PLR0914, PLR0915
    with ExitStack() as stack:
        logger.info("Starting job")
        setup_training(
            cfg.run.random_seed, num_workers=cfg.meta_update.num_workers, use_deterministic=cfg.run.use_deterministic
        )
        stack.callback(dist.destroy_process_group)
        global_rank = dist.get_rank()
        is_main = global_rank == 0
        ddp_size, ddp_rank = dist.get_world_size(group=pg_ddp()), dist.get_rank(group=pg_ddp())
        metalearning_size, metalearning_rank = (
            dist.get_world_size(group=pg_metalearning()),
            dist.get_rank(group=pg_metalearning()),
        )
        logger.info("DDP %s / %s; Metalearning: %s / %s", ddp_rank, ddp_size, metalearning_rank, metalearning_size)
        if is_main:
            init_wandb(cfg)
            stack.callback(wandb.finish)

        logger.info("Building model, optimizer, and dataloaders")
        device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[cfg.optimizer.dtype]
        model = build_model(cfg=cfg.model, model_type=cfg.run.model_type, checkpoint=cfg.run.init_ckpt)
        model = model.to(device).train()
        model.set_task_interval(getattr(cfg.meta_update, "task_interval", None))
        optimizer, scaler, scheduler = build_optimizer(
            model,
            cfg.optimizer,
            getattr(cfg.meta_update, "task_interval", None),
            getattr(cfg.meta_update, "inner_step", None),
        )
        dist.barrier(device_ids=[device.index])
        ckpt = Checkpointer(cfg.run.dir, cfg.run.save_interval, cfg.run.keep_latest)
        ckpt.init_state(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        resuming = ckpt.load_existing_run()
        step, epoch, supervised_epoch = int(ckpt.step), int(ckpt.epoch), int(ckpt.supervised_epoch)
        model.current_step.fill_(step)
        num_tasks = get_num_tasks(cfg.data)
        loader = get_meta_train_dataloader(
            cfg, step, epoch, supervised_epoch, metalearning_size, metalearning_rank, num_tasks
        )
        loader.set_task(step, cfg.meta_update.task_interval)

        if not resuming and is_main and ckpt.save(step, epoch, supervised_epoch=supervised_epoch):
            launch_validation(cfg, ResumeConfig(step=step, checkpoint=ckpt.last, results=ckpt.metrics))
        ddp_model = DistributedDataParallel(
            model, device_ids=[device.index], find_unused_parameters=True, process_group=pg_ddp()
        )

        meta_updater = META_UPDATER_CLASSES[cfg.meta_update.method](
            model=model, beta=cfg.meta_update.beta, task_interval=cfg.meta_update.task_interval
        )

        logger.info("Starting training loop")
        meters = AverageMeters(
            ["loss", "supervised_loss", "ssl_loss", "grad_norm", "batch_size", "target_ppl", "pred_ppl"], device=device
        )
        profiler = stack.enter_context(profiler_context(cfg.run.dir / "trace.html" if is_main else None))
        pbar = stack.enter_context(tqdm(total=cfg.optimizer.max_steps, initial=step, disable=not is_main))
        while step < cfg.optimizer.max_steps:
            loss_weights, batch_type = get_meta_loss_weights_and_batch_type(step, cfg.meta_update)
            batch = loader.load_batch_data(batch_type)

            if step >= cfg.optimizer.max_steps:
                break
            if step == cfg.model.freeze_step and len(optimizer.param_groups) > 1:
                remove_param_group(optimizer, 1)

            with torch.autocast("cuda", dtype, cfg.optimizer.mixed_precision):
                ssl_loss, supervised_loss, outputs = ddp_model(batch.to(device), loss_weights=loss_weights)
            num_frames = torch.tensor(ssl_loss.size(0), dtype=torch.long, device=device)
            dist.all_reduce(num_frames, group=pg_ddp())
            ssl_loss = ssl_loss.sum() * ddp_size / num_frames
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

            if step % cfg.meta_update.task_interval == 0:
                meta_updater(model)
                if should_reset_data_loader(
                    step, cfg.meta_update.task_interval, metalearning_size, metalearning_rank, num_tasks
                ):
                    loader = get_meta_train_dataloader(
                        cfg, step, epoch, supervised_epoch, metalearning_size, metalearning_rank, num_tasks
                    )
                loader.set_task(step, cfg.meta_update.task_interval)
            if step % cfg.meta_update.task_interval == cfg.meta_update.inner_step:
                meta_updater.perform_inner_update(model)

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
    meta_train(cfg)
