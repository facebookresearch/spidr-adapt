# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Reptile wrapper implementation."""

import logging

import torch
import torch.distributed as dist
from torch import nn

from spidr_adapt.environment import pg_metalearning
from spidr_adapt.metalearning.meta_updater import MetaUpdater, parameters_to_sync
from spidr_adapt.metalearning.utils import all_reduce_coalesced

logger = logging.getLogger()


class Reptile(MetaUpdater):
    def __init__(self, *, model: nn.Module, beta: float, task_interval: int) -> None:
        """Distributed Reptile.

        This class is generic and works with any PyTorch Module.
        """
        super().__init__(model=model, beta=beta, task_interval=task_interval)

    @torch.no_grad()
    def forward(self, model: nn.Module) -> None:
        step = model.current_step.item() - 1
        logger.debug(
            "Reptile meta-update at episode %s at local step %s, at the global step %s.",
            step // self.task_interval,
            step % self.task_interval,
            step,
        )
        logger.debug("Reptile updates performed at step %s", model.current_step.item())
        central_lerp, central_copy = parameters_to_sync(self.central)
        worker_lerp, worker_copy = parameters_to_sync(model)
        if central_copy:
            torch._foreach_copy_(central_copy, worker_copy)  # central <- worker
        if central_lerp:
            torch._foreach_lerp_(central_lerp, worker_lerp, self.beta)  # central <- (1-beta) * central + beta * worker
            if self.num_workers > 1:
                all_reduce_coalesced(central_lerp, pg_metalearning(), op=dist.ReduceOp.AVG)
        model.load_state_dict(self.central.state_dict())
