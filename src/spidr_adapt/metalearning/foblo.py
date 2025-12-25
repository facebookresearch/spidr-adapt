# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""FOBLO wrapper implementation."""

import copy
import logging

import torch
import torch.distributed as dist
from torch import nn

from spidr_adapt.environment import pg_metalearning
from spidr_adapt.metalearning.meta_updater import MetaUpdater, parameters_to_sync
from spidr_adapt.metalearning.utils import all_reduce_coalesced

logger = logging.getLogger()


class FOBLO(MetaUpdater):
    def __init__(self, *, model: nn.Module, beta: float, task_interval: int) -> None:
        """Distributed FOBLO.

        This class is generic and works with any PyTorch Module.
        """
        super().__init__(model=model, beta=beta, task_interval=task_interval)
        self.theta_inner = copy.deepcopy(model).eval()

    @torch.no_grad()
    def perform_inner_update(self, model: nn.Module) -> None:
        step = model.current_step.item() - 1
        logger.debug(
            "FOBLO updates theta_ssl at episode %s at local step %s, at the global step %s.",
            step // self.task_interval,
            step % self.task_interval,
            step,
        )
        theta_inner_lerp, theta_inner_copy = parameters_to_sync(self.theta_inner)
        worker_lerp, worker_copy = parameters_to_sync(model)
        torch._foreach_copy_(theta_inner_copy, worker_copy)
        torch._foreach_copy_(theta_inner_lerp, worker_lerp)

    @torch.no_grad()
    def forward(self, model: nn.Module) -> None:
        step = model.current_step.item() - 1
        logger.debug(
            "FOBLO meta-update at episode %s at local step %s, at the global step %s.",
            step // self.task_interval,
            step % self.task_interval,
            step,
        )
        central_lerp, central_copy = parameters_to_sync(self.central)
        worker_lerp, worker_copy = parameters_to_sync(model)
        theta_inner_lerp, _ = parameters_to_sync(self.theta_inner)
        if central_copy:
            torch._foreach_copy_(central_copy, worker_copy)
        if central_lerp:
            torch._foreach_add_(central_lerp, worker_lerp, alpha=self.beta)
            torch._foreach_add_(central_lerp, theta_inner_lerp, alpha=-self.beta)
            if self.num_workers > 1:
                all_reduce_coalesced(central_lerp, pg_metalearning(), op=dist.ReduceOp.AVG)
        model.load_state_dict(self.central.state_dict())
