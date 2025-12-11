# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Base class for meta update which wrap around the model."""

import copy
import itertools
import logging

import torch
import torch.distributed as dist
from torch import nn

from spidr_adapt.environment import pg_metalearning
from spidr_adapt.metalearning.utils import sync_module_states

logger = logging.getLogger()


def parameters_to_sync(model: nn.Module) -> tuple[tuple[nn.Parameter, ...], tuple[nn.Parameter, ...]]:
    """Lerp parameters that require grad and copy the rest.

    The only tensor to copy is 'current_step'.
    """
    to_lerp, to_copy = [], []
    for name, param in itertools.chain(model.named_parameters(), model.named_buffers()):
        if name == "current_step" or "convolution.norm.num_batches_tracked" in name:
            to_copy.append(param)
        else:
            to_lerp.append(param)
    return tuple(to_lerp), tuple(to_copy)


class MetaUpdater(nn.Module):
    def __init__(self, *, model: nn.Module, beta: float, task_interval: int) -> None:
        """Distributed MetaUpdater.

        This class is generic and works with any PyTorch Module.
        Reptile and FOBLO inherit from this class.
        """
        super().__init__()
        self.beta = beta
        self.task_interval = task_interval
        self.num_workers = dist.get_world_size(pg_metalearning()) if dist.is_initialized() else 1
        if self.num_workers > 1:
            sync_module_states(model, pg_metalearning(), src=0)
        self.central = copy.deepcopy(model).eval()

    @torch.no_grad()
    def perform_inner_update(self, model: nn.Module) -> None:
        pass

    @torch.no_grad()
    def forward(self, model: nn.Module) -> None:
        pass
