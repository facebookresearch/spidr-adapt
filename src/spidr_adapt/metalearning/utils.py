# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Metalearning utils."""

from collections.abc import Container

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.utils import _sync_module_states  # noqa: PLC2701


def sync_module_states(
    module: nn.Module,
    process_group: dist.ProcessGroup,
    *,
    src: int,
    params_and_buffers_to_ignore: Container[str] | None = None,
    broadcast_buffers: bool = True,
) -> None:
    """Synchronize the state of a module across the process group.

    Buffer size taken from DDP:
    https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/nn/parallel/distributed.py#L800
    """
    _sync_module_states(
        module,
        process_group,
        250 * 1024 * 1024,
        src,
        params_and_buffers_to_ignore or [],
        broadcast_buffers,
    )


def all_reduce_coalesced(
    tensors: tuple[torch.Tensor, ...],
    group: dist.ProcessGroup | None,
    *,
    op: dist.ReduceOp.RedOpType,
) -> None:
    """Not using `torch.distributed.all_reduce_coalesced` because of depreciation warning."""
    flattened_tensors = nn.utils.parameters_to_vector(tensors)
    dist.all_reduce(flattened_tensors, op=op, group=group)
    nn.utils.vector_to_parameters(flattened_tensors, tensors)
