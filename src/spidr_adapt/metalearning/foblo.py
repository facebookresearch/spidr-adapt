# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""FOBLO wrapper implementation."""

import logging

import torch
from torch import nn

from spidr_adapt.metalearning.meta_updater import MetaUpdater

logger = logging.getLogger()


class FOBLO(MetaUpdater):
    def __init__(self, *, model: nn.Module, beta: float) -> None:
        """Distributed FOBLO.

        This class is generic and works with any PyTorch Module.
        """
        # TODO: Jiayi to add FOBLO init
        super().__init__(model=model, beta=beta)

    @torch.no_grad()
    def forward(self, model: nn.Module) -> None:
        # TODO: Jiayi to add FOBLO forward
        pass
