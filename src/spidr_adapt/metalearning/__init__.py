# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Reptile wrapper."""

from spidr_adapt.metalearning.foblo import FOBLO
from spidr_adapt.metalearning.meta_updater import MetaUpdater
from spidr_adapt.metalearning.reptile import Reptile

__all__ = ["FOBLO", "MetaUpdater", "Reptile"]

META_UPDATER_CLASSES = {
    "reptile": Reptile,
    "foblo": FOBLO,
}
