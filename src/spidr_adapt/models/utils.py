# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Model loading utilities."""

import collections
import json
import logging
import typing as tp
import warnings
from pathlib import Path

import torch
from torch.hub import load_state_dict_from_url
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from spidr_adapt.config import ModelType, SpidRConfig, SpidRWithResetConfig
from spidr_adapt.models.spidr import SpidR
from spidr_adapt.models.spidr_reset import SpidRWithReset

logger = logging.getLogger()


def model_from_raw_checkpoint(
    model_class: type[SpidR],
    config_class: type[SpidRConfig],
    ckpt: str | Path,
    cfg: SpidRWithResetConfig | None = None,
) -> SpidR:
    path = Path(ckpt)
    if path.suffix != ".pt":
        raise ValueError("Only .pt files are supported.")
    if cfg is None:
        if (path.parent / "config.json").is_file():
            with (path.parent / "config.json").open() as f:
                json_cfg = json.load(f)
                if "model" in json_cfg:
                    json_cfg = json_cfg["model"]
                cfg = config_class(**json_cfg)
        else:
            warnings.warn("Config file not found when loading checkpoint. Using default config.", stacklevel=2)
            cfg = config_class()

    instance = model_class(cfg)
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    consume_prefix_in_state_dict_if_present(state_dict, "module.")
    missing_keys, unexpected_keys = instance.load_state_dict(state_dict, strict=False)
    if len(missing_keys) > 0:
        logger.warning("Missing keys when loading state_dict %s", missing_keys)
    if len(unexpected_keys) > 0:
        logger.warning("Unexpected keys when loading state_dict %s", unexpected_keys)
    return instance


def build_model(
    *,
    cfg: SpidRConfig | None = None,
    model_type: ModelType | None = None,
    checkpoint: str | Path | None = None,
) -> SpidR:
    if checkpoint is not None:
        match model_type:
            case "spidr":
                return model_from_raw_checkpoint(SpidR, SpidRConfig, checkpoint)
            case "spidr_reset":
                return model_from_raw_checkpoint(SpidRWithReset, SpidRWithResetConfig, checkpoint, cfg)
            case _:
                raise ValueError(f"Model type not recognized, acceptable models are {tp.get_args(ModelType)}.")
    if isinstance(cfg, SpidRWithResetConfig):
        return SpidRWithReset(cfg)
    if isinstance(cfg, SpidRConfig):
        return SpidR(cfg)
    raise ValueError("Invalid cfg class")


def spidr_adapt_base(*, pretrained: bool = True, check_hash: bool = False, progress: bool = True) -> SpidRWithReset:
    model = SpidRWithReset(SpidRWithResetConfig())
    if pretrained:
        url = ""  # TODO: add path
        checkpoint = load_state_dict_from_url(url, check_hash=check_hash, progress=progress, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    model.eval()
    return model
