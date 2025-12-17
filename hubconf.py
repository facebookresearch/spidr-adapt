# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""torch.hub configuration."""

from pathlib import Path

from torch.hub import _add_to_sys_path  # noqa: PLC2701

dependencies = ["torch", "numpy"]

with _add_to_sys_path(str(Path(__file__).parent / "src")):
    from spidr_adapt.models import spidr_adapt_base  # noqa: F401
