# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Data loading and dataset utilities."""

from spidr_adapt.data.dataset import (
    BatchType,
    InterleaveSLDatasetLoader,
    SampleBatch,
    build_dataloader,
    speech_dataset,
)
from spidr_adapt.data.tokenizer import Tokenizer, get_tokenizer_for_lang
from spidr_adapt.data.utils import get_number_of_data_chunks, num_samples, read_manifest

__all__ = [
    "BatchType",
    "InterleaveSLDatasetLoader",
    "SampleBatch",
    "Tokenizer",
    "build_dataloader",
    "get_number_of_data_chunks",
    "get_tokenizer_for_lang",
    "num_samples",
    "read_manifest",
    "speech_dataset",
]
