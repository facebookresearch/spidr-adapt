# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Script for creating adaptation manifests."""

import argparse
from pathlib import Path

import pandas as pd

ADAPTATION_SETS = {
    "vp_en": "datasets/adapt/vp_en",
    "vp_fr": "datasets/adapt/vp_fr",
    "vp_de": "datasets/adapt/vp_de",
}


def prepare_adaptation_manifest(adapt_subset: str, data_path: Path) -> None:
    train_manifest = pd.read_csv((data_path / "spidr-adapt" / "train" / f"{adapt_subset}_manifest").with_suffix(".csv"))
    adaptation_metadata_dir = ADAPTATION_SETS[adapt_subset]

    data_scales = ["10min", "1h", "10h", "100h"]

    for data_scale in data_scales:
        adaptation_set = pd.read_csv((Path(f"{adaptation_metadata_dir}_{data_scale}")).with_suffix(".csv"))
        adaptation_manifest = adaptation_set.merge(train_manifest, on="fileid", how="inner")
        output_dir = data_path / "spidr-adapt" / "adapt" / adapt_subset
        output_dir.mkdir(exist_ok=True, parents=True)
        adaptation_manifest.to_csv(output_dir / f"{data_scale}_manifest.csv")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True, help="dataset path containing spidr adapt data (parent dir of spidr-adapt)"
    )
    parser.add_argument(
        "--adapt_subset",
        type=str,
        choices=ADAPTATION_SETS.keys(),
        required=True,
        help="language subset to prepare adaptation sets",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    prepare_adaptation_manifest(args.adapt_subset, Path(args.data_path))
