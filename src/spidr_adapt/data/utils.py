# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Data utilities."""

import logging
import mmap
import os
from os import PathLike
from pathlib import Path

import polars as pl
from polars import DataFrame
from torchcodec.decoders import AudioDecoder

logger = logging.getLogger(__name__)


MERGE_CHUNK_THRESH = 100  # if remainder data lang chunk is greater than this (in seconds), a new chunk will be created


def num_samples(source: str | Path | bytes, *, verify: bool = False) -> int:
    metadata = AudioDecoder(source).metadata
    samples = metadata.duration_seconds_from_header * metadata.sample_rate
    if verify and not samples.is_integer():
        raise ValueError(
            f"Number of samples {samples} is not an integer"
            + (f" in {source}" if isinstance(source, (str, Path)) else "")
        )
    return int(samples)


def bytes_from_archive(archive: Path | str, offset: int, file_size: int) -> bytes:
    with Path(archive).open("rb") as path, mmap.mmap(path.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_o:
        return mmap_o[offset : offset + file_size]


def read_alignments(alignments_path: PathLike) -> dict[str, list[str]]:
    alignments = pl.read_csv(alignments_path)
    assert "language" in alignments.columns, "Need language column in alignments to identify phoneme tokenizer."
    return {row["fileid"]: (row["phones"], row["language"]) for row in alignments.iter_rows(named=True)}


def read_manifest(path: Path | str) -> pl.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        return pl.read_csv(path)
    if path.suffix == ".jsonl":
        return pl.read_ndjson(path)
    if path.suffix != ".tsv":
        raise ValueError("Only .csv, .jsonl and .tsv files are supported")
    with path.open("r") as file:
        root = Path(file.readline().strip())
    if not root.is_dir():
        raise ValueError("First line must be the root directory of the dataset")
    return (
        pl.scan_csv(path, separator="\t", skip_rows=1, has_header=False, new_columns=["fileid", "num_samples"])
        .with_columns((f"{root}{os.sep}" + pl.col("fileid")).alias("path"))
        .select("fileid", "path", "num_samples")
        .collect()
    )


def get_number_of_data_chunks(manifest_file: str, sample_rate: int, lang_task_chunk_duration: int | None) -> int:
    """Calculate number of data chunks in the manifest file across languages.

    manifest_file: str, path to the manifest file.
    sample_rate: int, sample rate of the audio files.
    lang_task_chunk_duration: int | None, desired language chunk size in seconds.
    """
    manifest: DataFrame = read_manifest(manifest_file)
    if "language" not in manifest.columns:
        raise ValueError("Manifest must contain 'language' column for language chunking.")
    if not lang_task_chunk_duration:
        return len(manifest["language"].unique())
    manifest = (
        manifest.group_by("language")
        .agg(pl.col("num_frames").sum())
        .with_columns((pl.col("num_frames") // (sample_rate * lang_task_chunk_duration)).alias("floor_chunks"))
        .with_columns(
            (pl.col("num_frames") % (sample_rate * lang_task_chunk_duration) > sample_rate * MERGE_CHUNK_THRESH)
            .alias("remainder")
            .cast(int)
        )
    )
    return (manifest["floor_chunks"] + manifest["remainder"]).sum()


def divide_manifest_language_by_chunks(
    manifest: DataFrame, sample_rate: int, lang_task_chunk_duration: int, seed: int
) -> DataFrame:
    """Divide the manifest into chunks of a specified duration for each language.

    manifest: DataFrame with columns ["fileid", "path", "num_frames", "language"]
    sample_rate: int, sample rate of the audio files.
    lang_task_chunk_duration is desired language chunk in seconds
    """
    chunked_manifest = []
    for lang in manifest["language"].unique(maintain_order=True):
        lang_manifest = manifest.filter(pl.col("language") == lang).sample(fraction=1.0, seed=seed, shuffle=True)
        frames_chunk = lang_task_chunk_duration * sample_rate
        assert lang_manifest["num_frames"].sum() >= frames_chunk, "Not enough frames for chunking language"
        lang_manifest = lang_manifest.with_columns(
            (pl.col("num_frames").cum_sum()).alias("cumsum_frames")
        ).with_columns((pl.lit(None, dtype=int)).alias("language_chunk"))
        frame_chunk_distances = range(0, lang_manifest["num_frames"].sum(), frames_chunk)
        for i, c in enumerate(frame_chunk_distances):
            lang_manifest = lang_manifest.with_columns(
                pl.when(pl.col("cumsum_frames") > c)
                .then(i)
                .otherwise(pl.col("language_chunk"))
                .alias("language_chunk")
            )
        max_chunk = lang_manifest["language_chunk"].max()
        if (
            lang_manifest.filter(pl.col("language_chunk") == max_chunk)["num_frames"].sum()
            < MERGE_CHUNK_THRESH * sample_rate
        ):
            lang_manifest = lang_manifest.with_columns(
                pl.when(pl.col("language_chunk") == max_chunk)
                .then(max_chunk - 1)
                .otherwise(pl.col("language_chunk"))
                .alias("language_chunk")
            )
        lang_manifest = lang_manifest.with_columns(
            pl.format("{}_chunk{}", pl.col("language"), pl.col("language_chunk")).alias("language")
        )
        chunked_manifest.append(lang_manifest)
    chunked_manifest = pl.concat(chunked_manifest)
    chunked_manifest = chunked_manifest.drop(["cumsum_frames", "language_chunk"])
    logger.info("Split data into %s language tasks", chunked_manifest["language"].n_unique())
    return chunked_manifest
