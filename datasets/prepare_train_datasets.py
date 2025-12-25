# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Script for segmenting train dataset and creating manifests."""

import argparse
from pathlib import Path

import pandas as pd
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder
from tqdm import tqdm
from utils import download

DOWNLOAD_URL_BASE = "https://dl.fbaipublicfiles.com/shared/devai/assets/spidr-adapt/datasets/assets/manifests/train"

TRAIN_SETS = ["vp_19lang", "vp_en", "vp_fr", "vp_de"]


def segment_audios_and_prepare_train_manifest(train_subset: str, data_path: Path) -> None:
    event_ids = {}
    meta_data_path = data_path / f"{train_subset}.csv"
    download(url=f"{DOWNLOAD_URL_BASE}/{train_subset}.csv", dest=meta_data_path)
    train_meta_data = pd.read_csv(meta_data_path)

    for _, row in train_meta_data.iterrows():
        if row["event_id"] not in event_ids:
            event_ids[row["event_id"]] = [row]
        else:
            event_ids[row["event_id"]].append(row)

    manifest = []
    for event_id, rows in tqdm(event_ids.items()):
        year = event_id[:4]
        language = event_id.split("_")[-1]
        audio_path = (data_path / "raw_audios" / language / year / event_id).with_suffix(".ogg")
        samples = AudioDecoder(audio_path).get_all_samples()
        wav, sr = samples.data, samples.sample_rate
        for row in rows:
            output_lang_dir = data_path / "spidr-adapt" / "segmented_audio" / language
            output_lang_dir.mkdir(exist_ok=True, parents=True)
            output_path = (output_lang_dir / row["fileid"]).with_suffix(".wav")
            segmented_audio = wav[:, int(row["start_time"] * sr) : int(row["end_time"] * sr)]
            encoder = AudioEncoder(samples=segmented_audio, sample_rate=sr)
            encoder.to_file(str(output_path), sample_rate=sr)
            manifest.append(
                {
                    "fileid": row["fileid"],
                    "num_frames": segmented_audio.shape[-1],
                    "language": language,
                    "path": output_path,
                }
            )
    manifest = pd.DataFrame(manifest)
    manifest_dir = data_path / "spidr-adapt" / "train"
    manifest_dir.mkdir(exist_ok=True, parents=True)
    manifest.to_csv((manifest_dir / f"{train_subset}_manifest").with_suffix(".csv"))
    meta_data_path.unlink()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="dataset path containing voxpopuli data (parent dir of raw_audios)",
    )
    parser.add_argument(
        "--train_subset",
        type=str,
        choices=TRAIN_SETS,
        required=True,
        help="train subset to prepare",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    segment_audios_and_prepare_train_manifest(args.train_subset, Path(args.data_path))
