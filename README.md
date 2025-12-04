# SpidR-Adapt: A Universal Speech Representation Model for Few-Shot Adaptation

## Overview

## Installation

- With [uv](https://docs.astral.sh/uv/#getting-started) (recommended):
```bash
uv sync
```

- With conda (same for mamba or micromamba):
```bash
conda create -n spidr-adapt python=3.12 -c conda-forge
conda activate 
uv pip install -e . --group dev
```

- With standard `pip`:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e . --group dev
```

Further, you must have FFmpeg installed on your cluster for torchcodec audio loading. Some versions of FFmpeg (e.g., FFmpeg4 and FFmpeg8) are incompatible with torchcodec and fail to load certain audio files--it is recommended to have FFmpeg7. Check the version available with `ffmpeg -version`. If you get a `command not found` error or if the version available is not 7.x.x, proceed with the conda instructions and add `conda install ffmpeg=7.0.0 -c conda-forge`.

## Usage

## License

The source code and our two model checkpoints are provided under the __ License.
