# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Utils for creating manifests."""

from pathlib import Path

import httpx
from tqdm import tqdm


def download(url: str, dest: str | Path) -> None:
    with Path(dest).open("wb") as download_file, httpx.stream("GET", url) as response:
        total, name = int(response.headers["Content-Length"]), Path(response.url.path).name
        with tqdm(desc=f"Downloading {name}", total=total, unit_scale=True, unit_divisor=1024, unit="B") as progress:
            num_bytes_downloaded = response.num_bytes_downloaded
            for chunk in response.iter_bytes():
                download_file.write(chunk)
                progress.update(response.num_bytes_downloaded - num_bytes_downloaded)
                num_bytes_downloaded = response.num_bytes_downloaded
