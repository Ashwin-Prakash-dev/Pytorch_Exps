"""CIFAR-10 loader sourced from Hugging Face's CDN instead of torchvision's
built-in downloader.

torchvision.datasets.CIFAR10 only knows one source (the University of
Toronto server), which was unusable when this experiment was built --
downloads stalled completely after ~50MB, then crawled at <15kB/s on retry
(general network speed to other hosts tested fine at ~100kB/s, so this was
specific to that server). Hugging Face serves the identical images/labels
as two parquet files at ~6-8MB/s. This module downloads them (once, ~140MB
combined) and decodes them into a torchvision-compatible Dataset -- same
(PIL image, label) interface, same `.targets` attribute, same CIFAR-10
label ordering (confirmed via the parquet's embedded label metadata:
0=airplane, ..., 9=truck), so nothing downstream (train.py, experiment 07)
needs to know the data didn't come from torchvision's own downloader.

Run standalone to pre-download and decode:
    python experiments/06_cifar10_cnn_airplane/cifar10_data.py
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

HF_URLS = {
    "train": "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet",
    "test": "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet",
}

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)


def _decode_parquet(parquet_path: Path, cache_path: Path) -> dict:
    if cache_path.exists():
        return torch.load(cache_path)

    print(f"decoding {parquet_path} (first run only, cached after)")
    df = pd.read_parquet(parquet_path)
    images = torch.empty(len(df), 32, 32, 3, dtype=torch.uint8)
    for i, row in enumerate(df.itertuples(index=False)):
        img = Image.open(io.BytesIO(row.img["bytes"])).convert("RGB")
        images[i] = torch.from_numpy(np.array(img, dtype=np.uint8))
    labels = torch.tensor(df["label"].to_numpy(), dtype=torch.int64)

    cache = {"images": images, "labels": labels}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    return cache


class CIFAR10HF(Dataset):
    """Drop-in replacement for torchvision.datasets.CIFAR10: same
    (PIL-image-then-transform, label) interface, same `.targets` list."""

    def __init__(self, root, train: bool, download: bool = True, transform=None):
        root = Path(root)
        split = "train" if train else "test"
        parquet_path = root / "cifar10_hf" / f"{split}.parquet"
        cache_path = root / "cifar10_hf" / f"{split}_decoded.pt"

        if download and not parquet_path.exists():
            _download(HF_URLS[split], parquet_path)

        cache = _decode_parquet(parquet_path, cache_path)
        self.data = cache["images"]          # (N, 32, 32, 3) uint8
        self.targets = cache["labels"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        img = Image.fromarray(self.data[idx].numpy())
        target = self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, target


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2] / "data" / "raw"
    train_set = CIFAR10HF(root, train=True)
    test_set = CIFAR10HF(root, train=False)
    print(f"train: {len(train_set)}  test: {len(test_set)}")
    img, label = train_set[0]
    print(f"sample: label={label} ({CLASS_NAMES[label]})  image size={img.size}  mode={img.mode}")
