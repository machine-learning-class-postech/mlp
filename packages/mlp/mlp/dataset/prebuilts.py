from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np

from .protocols import Dataset


class MnistDataset(Dataset[tuple[np.ndarray, np.ndarray]]):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self._images = images
        self._labels = labels

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self._images[index], self._labels[index]

    @staticmethod
    def from_cache_directory_or_download(
        cache_directory: Path,
        base_url: str = "https://raw.githubusercontent.com/fgnt/mnist/master",
    ) -> tuple[MnistDataset, MnistDataset]:
        train_images_ubyte = cache_directory / "train-images-idx3-ubyte.gz"
        train_labels_ubyte = cache_directory / "train-labels-idx1-ubyte.gz"
        test_images_ubyte = cache_directory / "t10k-images-idx3-ubyte.gz"
        test_labels_ubyte = cache_directory / "t10k-labels-idx1-ubyte.gz"

        if not train_images_ubyte.exists():
            urllib.request.urlretrieve(
                f"{base_url}/train-images-idx3-ubyte.gz",
                train_images_ubyte,
            )
        if not train_labels_ubyte.exists():
            urllib.request.urlretrieve(
                f"{base_url}/train-labels-idx1-ubyte.gz",
                train_labels_ubyte,
            )
        if not test_images_ubyte.exists():
            urllib.request.urlretrieve(
                f"{base_url}/t10k-images-idx3-ubyte.gz",
                test_images_ubyte,
            )
        if not test_labels_ubyte.exists():
            urllib.request.urlretrieve(
                f"{base_url}/t10k-labels-idx1-ubyte.gz",
                test_labels_ubyte,
            )
        with gzip.open(train_images_ubyte, "rb") as f:
            train_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(
                -1, 28, 28
            )
        with gzip.open(train_labels_ubyte, "rb") as f:
            train_labels = np.frombuffer(f.read(), np.uint8, offset=8)
        with gzip.open(test_images_ubyte, "rb") as f:
            test_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(
                -1, 28, 28
            )
        with gzip.open(test_labels_ubyte, "rb") as f:
            test_labels = np.frombuffer(f.read(), np.uint8, offset=8)
        train_images = train_images.reshape(-1, 28 * 28).astype(np.float64) / 255.0
        test_images = test_images.reshape(-1, 28 * 28).astype(np.float64) / 255.0
        train_labels = train_labels.astype(np.int64)
        test_labels = test_labels.astype(np.int64)
        train_dataset = MnistDataset(train_images, train_labels)
        test_dataset = MnistDataset(test_images, test_labels)
        return train_dataset, test_dataset

    @staticmethod
    def collate(
        x: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray]:
        images = np.stack([item[0] for item in x], axis=0)
        labels = np.eye(10, dtype=np.float64)[[item[1] for item in x]]
        return images, labels


class EquationDataset(Dataset[tuple[np.ndarray, np.ndarray]]):
    def __init__(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        length: int | None = None,
        low: float | None = None,
        high: float | None = None,
        random_generator: np.random.Generator | None = None,
        xs: np.ndarray | None = None,
    ):
        if xs is None:
            assert length is not None, "Either xs or length must be provided."
            assert low is not None and high is not None, (
                "Either xs or both low and high must be provided."
            )
            assert random_generator is not None, (
                "Either xs or random_generator must be provided."
            )
            self._xs = random_generator.uniform(low, high, size=(length, 1)).astype(
                np.float64
            )
        else:
            self._xs = xs

        self._y = f(self._xs).astype(np.float64)

    def __len__(self) -> int:
        return len(self._xs)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self._xs[index], self._y[index]

    @staticmethod
    def collate(
        x: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray]:
        xs = np.stack([item[0] for item in x], axis=0)
        ys = np.stack([item[1] for item in x], axis=0)
        return xs, ys
