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


class TinyShakespeareDataset(Dataset[tuple[np.ndarray, np.ndarray]]):
    """
    Tiny Shakespeare character-level LM dataset.

    Each item is a window:
        x: (T,) int64 chars
        y: (T,) int64 next chars
    """

    def __init__(
        self,
        inputs: np.ndarray,  # (N, T) int64
        targets: np.ndarray,  # (N, T) int64
        stoi: dict[str, int],
        itos: list[str],
    ):
        assert inputs.dtype == np.int64 and targets.dtype == np.int64
        assert inputs.shape == targets.shape and inputs.ndim == 2
        self._x = inputs
        self._y = targets
        self.stoi = dict(stoi)
        self.itos = list(itos)
        self.vocab_size = len(self.itos)

    def __len__(self) -> int:
        return self._x.shape[0]

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self._x[index], self._y[index]

    @staticmethod
    def _build_vocab(text: str) -> tuple[dict[str, int], list[str]]:
        chars = sorted(list(set(text)))
        itos = chars
        stoi = {ch: i for i, ch in enumerate(itos)}
        return stoi, itos

    @staticmethod
    def _encode(text: str, stoi: dict[str, int]) -> np.ndarray:
        arr = np.fromiter((stoi[c] for c in text), dtype=np.int64, count=len(text))
        return arr

    @staticmethod
    def _make_windows(
        tokens: np.ndarray,  # (L,)
        seq_len: int,
        stride: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Turn a long token array into (N, T) windows for x and y (teacher forcing).
        Only windows that fully fit (T+1) are kept.
        """
        L = tokens.shape[0]
        T = int(seq_len)
        if L < T + 1:
            return np.empty((0, T), dtype=np.int64), np.empty((0, T), dtype=np.int64)

        # number of windows with given stride that have T+1 length available
        N = 1 + (L - (T + 1)) // stride
        xs = np.empty((N, T), dtype=np.int64)
        ys = np.empty((N, T), dtype=np.int64)
        for i in range(N):
            start = i * stride
            chunk = tokens[start : start + T + 1]
            xs[i] = chunk[:-1]
            ys[i] = chunk[1:]
        return xs, ys

    @staticmethod
    def from_cache_directory_or_download(
        cache_directory: Path,
        *,
        seq_len: int = 128,
        stride: int = 128,
        train_ratio: float = 0.9,
        base_url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        filename: str = "tinyshakespeare_input.txt",
    ) -> tuple[TinyShakespeareDataset, TinyShakespeareDataset]:
        """
        Download (if needed) and build train/val datasets of windows.

        Returns:
            (train_dataset, val_dataset)
        """
        cache_directory.mkdir(parents=True, exist_ok=True)
        raw_path = cache_directory / filename
        if not raw_path.exists():
            urllib.request.urlretrieve(base_url, raw_path)

        text = raw_path.read_text(encoding="utf-8")
        stoi, itos = TinyShakespeareDataset._build_vocab(text)
        tokens = TinyShakespeareDataset._encode(text, stoi)  # (L,)

        L = tokens.shape[0]
        split = int(L * train_ratio)
        train_tok = tokens[:split]
        val_tok = tokens[split:]

        x_tr, y_tr = TinyShakespeareDataset._make_windows(train_tok, seq_len, stride)
        x_va, y_va = TinyShakespeareDataset._make_windows(val_tok, seq_len, stride)

        train_ds = TinyShakespeareDataset(x_tr, y_tr, stoi, itos)
        val_ds = TinyShakespeareDataset(x_va, y_va, stoi, itos)
        return train_ds, val_ds

    @staticmethod
    def collate(
        x: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Stack into (B, T) int64 inputs/targets (no one-hot).
        """
        inputs = np.stack([item[0] for item in x], axis=0).astype(np.int64, copy=False)
        targets = np.stack([item[1] for item in x], axis=0).astype(np.int64, copy=False)
        return inputs, targets
