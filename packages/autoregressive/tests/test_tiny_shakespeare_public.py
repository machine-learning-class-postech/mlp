from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from autoregressive import Adam, CrossEntropyLoss, Decoder
from mlp.dataset.prebuilts import TinyShakespeareDataset

ParamTree = dict[str, "ParamTree"] | list["ParamTree"] | np.ndarray

SEQ_LEN: int = 256
D_MODEL: int = 512
HIDDEN: int = 4 * D_MODEL
HEADS: int = 8  # must divide D_MODEL
LR: float = 1e-3
STEPS: int = 20
VAL_N: int = 512
BATCH_SIZE: int = 32
WARMUP: int = 4
CLIP_NORM: float = 1.0


LOSS = CrossEntropyLoss()


@pytest.mark.public
def test_decoder_trains_on_tiny_shakespeare(
    tiny_datasets: tuple[TinyShakespeareDataset, TinyShakespeareDataset],
    model_and_optimizer: tuple[Decoder, Adam],
) -> None:
    train_dataset, validation_dataset = tiny_datasets
    model, optimizer = model_and_optimizer
    random_generator = np.random.default_rng(0)

    validation_xs: list[np.ndarray] = []
    validation_ys: list[np.ndarray] = []
    for i in range(min(VAL_N, len(validation_dataset))):
        x, y = _split_xy(validation_dataset[i])
        validation_xs.append(x)
        validation_ys.append(y)

    val0_ce, val0_acc = validation_token_ce_and_accuracy(
        model, validation_xs, validation_ys
    )
    losses: list[float] = []

    for step in range(STEPS):
        idxs = random_generator.integers(0, len(train_dataset), size=BATCH_SIZE)
        acc_grads = _zeros_like_params(model.parameters)  # type: ignore
        batch_loss = 0.0

        for idx in idxs:
            x, y = _split_xy(train_dataset[int(idx)])  # (T,), (T,)
            logits = model.forward(x=x)  # (T, V)
            loss, grad_logits = token_ce_and_grad(logits, y)
            batch_loss += float(loss)

            grads, _ = model.backward(grad_logits, x=x)
            acc_grads = _add_params(acc_grads, grads)  # type: ignore

        acc_grads = _scale_params(acc_grads, 1.0 / float(BATCH_SIZE))
        losses.append(batch_loss / float(BATCH_SIZE))

        current_lr = float(LR * _lr_mult(step))
        if hasattr(optimizer, "_learning_rate"):
            optimizer._learning_rate = current_lr  # type: ignore[attr-defined]
        elif hasattr(optimizer, "learning_rate"):
            try:
                setattr(optimizer, "learning_rate", current_lr)
            except Exception:
                pass

        gnorm = _global_l2(acc_grads)
        if gnorm > CLIP_NORM:
            acc_grads = _scale_params(acc_grads, CLIP_NORM / (gnorm + 1e-8))

        new_params, optimizer = optimizer.step(model.parameters, acc_grads)
        model = model.with_parameters(new_params)

    val1_ce, val1_acc = validation_token_ce_and_accuracy(
        model, validation_xs, validation_ys
    )

    assert len(losses) > 0, "No losses recorded."
    assert np.std(np.asarray(losses, dtype=np.float64)) > 0, "Loss should vary."
    assert np.isfinite(val0_ce) and np.isfinite(val1_ce), (
        "Validation loss is not finite."
    )
    assert val1_ce < val0_ce, (
        f"Validation loss did not decrease: {val0_ce:.4f} → {val1_ce:.4f}"
    )
    assert val1_acc >= max(val0_acc, val0_acc + 0.01 - 1e-6), (
        f"Validation accuracy did not improve: {val0_acc:.3f} → {val1_acc:.3f}"
    )


# Simple tree helpers (local; no dependency on mlp.tree)


def _zeros_like_params(node: ParamTree) -> ParamTree:
    if isinstance(node, dict):
        return {k: _zeros_like_params(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_zeros_like_params(v) for v in node]
    return np.zeros_like(node)


def _add_params(a: ParamTree, b: ParamTree) -> ParamTree:
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: _add_params(a[k], b[k]) for k in a}
    if isinstance(a, list) and isinstance(b, list):
        return [_add_params(a[i], b[i]) for i in range(len(a))]
    return cast(np.ndarray, a) + cast(np.ndarray, b)


def _scale_params(node: ParamTree, s: float) -> ParamTree:
    if isinstance(node, dict):
        return {k: _scale_params(v, s) for k, v in node.items()}
    if isinstance(node, list):
        return [_scale_params(v, s) for v in node]
    return node * s


# Grad/LR helpers


def _global_l2(node: ParamTree) -> float:
    total = 0.0

    def _acc(t: ParamTree) -> None:
        nonlocal total
        if isinstance(t, dict):
            for v in t.values():
                _acc(v)
        elif isinstance(t, list):
            for v in t:
                _acc(v)
        else:
            arr = t
            total += float(np.sum(arr * arr))

    _acc(node)
    return float(np.sqrt(total))


def _lr_mult(step: int) -> float:
    # Linear warmup then cosine decay multiplier in [0,1]
    if step < WARMUP:
        return float((step + 1) / max(1, WARMUP))
    t = step - WARMUP
    T = max(1, STEPS - WARMUP)
    return float(0.5 * (1.0 + np.cos(np.pi * t / T)))


# Helper functions for decoding and greedy generation


def greedy_generate(
    model: Decoder, start_ids: np.ndarray, *, max_new_tokens: int, pe_cap: int
) -> np.ndarray:
    """
    Greedy decoding: repeatedly append argmax(logits[-1]).
    Stops at pe_cap to respect positional encoding length.
    """
    seq = np.asarray(start_ids, dtype=np.int64)
    for _ in range(max_new_tokens):
        if seq.shape[0] >= pe_cap:
            break
        logits = model.forward(x=seq)  # (T, V)
        nxt = int(np.argmax(logits[-1]))
        seq = np.concatenate([seq, np.array([nxt], dtype=np.int64)], axis=0)
    return seq


def _split_xy(sample: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    TinyShakespeare items are (x, y). Keep both as int64.
    """
    if isinstance(sample, (tuple, list)) and len(sample) >= 2:  # type: ignore
        x = np.asarray(sample[0], dtype=np.int64)
        y = np.asarray(sample[1], dtype=np.int64)
        return x, y
    x = np.asarray(sample, dtype=np.int64)
    if x.shape[0] >= 2:
        return x, x[1:]
    return x, x


def token_ce_and_grad(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """
    CE over all positions using teacher-forced targets y (shape (T,)).
    """
    T, _ = logits.shape
    assert y.shape == (T,)
    loss = LOSS.forward(logits, y)
    grad = LOSS.backward(logits, y)
    return float(loss), grad


def validation_token_ce_and_accuracy(
    model: Decoder, xs: list[np.ndarray], ys: list[np.ndarray]
) -> tuple[float, float]:
    """
    Mean CE and token accuracy over validation set (teacher forcing).
    Returns (mean_ce, mean_accuracy).
    """
    losses: list[float] = []
    correct = 0
    total = 0
    for x, y in zip(xs, ys):
        logits = model.forward(x=x)  # (T, V)
        loss, _ = token_ce_and_grad(logits, y)
        losses.append(loss)
        pred = np.argmax(logits, axis=1)  # (T,)
        correct += int(np.sum(pred.astype(np.int64) == y.astype(np.int64)))
        total += int(y.shape[0])
    mean_ce = float(np.mean(losses)) if losses else float("nan")
    accuracy = (correct / total) if total > 0 else float("nan")
    return mean_ce, float(accuracy)


@pytest.fixture(scope="module")
def cache_directory() -> Path:
    directory = Path(__file__).parent.parent.parent.parent / ".cache"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@pytest.fixture(scope="module")
def tiny_datasets(
    cache_directory: Path,
) -> tuple[TinyShakespeareDataset, TinyShakespeareDataset]:
    """
    Download (or load from cache) Tiny Shakespeare and return train/val windows.
    """
    train_dataset, validation_dataset = (
        TinyShakespeareDataset.from_cache_directory_or_download(
            cache_directory=cache_directory,
            seq_len=SEQ_LEN,
            stride=1,  # maximal training signal; overlapping windows
            train_ratio=0.9,
        )
    )
    return train_dataset, validation_dataset


@pytest.fixture
def model_and_optimizer(
    tiny_datasets: tuple[TinyShakespeareDataset, TinyShakespeareDataset],
) -> tuple[Decoder, Adam]:
    random_generator = np.random.default_rng(0)
    train_dataset, _ = tiny_datasets

    vocab_size = int(getattr(train_dataset, "vocab_size", 0))
    if vocab_size <= 0:
        x0, _ = _split_xy(train_dataset[0])
        vocab_size = int(np.max(x0) + 1)

    model = Decoder.create(
        input_size=vocab_size,
        output_size=D_MODEL,
        hidden_size=HIDDEN,
        random_generator=random_generator,
        num_heads=HEADS,
        seq_len=SEQ_LEN,
        causal=True,
    )
    optimizer = Adam(learning_rate=LR, beta1=0.9, beta2=0.98, eps=1e-8)
    return model, optimizer
