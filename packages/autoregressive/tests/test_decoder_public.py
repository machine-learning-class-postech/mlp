from __future__ import annotations

import numpy as np
import pytest
from autoregressive import (
    Decoder,
    DecoderLayer,
    Embedding,
    FeedForward,
    LayerNorm,
    Linear,
    MultiHeadAttention,
    PositionalEncoding,
    ScaledSelfAttention,
)

RTOL = 5e-4
ATOL = 1e-8

# DecoderLayer — public tests


@pytest.mark.public
def test_decoder_layer_forward_shape_public(
    decoder_layer: DecoderLayer, decoder_x: np.ndarray
):
    """Forward returns (T, D) for given (T, D)."""
    y = decoder_layer.forward(x=decoder_x)
    assert isinstance(y, np.ndarray)
    assert y.shape == decoder_x.shape


@pytest.mark.public
def test_decoder_layer_forward_values_public(
    decoder_layer: DecoderLayer, decoder_x: np.ndarray
):
    y = decoder_layer.forward(x=decoder_x)
    expected = np.asarray(
        [
            [0.2744272, -0.37935347, 0.98637596, -1.0350688, 1.44147367, -1.28785456],
            [-0.99983259, 0.30165748, -0.2027053, 0.70493585, -1.37716446, 1.57310903],
            [-0.19593548, 0.34380614, -0.86484723, 0.56317873, -1.46686091, 1.62065876],
            [-0.22565146, 0.40576174, 0.88069407, -1.23159794, 1.4039828, -1.23318919],
        ]
    )
    np.testing.assert_allclose(y, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.public
def test_decoder_layer_with_parameters_public(
    decoder_layer: DecoderLayer, decoder_x: np.ndarray
):
    """with_parameters copies weights and preserves the forward output."""
    params = decoder_layer.parameters
    copied = decoder_layer.with_parameters(params)

    y1 = decoder_layer.forward(x=decoder_x)
    y2 = copied.forward(x=decoder_x)
    np.testing.assert_allclose(y1, y2, rtol=1e-7, atol=1e-8)


@pytest.mark.public
def test_decoder_layer_create_factory_public(random_generator: np.random.Generator):
    """Factory creates a valid DecoderLayer that runs forward."""
    layer = DecoderLayer.create(
        embed_dim=8, num_heads=2, hidden_size=4 * 8, random_generator=random_generator
    )
    T, D = 3, 8
    x = random_generator.normal(size=(T, D))
    y = layer.forward(x=x)
    assert y.shape == (T, D)


# Decoder — public tests


@pytest.mark.public
def test_decoder_forward_shape_public(decoder: Decoder, token_ids: np.ndarray):
    """Decoder forward returns logits (T, V)."""
    logits = decoder.forward(x=token_ids)
    T = token_ids.shape[0]
    V = decoder.lm_head.bias.shape[0]
    assert isinstance(logits, np.ndarray)
    assert logits.shape == (T, V)


@pytest.mark.public
def test_decoder_forward_values_public(decoder: Decoder, token_ids: np.ndarray):
    y = decoder.forward(x=token_ids)
    expected = np.asarray(
        [
            [
                -0.0027109,
                -0.00271089,
                -0.00271089,
                -0.00271089,
                -0.00271089,
                -0.00271089,
                -0.00271089,
                -0.00271089,
                -0.0027109,
            ],
            [
                -0.00813035,
                -0.00813035,
                -0.00813035,
                -0.00813035,
                -0.00813035,
                -0.00813035,
                -0.00813035,
                -0.00813035,
                -0.00813035,
            ],
            [
                -0.00551611,
                -0.00551611,
                -0.00551611,
                -0.00551611,
                -0.00551611,
                -0.0055161,
                -0.0055161,
                -0.00551611,
                -0.00551611,
            ],
        ]
    )
    np.testing.assert_allclose(y, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.public
def test_decoder_with_parameters_public(decoder: Decoder, token_ids: np.ndarray):
    """Copying parameters via with_parameters preserves outputs."""
    params = decoder.parameters
    copied = decoder.with_parameters(params)

    y1 = decoder.forward(x=token_ids)
    y2 = copied.forward(x=token_ids)
    np.testing.assert_allclose(y1, y2, rtol=1e-7, atol=1e-8)


@pytest.mark.public
def test_decoder_create_factory_public(random_generator: np.random.Generator):
    """Factory creates a Decoder that produces logits with correct shape."""
    dec = Decoder.create(
        input_size=13,
        output_size=8,
        hidden_size=4 * 8,
        random_generator=random_generator,
        num_heads=2,
    )
    T = 5
    tok = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    logits = dec.forward(x=tok)
    assert logits.shape == (T, dec.lm_head.bias.shape[0])


# Fixtures — placed at end (public)


@pytest.fixture
def random_generator() -> np.random.Generator:
    return np.random.default_rng(2026)


@pytest.fixture
def decoder_layer() -> DecoderLayer:
    return build_decoder_layer_fixed(d_model=6, num_heads=2)


@pytest.fixture
def decoder_x() -> np.ndarray:
    return np.asarray(
        [
            [0.10, -0.20, 0.30, -0.40, 0.50, -0.60],
            [-0.25, 0.15, -0.05, 0.35, -0.45, 0.55],
            [0.05, 0.10, -0.15, 0.20, -0.25, 0.30],
            [0.00, 0.05, 0.10, -0.15, 0.20, -0.25],
        ]
    )


@pytest.fixture
def decoder() -> Decoder:
    return build_decoder_fixed(vocab_size=9, d_model=6, num_heads=2, num_layers=2)


@pytest.fixture
def token_ids() -> np.ndarray:
    return np.array([1, 0, 3], dtype=np.int32)


# Deterministic weights & builders (decoder-only)


def _lin(
    out_dim: int, in_dim: int, scale: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    W = (
        np.arange(out_dim * in_dim).reshape(out_dim, in_dim) - (out_dim * in_dim) / 2
    ) * (scale / (out_dim * in_dim))
    b = np.zeros((out_dim,))
    return W, b


def build_decoder_layer_fixed(d_model: int = 6, num_heads: int = 2) -> DecoderLayer:
    # Per-head dims
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    d_sub = d_model // num_heads  # treat as d_v

    # Build heads: W_q/W_k/W_v are (in=d_model, out=d_sub) with simple deterministic patterns
    heads: list[ScaledSelfAttention] = []
    for h in range(num_heads):
        base = h + 1
        W_q = (base * np.eye(d_model, d_sub)) / d_sub
        W_k = ((base + 1) * np.eye(d_model, d_sub)) / d_sub
        W_v = ((base + 2) * np.eye(d_model, d_sub)) / d_sub
        heads.append(ScaledSelfAttention(W_q=W_q, W_k=W_k, W_v=W_v))

    # W_out: (h*d_sub, d_model) == (d_model, d_model). Use near-identity but slightly perturbed.
    W_out = np.eye(d_model)
    W_out += np.tril(np.ones((d_model, d_model)), k=-1) * (1e-2)

    mha = MultiHeadAttention(heads=heads, W_out=W_out)

    # LayerNorms: gamma=1, beta=0
    norm1 = LayerNorm(weights=np.ones((d_model,)), bias=np.zeros((d_model,)))
    norm2 = LayerNorm(weights=np.ones((d_model,)), bias=np.zeros((d_model,)))

    # FFN: two Linear layers with fixed weights
    H = 4 * d_model
    W1, b1 = _lin(H, d_model, scale=0.2)
    W2, b2 = _lin(d_model, H, scale=0.2)
    linear1 = Linear(weights=W1, bias=b1)
    linear2 = Linear(weights=W2, bias=b2)
    ffn = FeedForward(layer1=linear1, layer2=linear2)

    return DecoderLayer(mha=mha, norm1=norm1, ffn=ffn, norm2=norm2)


def build_decoder_fixed(
    vocab_size: int = 9, d_model: int = 6, num_heads: int = 2, num_layers: int = 2
) -> Decoder:
    # Token embedding: deterministic table
    emb_table = np.arange(vocab_size * d_model).reshape(vocab_size, d_model) / (
        vocab_size * d_model
    )
    token = Embedding(weight=emb_table)

    # Positional encoding: create via constructor (seq_len, d_model, base)
    L = 512
    posenc = PositionalEncoding(seq_len=L, d_model=d_model, base=10000.0)

    # Layers
    layers = [
        build_decoder_layer_fixed(d_model=d_model, num_heads=num_heads)
        for _ in range(num_layers)
    ]

    # LM head: fixed linear D -> V
    W_head, b_head = _lin(vocab_size, d_model, scale=0.15)
    lm_head = Linear(weights=W_head, bias=b_head)

    return Decoder(
        token_embedding=token, pos_encoding=posenc, layers=layers, lm_head=lm_head
    )
