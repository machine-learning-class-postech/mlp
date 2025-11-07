from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from autoregressive import MultiHeadAttention, ScaledSelfAttention

RTOL = 1e-6
ATOL = 1e-6

# ScaledHeadAttention test


@pytest.mark.public
def test_scaled_self_attention_forward_shape(
    sdp_attention: ScaledSelfAttention, toy_qkv: dict[str, np.ndarray]
):
    out = sdp_attention.forward(**toy_qkv)
    # (T, d_v) where T=2, d_v=2
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 2)


@pytest.mark.public
def test_scaled_self_attention_forward_values(
    sdp_attention: ScaledSelfAttention, toy_qkv: dict[str, np.ndarray]
):
    """Check forward numerics against manual computation."""

    out = sdp_attention.forward(**toy_qkv)
    expected = np.asarray([[0.794403, 0.09573], [0.696165, 0.215798]])

    np.testing.assert_allclose(out, expected.astype(out.dtype), rtol=RTOL, atol=ATOL)


@pytest.mark.public
def test_scaled_self_attention_forward_values_causal(
    simple_weights: dict[str, np.ndarray], toy_qkv: dict[str, np.ndarray]
):
    """Check single-head forward numerics with **causal=True**.

    For T=2:
      - i=0 can only attend to j=0 → output equals v[0]
      - i=1 has no future positions to block → same as non-causal result
    """
    attention = ScaledSelfAttention(
        W_q=simple_weights["W_q"],
        W_k=simple_weights["W_k"],
        W_v=simple_weights["W_v"],
        causal=True,
    )

    out = attention.forward(**toy_qkv)
    expected = np.asarray(
        [
            [0.300000, 0.700000],  # attends only to v[0]
            [0.696165, 0.215798],  # identical to non-causal for i=1
        ]
    )

    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out, expected.astype(out.dtype), rtol=RTOL, atol=ATOL)


@pytest.mark.public
def test_scaled_self_attention_with_parameters(
    sdp_attention: ScaledSelfAttention, toy_qkv: dict[str, np.ndarray]
):
    parameters = sdp_attention.parameters
    new_attention = sdp_attention.with_parameters(parameters)
    out1 = sdp_attention.forward(**toy_qkv)
    out2 = new_attention.forward(**toy_qkv)
    np.testing.assert_allclose(out1, out2, rtol=RTOL, atol=ATOL)


@pytest.mark.public
def test_scaled_self_attention_create(random_generator: np.random.Generator):
    attention = ScaledSelfAttention.create(
        input_size=4, output_size=6, random_generator=random_generator
    )
    parameters = attention.parameters
    assert isinstance(parameters, dict)
    W_q = cast(np.ndarray, parameters["W_q"])  # help Pylance infer type
    W_k = cast(np.ndarray, parameters["W_k"])  # help Pylance infer type
    W_v = cast(np.ndarray, parameters["W_v"])  # help Pylance infer type
    assert W_q.shape == (4, 6)
    assert W_k.shape == (4, 6)
    assert W_v.shape == (4, 6)


# MultiHeadAttention test


@pytest.mark.public
def test_multihead_attention_forward_values(
    simple_weights: dict[str, np.ndarray], toy_qkv: dict[str, np.ndarray]
):
    """Check MHA forward numerics against manual expectation using two identical heads.

    Setup:
      - Two heads with the same (W_q, W_k, W_v) as the single-head test
      - d_in=2, d_head=d_v=2, num_heads=2 -> embed_dim=4
      - W_out = I_4 so the output is just the concatenation of per-head outputs
      - Expected = [Y_single | Y_single], where Y_single is the single-head expected output
    """
    # Build two identical heads
    head1 = ScaledSelfAttention(
        W_q=simple_weights["W_q"],
        W_k=simple_weights["W_k"],
        W_v=simple_weights["W_v"],
    )
    head2 = ScaledSelfAttention(
        W_q=simple_weights["W_q"],
        W_k=simple_weights["W_k"],
        W_v=simple_weights["W_v"],
    )
    w_out = np.eye(4)  # (2 heads * d_v=2) -> embed_dim=4

    multi_head_attention = MultiHeadAttention(heads=[head1, head2], W_out=w_out)

    # Forward
    out = multi_head_attention.forward(**toy_qkv)  # (T=2, E=4)

    # Expected: concatenate the verified single-head expected along the last dim
    expected_single = np.asarray(
        [
            [0.794403, 0.09573],
            [0.696165, 0.215798],
        ]
    )
    expected = np.concatenate([expected_single, expected_single], axis=1)  # (2,4)

    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 4)
    np.testing.assert_allclose(out, expected.astype(out.dtype), rtol=RTOL, atol=ATOL)


@pytest.mark.public
def test_multihead_attention_forward_values_causal(
    simple_weights: dict[str, np.ndarray], toy_qkv: dict[str, np.ndarray]
):
    """Check MHA forward numerics with **causal=True** using two identical heads.

    Setup:
      - Two heads with the same (W_q, W_k, W_v) as the single-head test
      - Both heads use causal masking (strict upper-triangular mask)
      - d_in=2, d_head=d_v=2, num_heads=2 -> embed_dim=4
      - W_out = I_4 so the output is just the concatenation of per-head outputs
      - Expected = [Y_single_causal | Y_single_causal], where:
          * for T=2, the first query attends only to key 0 → output equals v[0]
    """
    # Two identical causal heads
    head1 = ScaledSelfAttention(
        W_q=simple_weights["W_q"],
        W_k=simple_weights["W_k"],
        W_v=simple_weights["W_v"],
        causal=True,
    )
    head2 = ScaledSelfAttention(
        W_q=simple_weights["W_q"],
        W_k=simple_weights["W_k"],
        W_v=simple_weights["W_v"],
        causal=True,
    )
    W_out = np.eye(4)  # (2 heads * d_v=2) -> embed_dim=4

    mha = MultiHeadAttention(heads=[head1, head2], W_out=W_out)

    # Forward
    out = mha.forward(**toy_qkv)  # (T=2, E=4)

    # Expected (per-head, causal):
    #  i=0 attends only to j=0 → output = v[0] = [0.3, 0.7]
    #  i=1 attends to j∈{0,1} (no future block) → matches masked single-head value
    expected_single_causal = np.asarray(
        [
            [0.300000, 0.700000],
            [0.696165, 0.215798],
        ]
    )
    expected = np.concatenate(
        [expected_single_causal, expected_single_causal], axis=1
    )  # (2,4)

    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 4)
    np.testing.assert_allclose(out, expected.astype(out.dtype), rtol=RTOL, atol=ATOL)


@pytest.mark.public
def test_multihead_attention_forward_shape(random_generator: np.random.Generator):
    # d_model = 4, num_heads = 2, d_head = 2
    mha = MultiHeadAttention.create(
        input_size=4, embed_dim=4, num_heads=2, random_generator=random_generator
    )

    T = 3
    x = random_generator.normal(size=(T, 4))
    out = mha.forward(q=x, k=x, v=x)

    assert isinstance(out, np.ndarray)
    assert out.shape == (T, 4)


@pytest.mark.public
def test_multihead_attention_parameter_tree_shapes(
    random_generator: np.random.Generator,
):
    embed_dim = 6
    num_heads = 3  # d_head = 2
    multi_head_attention = MultiHeadAttention.create(
        input_size=6,
        embed_dim=embed_dim,
        num_heads=num_heads,
        random_generator=random_generator,
    )

    parameters = multi_head_attention.parameters
    assert isinstance(parameters, dict)
    W_out = cast(np.ndarray, parameters["W_out"])  # help Pylance infer type
    assert "W_out" in parameters and W_out.shape == (embed_dim, embed_dim)
    for i in range(num_heads):
        head_params = parameters[f"head_{i}"]
        assert isinstance(head_params, dict)
        W_q = cast(np.ndarray, head_params["W_q"])  # help Pylance infer type
        W_k = cast(np.ndarray, head_params["W_k"])  # help Pylance infer type
        W_v = cast(np.ndarray, head_params["W_v"])  # help Pylance infer type
        assert W_q.shape == (6, embed_dim // num_heads)
        assert W_k.shape == (6, embed_dim // num_heads)
        assert W_v.shape == (6, embed_dim // num_heads)


@pytest.mark.public
def test_multihead_attention_with_parameters(random_generator: np.random.Generator):
    mha = MultiHeadAttention.create(
        input_size=4, embed_dim=4, num_heads=2, random_generator=random_generator
    )
    params = mha.parameters
    new_mha = mha.with_parameters(params)

    x = random_generator.normal(size=(2, 4))
    out1 = mha.forward(q=x, k=x, v=x)
    out2 = new_mha.forward(q=x, k=x, v=x)

    np.testing.assert_allclose(out1, out2, rtol=RTOL, atol=ATOL)


@pytest.fixture
def random_generator() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def simple_weights() -> dict[str, np.ndarray]:
    """Deterministic small projection matrices (d_in=2, d_k=d_v=2)."""
    W_q = np.array([[0.5, -0.25], [0.1, 0.3]])
    W_k = np.array([[0.2, 0.4], [-0.1, 0.6]])
    W_v = np.array([[1.0, 0.0], [0.0, 1.0]])
    return {"W_q": W_q, "W_k": W_k, "W_v": W_v}


@pytest.fixture
def toy_qkv() -> dict[str, np.ndarray]:
    """Toy inputs with T=2, d_in=2."""
    q = np.array([[1.0, 2.0], [0.5, -1.0]])
    k = np.array([[1.5, -0.5], [2.0, 0.5]])
    v = np.array([[0.3, 0.7], [1.2, -0.4]])
    return {"q": q, "k": k, "v": v}


@pytest.fixture
def sdp_attention(simple_weights: dict[str, np.ndarray]) -> ScaledSelfAttention:
    return ScaledSelfAttention(
        W_q=simple_weights["W_q"],
        W_k=simple_weights["W_k"],
        W_v=simple_weights["W_v"],
    )
