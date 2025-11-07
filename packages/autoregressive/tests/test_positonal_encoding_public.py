from __future__ import annotations

import numpy as np
import pytest

from autoregressive import PositionalEncoding

RTOL = 1e-6
ATOL = 1e-6


@pytest.mark.public
def test_positional_encoding_forward_shape_and_values(
    positional_encoding: PositionalEncoding, toy_input: np.ndarray
):
    x = toy_input
    y = positional_encoding.forward(x=x)

    # shape check
    assert isinstance(y, np.ndarray)
    assert y.shape == x.shape

    # value check
    expected = np.asarray(
        [
            [0.0, 1.1, 0.2, 1.3],
            [1.241471, 1.040302, 0.61, 1.69995],
            [1.709297, 0.483853, 1.019999, 2.0998],
        ]
    )
    np.testing.assert_allclose(y, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.public
def test_positional_encoding_with_parameters_public(
    positional_encoding: PositionalEncoding, toy_input: np.ndarray
):
    """with_parameters({}) clones hyperparameters and preserves outputs."""
    positional_encoding_clone = positional_encoding.with_parameters({})
    y1 = positional_encoding.forward(x=toy_input)
    y2 = positional_encoding_clone.forward(x=toy_input)
    np.testing.assert_allclose(y1, y2, rtol=RTOL, atol=ATOL)


@pytest.mark.public
def test_positional_encoding_create_factory_public():
    """create(L, D, _) produces table of shape (L, D) and forward adds PE correctly."""
    L, D = 7, 4
    positional_encoding = PositionalEncoding.create(
        input_size=L, output_size=D, random_generator=np.random.default_rng(0)
    )
    assert isinstance(positional_encoding, PositionalEncoding)

    x = np.zeros((L, D))
    y = positional_encoding.forward(x=x)

    expected = np.asarray(
        [
            [0.0, 1.0, 0.0, 1.0],
            [0.84147098, 0.54030231, 0.01, 0.99995],
            [0.90929743, -0.41614684, 0.019999, 0.9998],
            [0.14112001, -0.9899925, 0.0299995, 0.99955],
            [-0.7568025, -0.65364362, 0.03999933, 0.9992],
            [-0.95892427, 0.28366219, 0.04999917, 0.99875],
            [-0.2794155, 0.96017029, 0.059999, 0.9982],
        ]
    )
    print(y)
    np.testing.assert_allclose(y, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.public
def test_positional_encoding_parameters_empty_public(
    positional_encoding: PositionalEncoding,
):
    """parameters property should be an empty dict for non-learnable module."""
    params = positional_encoding.parameters
    assert isinstance(params, dict)
    assert params == {}


@pytest.fixture
def positional_encoding() -> PositionalEncoding:
    """PositionalEncoding with seq_len=10, d_model=6."""
    return PositionalEncoding(seq_len=10, d_model=4, base=10000.0)


@pytest.fixture
def toy_input() -> np.ndarray:
    """Simple (T, D) input with T<=seq_len and D==d_model."""
    T, D = 3, 4
    return np.arange(T * D).reshape(T, D) / 10.0
