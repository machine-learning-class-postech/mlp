from __future__ import annotations

import numpy as np
import pytest
from autoregressive import FeedForward, Linear


@pytest.mark.public
def test_ffnn_forward_shape(ffnn_module: FeedForward, input_data: np.ndarray):
    """Forward should return (B, out_dim) for batched inputs."""
    y = ffnn_module.forward(x=input_data)
    assert isinstance(y, np.ndarray)
    assert y.shape == (2, 2)


@pytest.mark.public
def test_ffnn_forward_computation(
    simple_w1: np.ndarray,
    simple_b1: np.ndarray,
    simple_w2: np.ndarray,
    simple_b2: np.ndarray,
):
    """Numerical forward check with fixed small weights."""
    layer1 = Linear(weights=simple_w1, bias=simple_b1)
    layer2 = Linear(weights=simple_w2, bias=simple_b2)
    ffnn = FeedForward(layer1, layer2)

    x = np.array([[1.0, 2.0, 3.0]])
    y = ffnn.forward(x=x)
    expected = np.array([[0.1, 18.7]])
    np.testing.assert_allclose(y, expected, rtol=1e-10)


@pytest.mark.public
def test_ffnn_backward_structure(ffnn_module: FeedForward, input_data: np.ndarray):
    """Backward should return (param_grads dict, {'x': grad})."""
    g_out = np.ones((2, 2))
    params, grads = ffnn_module.backward(g_out, x=input_data)
    assert isinstance(params, dict)
    assert "layer1" in params and "layer2" in params
    assert isinstance(grads, dict) and "x" in grads
    assert isinstance(grads["x"], np.ndarray)
    assert grads["x"].shape == input_data.shape


@pytest.mark.public
def test_ffnn_with_parameters_roundtrip(
    ffnn_module: FeedForward, input_data: np.ndarray
):
    """with_parameters should clone weights producing identical outputs."""
    params = ffnn_module.parameters
    clone = ffnn_module.with_parameters(params)
    y1 = ffnn_module.forward(x=input_data)
    y2 = clone.forward(x=input_data)
    np.testing.assert_allclose(y1, y2, rtol=1e-12)


# Fixtures


@pytest.fixture
def simple_w1() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, -1.0],
            [2.0, 1.0, 0.0],
            [0.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )


@pytest.fixture
def simple_b1() -> np.ndarray:
    return np.array([0.5, -0.5, 0.0, 1.0])


@pytest.fixture
def simple_w2() -> np.ndarray:
    return np.array(
        [
            [1.0, 2.0, 0.0, -1.0],
            [0.0, 1.0, 1.0, 2.0],
        ]
    )


@pytest.fixture
def simple_b2() -> np.ndarray:
    return np.array([0.1, 0.2])


@pytest.fixture
def ffnn_module(
    simple_w1: np.ndarray,
    simple_b1: np.ndarray,
    simple_w2: np.ndarray,
    simple_b2: np.ndarray,
) -> FeedForward:
    return FeedForward(Linear(simple_w1, simple_b1), Linear(simple_w2, simple_b2))


@pytest.fixture
def input_data() -> np.ndarray:
    # Two examples (batch=2), input_dim=3
    return np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
