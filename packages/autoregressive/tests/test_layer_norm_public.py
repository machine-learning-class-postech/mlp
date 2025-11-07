from __future__ import annotations

import numpy as np
import pytest
from mlp.tree.functions import are_equal
from autoregressive import LayerNorm

RTOL = 1e-5
ATOL = 1e-6


@pytest.mark.public
def test_layer_norm_forward_shape(layer_norm: LayerNorm, input_data: np.ndarray):
    """Forward pass produces correct output shape."""
    output = layer_norm.forward(x=input_data)
    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 3)


@pytest.mark.public
def test_layer_norm_forward_computation(
    simple_weights: np.ndarray, simple_bias: np.ndarray
):
    """Forward pass computes expected normalized values with affine transform."""
    ln = LayerNorm(weights=simple_weights, bias=simple_bias)
    x = np.array([[1.0, 2.0, 3.0]])

    out = ln.forward(x=x)
    expected = np.asarray([[-1.124745, 0.2, 1.524745]])
    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.public
def test_layer_norm_backward(layer_norm: LayerNorm, input_data: np.ndarray):
    """Backward returns gradients with correct keys and shapes."""
    # Forward once to set internal caches
    _ = layer_norm.forward(x=input_data)

    grad_out = np.ones_like(input_data)
    param_grads, input_grads = layer_norm.backward(
        gradient_outputs=grad_out, x=input_data
    )

    assert isinstance(param_grads, dict)
    assert "weights" in param_grads and "bias" in param_grads
    assert isinstance(param_grads["weights"], np.ndarray)
    assert isinstance(param_grads["bias"], np.ndarray)

    assert isinstance(input_grads, dict)
    assert "x" in input_grads
    assert isinstance(input_grads["x"], np.ndarray)
    assert input_grads["x"].shape == input_data.shape


@pytest.mark.public
def test_layer_norm_zero_input(layer_norm: LayerNorm):
    """Zero input should output bias (since normalized term is zero)."""
    x_zero = np.zeros((1, 3))
    out = layer_norm.forward(x=x_zero)
    expected = np.broadcast_to(layer_norm.bias, x_zero.shape)
    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.public
def test_layer_norm_with_parameters(layer_norm: LayerNorm):
    """LayerNorm.with_parameters should return an equivalent module."""
    original_params = layer_norm.parameters
    new_ln = layer_norm.with_parameters(original_params)

    assert isinstance(new_ln, LayerNorm)
    new_params = new_ln.parameters
    assert are_equal(original_params, new_params, np.array_equal)


@pytest.fixture
def random_generator() -> np.random.Generator:
    """Seeded RNG for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def simple_weights() -> np.ndarray:
    """Simple scale vector of length 3."""
    return np.array([1.0, 1.0, 1.0])


@pytest.fixture
def simple_bias() -> np.ndarray:
    """Simple shift vector of length 3."""
    return np.array([0.1, 0.2, 0.3])


@pytest.fixture
def layer_norm(simple_weights: np.ndarray, simple_bias: np.ndarray) -> LayerNorm:
    """Create a LayerNorm instance for testing."""
    return LayerNorm(weights=simple_weights, bias=simple_bias)


@pytest.fixture
def input_data() -> np.ndarray:
    """Sample 2D input (batch, features)."""
    return np.array([[1.0, 2.0, 3.0]])
