from __future__ import annotations

import numpy as np
import pytest
from mlp.tree.functions import are_equal
from supervised_learning import Linear


@pytest.mark.public
def test_linear_forward_shape(linear_module: Linear, input_data: np.ndarray):
    """Test that forward pass produces correct output shape."""
    output = linear_module.forward(x=input_data)
    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 2)


@pytest.mark.public
def test_linear_forward_computation(
    simple_weights: np.ndarray, simple_bias: np.ndarray
):
    """Test that forward pass computes correct values."""
    linear = Linear(weights=simple_weights, bias=simple_bias)
    input_data = np.array([[1.0, 2.0, 3.0]])

    output = linear.forward(x=input_data)

    expected = np.array([[14.1, 32.2]])
    np.testing.assert_allclose(output, expected, rtol=1e-10)


@pytest.mark.public
def test_linear_backward(linear_module: Linear, input_data: np.ndarray):
    """Test backward pass returns correct gradient structure."""
    gradient_outputs = np.array([[1.0, 1.0]])

    parameter_gradients, input_gradients = linear_module.backward(
        gradient_outputs=gradient_outputs, x=input_data
    )

    assert parameter_gradients is not None
    assert isinstance(input_gradients, dict)
    assert "x" in input_gradients
    assert isinstance(input_gradients["x"], np.ndarray)


@pytest.mark.public
def test_linear_zero_input(linear_module: Linear):
    """Test Linear module with zero input."""
    output = linear_module.forward(x=np.zeros(3))

    expected = np.array([0.1, 0.2])
    np.testing.assert_allclose(output, expected, rtol=1e-10)


@pytest.mark.public
def test_linear_with_parameters(linear_module: Linear):
    """Test Linear.with_parameters constructor."""

    original_params = linear_module.parameters
    new_linear = linear_module.with_parameters(original_params)

    assert isinstance(new_linear, Linear)
    new_params = new_linear.parameters
    assert are_equal(original_params, new_params, np.array_equal)


@pytest.fixture
def random_generator() -> np.random.Generator:
    """Provide a seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def simple_weights() -> np.ndarray:
    """Simple 2x3 weight matrix for testing."""
    return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


@pytest.fixture
def simple_bias() -> np.ndarray:
    """Simple bias vector for testing."""
    return np.array([0.1, 0.2])


@pytest.fixture
def linear_module(simple_weights: np.ndarray, simple_bias: np.ndarray) -> Linear:
    """Create a Linear module instance for testing."""
    return Linear(weights=simple_weights, bias=simple_bias)


@pytest.fixture
def input_data() -> np.ndarray:
    """Sample input data for testing."""
    return np.array([[1.0, 2.0, 3.0]])
