from __future__ import annotations

import numpy as np
import pytest
from supervised_learning import ReLU


@pytest.mark.public
def test_relu_create():
    """Test ReLU.create factory method."""
    relu = ReLU.create()
    assert isinstance(relu, ReLU)


@pytest.mark.public
def test_relu_parameters(relu_module: ReLU):
    """Test that ReLU has no parameters."""
    parameters = relu_module.parameters
    assert parameters is not None


@pytest.mark.public
def test_relu_forward_positive_inputs(relu_module: ReLU, positive_input: np.ndarray):
    """Test ReLU forward pass with positive inputs."""
    output = relu_module.forward(x=positive_input)

    assert isinstance(output, np.ndarray)
    assert output.shape == positive_input.shape
    np.testing.assert_array_equal(output, positive_input)


@pytest.mark.public
def test_relu_forward_negative_inputs(relu_module: ReLU, negative_input: np.ndarray):
    """Test ReLU forward pass with negative inputs."""
    output = relu_module.forward(x=negative_input)

    assert isinstance(output, np.ndarray)
    assert output.shape == negative_input.shape
    expected = np.zeros_like(negative_input)
    np.testing.assert_array_equal(output, expected)


@pytest.mark.public
def test_relu_forward_mixed_inputs(relu_module: ReLU, mixed_input: np.ndarray):
    """Test ReLU forward pass with mixed positive/negative inputs."""
    output = relu_module.forward(x=mixed_input)

    assert isinstance(output, np.ndarray)
    assert output.shape == mixed_input.shape

    expected = np.array([[0.0, 1.5, 0.0, 3.0, 0.0]])
    np.testing.assert_array_equal(output, expected)


@pytest.mark.public
def test_relu_forward_zero_input(relu_module: ReLU):
    """Test ReLU forward pass with zero input."""
    zero_input = np.array([[0.0, 0.0, 0.0]])
    output = relu_module.forward(x=zero_input)

    expected = np.array([[0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(output, expected)


@pytest.fixture
def relu_module() -> ReLU:
    """Create a ReLU module instance for testing."""
    return ReLU.create()


@pytest.fixture
def positive_input() -> np.ndarray:
    """Positive input data for testing."""
    return np.array([[1.0, 2.5, 3.7, 0.1]])


@pytest.fixture
def negative_input() -> np.ndarray:
    """Negative input data for testing."""
    return np.array([[-1.0, -2.5, -3.7, -0.1]])


@pytest.fixture
def mixed_input() -> np.ndarray:
    """Mixed positive and negative input data."""
    return np.array([[-2.0, 1.5, -0.5, 3.0, 0.0]])
