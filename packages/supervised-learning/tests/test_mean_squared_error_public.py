from __future__ import annotations

import numpy as np
import pytest
from supervised_learning import MeanSquaredError


@pytest.mark.public
def test_mean_squared_error_forward_basic(
    mean_squared_error_loss: MeanSquaredError,
    simple_predictions: np.ndarray,
    simple_targets: np.ndarray,
):
    """Test basic MSE forward pass."""
    loss_value = mean_squared_error_loss.forward(
        predictions=simple_predictions, targets=simple_targets
    )

    assert isinstance(loss_value, (float, np.floating))
    assert loss_value >= 0.0


@pytest.mark.public
def test_mean_squared_error_forward_computation(
    mean_squared_error_loss: MeanSquaredError,
):
    """Test MSE forward pass computation correctness."""
    predictions = np.array([2.0, 4.0, 6.0])
    targets = np.array([1.0, 3.0, 5.0])

    loss_value = mean_squared_error_loss.forward(
        predictions=predictions, targets=targets
    )

    expected = 1.0
    assert np.isclose(loss_value, expected, rtol=1e-10)


@pytest.mark.public
def test_mean_squared_error_forward_perfect_prediction(
    mean_squared_error_loss: MeanSquaredError,
):
    """Test MSE with perfect predictions (zero loss)."""
    predictions = np.array([1.0, 2.0, 3.0])
    targets = predictions.copy()

    loss_value = mean_squared_error_loss.forward(
        predictions=predictions, targets=targets
    )

    assert np.isclose(loss_value, 0.0, atol=1e-10)


@pytest.mark.public
def test_mean_squared_error_backward_basic(
    mean_squared_error_loss: MeanSquaredError,
    simple_predictions: np.ndarray,
    simple_targets: np.ndarray,
):
    """Test basic MSE backward pass."""
    gradients = mean_squared_error_loss.backward(
        predictions=simple_predictions, targets=simple_targets
    )

    assert isinstance(gradients, np.ndarray)
    assert gradients.shape == simple_predictions.shape


@pytest.mark.public
def test_mean_squared_error_zero_arrays(mean_squared_error_loss: MeanSquaredError):
    """Test MSE with arrays of zeros."""
    predictions = np.zeros(3)
    targets = np.zeros(3)

    loss_value = mean_squared_error_loss.forward(
        predictions=predictions, targets=targets
    )
    gradients = mean_squared_error_loss.backward(
        predictions=predictions, targets=targets
    )

    assert np.isclose(loss_value, 0.0, atol=1e-10)
    np.testing.assert_allclose(gradients, np.zeros(3), atol=1e-10)


@pytest.fixture
def mean_squared_error_loss() -> MeanSquaredError:
    """Create an MSE loss instance for testing."""
    return MeanSquaredError()


@pytest.fixture
def simple_predictions() -> np.ndarray:
    """Simple predictions for testing."""
    return np.array([[1.0, 2.0, 3.0]])


@pytest.fixture
def simple_targets() -> np.ndarray:
    """Simple targets for testing."""
    return np.array([[1.5, 2.5, 2.5]])
