from __future__ import annotations

import numpy as np
import pytest
from supervised_learning import CrossEntropy


@pytest.mark.public
def test_cross_entropy_forward_basic(
    cross_entropy_loss: CrossEntropy,
    binary_predictions: np.ndarray,
    binary_targets: np.ndarray,
):
    """Test basic Cross Entropy forward pass."""
    loss_value = cross_entropy_loss.forward(
        predictions=binary_predictions, targets=binary_targets
    )

    assert isinstance(loss_value, (float, np.floating))
    assert loss_value >= 0.0


@pytest.mark.public
def test_cross_entropy_forward_multiclass(
    cross_entropy_loss: CrossEntropy,
    multiclass_predictions: np.ndarray,
    multiclass_targets: np.ndarray,
):
    """Test Cross Entropy forward pass with multi-class data."""
    loss_value = cross_entropy_loss.forward(
        predictions=multiclass_predictions, targets=multiclass_targets
    )

    assert isinstance(loss_value, (float, np.floating))
    assert loss_value >= 0.0


@pytest.mark.public
def test_cross_entropy_backward_basic(
    cross_entropy_loss: CrossEntropy,
    binary_predictions: np.ndarray,
    binary_targets: np.ndarray,
):
    """Test basic Cross Entropy backward pass."""
    gradients = cross_entropy_loss.backward(
        predictions=binary_predictions, targets=binary_targets
    )

    assert isinstance(gradients, np.ndarray)
    assert gradients.shape == binary_predictions.shape


@pytest.mark.public
def test_cross_entropy_backward_multiclass(
    cross_entropy_loss: CrossEntropy,
    multiclass_predictions: np.ndarray,
    multiclass_targets: np.ndarray,
):
    """Test Cross Entropy backward pass with multi-class data."""
    gradients = cross_entropy_loss.backward(
        predictions=multiclass_predictions, targets=multiclass_targets
    )

    assert isinstance(gradients, np.ndarray)
    assert gradients.shape == multiclass_predictions.shape


@pytest.mark.public
def test_cross_entropy_zero_predictions(cross_entropy_loss: CrossEntropy):
    """Test Cross Entropy with zero predictions (neutral logits)."""
    predictions = np.array([[0.0, 0.0]])
    targets = np.array([[1.0, 0.0]])

    loss_value = cross_entropy_loss.forward(predictions=predictions, targets=targets)
    gradients = cross_entropy_loss.backward(predictions=predictions, targets=targets)

    assert np.isfinite(loss_value)
    assert np.all(np.isfinite(gradients))

    expected_loss = -np.log(0.5)
    assert np.isclose(loss_value, expected_loss, rtol=1e-5)


@pytest.fixture
def cross_entropy_loss() -> CrossEntropy:
    """Create a Cross Entropy loss instance for testing."""
    from supervised_learning import CrossEntropy

    return CrossEntropy()


@pytest.fixture
def binary_predictions() -> np.ndarray:
    """Binary classification predictions (logits)."""
    return np.array([[2.0, -1.0]])


@pytest.fixture
def binary_targets() -> np.ndarray:
    """Binary classification targets (one-hot or class indices)."""
    return np.array([[1.0, 0.0]])


@pytest.fixture
def multiclass_predictions() -> np.ndarray:
    """Multi-class classification predictions (logits)."""
    return np.array([[1.0, 2.0, 0.5]])


@pytest.fixture
def multiclass_targets() -> np.ndarray:
    """Multi-class classification targets."""
    return np.array([[0.0, 1.0, 0.0]])
