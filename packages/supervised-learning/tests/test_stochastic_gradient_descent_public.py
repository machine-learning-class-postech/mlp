from __future__ import annotations

from typing import TypedDict, cast

import numpy as np
import pytest
from mlp.tree.functions import are_equal
from mlp.tree.types import Tree
from supervised_learning import StochasticGradientDescent


class _Parameters(TypedDict):
    weights: np.ndarray
    bias: np.ndarray


@pytest.mark.public
def test_stochastic_gradient_descent_single_step(
    stochastic_gradient_descent_optimizer: StochasticGradientDescent,
    simple_parameters: _Parameters,
    simple_gradients: _Parameters,
):
    """Test a single SGD optimization step."""
    updated_parameters, _ = stochastic_gradient_descent_optimizer.step(
        parameters=cast(Tree[np.ndarray], simple_parameters),
        gradients=cast(Tree[np.ndarray], simple_gradients),
    )

    updated_parameters = cast(_Parameters, updated_parameters)

    assert updated_parameters["weights"].shape == simple_parameters["weights"].shape
    assert updated_parameters["bias"].shape == simple_parameters["bias"].shape


@pytest.mark.public
def test_stochastic_gradient_descent_step_modifies_parameters(
    stochastic_gradient_descent_optimizer: StochasticGradientDescent,
    simple_parameters: _Parameters,
    simple_gradients: _Parameters,
):
    """Test that SGD step actually modifies parameters."""
    updated_parameters, _ = stochastic_gradient_descent_optimizer.step(
        parameters=cast(Tree[np.ndarray], simple_parameters),
        gradients=cast(Tree[np.ndarray], simple_gradients),
    )

    updated_parameters = cast(_Parameters, updated_parameters)

    assert not are_equal(
        updated_parameters["weights"], simple_parameters["weights"], np.array_equal
    )
    assert not are_equal(
        updated_parameters["bias"], simple_parameters["bias"], np.array_equal
    )


@pytest.mark.public
def test_stochastic_gradient_descent_step_returns_finite_values(
    stochastic_gradient_descent_optimizer: StochasticGradientDescent,
    simple_parameters: _Parameters,
    simple_gradients: _Parameters,
):
    """Test that SGD step returns finite values."""
    updated_parameters, _ = stochastic_gradient_descent_optimizer.step(
        parameters=cast(Tree[np.ndarray], simple_parameters),
        gradients=cast(Tree[np.ndarray], simple_gradients),
    )

    updated_parameters = cast(_Parameters, updated_parameters)

    assert np.all(np.isfinite(updated_parameters["weights"]))
    assert np.all(np.isfinite(updated_parameters["bias"]))


@pytest.mark.public
def test_stochastic_gradient_descent_with_different_parameter_structures():
    """Test SGD optimizer with different parameter tree structures."""
    optimizer = StochasticGradientDescent()

    parameters_1 = {"w": np.array([1.0, 2.0]), "b": np.array([0.1])}
    gradients_1 = {"w": np.array([0.1, 0.2]), "b": np.array([0.01])}

    updated_1, _ = optimizer.step(
        parameters=cast(Tree[np.ndarray], parameters_1),
        gradients=cast(Tree[np.ndarray], gradients_1),
    )

    assert "w" in updated_1
    assert "b" in updated_1
    assert updated_1["w"].shape == (2,)  # type: ignore
    assert updated_1["b"].shape == (1,)  # type: ignore

    parameters_2 = {"layer1": {"weight": np.array([[1.0, 2.0]])}}
    gradients_2 = {"layer1": {"weight": np.array([[0.1, 0.2]])}}

    updated_2, _ = optimizer.step(
        parameters=cast(Tree[np.ndarray], parameters_2),
        gradients=cast(Tree[np.ndarray], gradients_2),
    )

    assert "layer1" in updated_2
    assert "weight" in updated_2["layer1"]  # type: ignore


@pytest.fixture
def simple_parameters() -> Tree[np.ndarray]:
    """Simple parameter tree for testing."""
    return {"weights": np.array([[1.0, 2.0], [3.0, 4.0]]), "bias": np.array([0.1, 0.2])}


@pytest.fixture
def simple_gradients() -> Tree[np.ndarray]:
    """Simple gradient tree matching parameters."""
    return {
        "weights": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "bias": np.array([0.01, 0.02]),
    }


@pytest.fixture
def stochastic_gradient_descent_optimizer() -> StochasticGradientDescent:
    """Create an SGD optimizer instance for testing."""
    return StochasticGradientDescent()
