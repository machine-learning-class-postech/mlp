"""
Assignment: Supervised Learning
"""

from __future__ import annotations

from typing import TypedDict, Unpack

import numpy as np
from mlp.loss.protocols import SecondOrderLoss
from mlp.module.protocols import SecondOrderModule
from mlp.optimizer.protocols import Optimizer, SecondOrderOptimizer
from mlp.tree.types import Tree

# from mlp.tree.functions import tree_map, zeros_like


class _Singleton(TypedDict):
    x: np.ndarray


class Linear(SecondOrderModule[np.ndarray]):
    """
    Linear transformation layer implementing y = x @ W^T + b.

    This is a fully connected layer that performs an affine transformation
    on the input. Supports both forward, backward, and second-order backward passes.

    Args:
        weights: Weight matrix of shape (output_size, input_size)
        bias: Bias vector of shape (output_size,)
    """

    def __init__(self, weights: np.ndarray, bias: np.ndarray):
        """Initialize linear layer with given weights and bias."""
        ...

    @property
    def parameters(self) -> Tree[np.ndarray]:
        """Return the learnable parameters as a tree structure."""
        ...

    def forward(self, **inputs: Unpack[_Singleton]) -> np.ndarray:
        """
        Forward pass: compute y = x @ W^T + b.

        Args:
            x: Input vector of shape (input_size,)

        Returns:
            Output vector of shape (output_size,)
        """
        ...

    def backward(
        self, gradient_outputs: np.ndarray, **inputs: Unpack[_Singleton]
    ) -> tuple[Tree[np.ndarray], dict[str, np.ndarray]]:
        """
        Backward pass: compute gradients with respect to parameters and inputs.

        Args:
            gradient_outputs: Gradient of loss with respect to output
            x: Input from forward pass

        Returns:
            Tuple of (parameter_gradients, input_gradients)
        """
        ...

    def hessian(
        self,
        hessian_outputs: np.ndarray,
        gradient_outputs: np.ndarray,
        **inputs: np.ndarray,
    ) -> tuple[Tree[np.ndarray], dict[str, np.ndarray]]:
        """
        Second-order backward pass: compute Hessian information.

        Args:
            hessian_outputs: Hessian of loss with respect to output
            gradient_outputs: Gradient of loss with respect to output
            x: Input from forward pass

        Returns:
            Tuple of (parameter_hessians, input_hessians)
        """
        ...

    def with_parameters(
        self,
        parameters: Tree[np.ndarray],
    ) -> Linear:
        """
        Create a new Linear layer with different parameters.

        Args:
            parameters: New parameters to set for the layer
        """
        ...

    @staticmethod
    def create(
        input_size: int, output_size: int, random_generator: np.random.Generator
    ) -> Linear:
        """
        Factory method to create a Linear layer with random initialization.

        Args:
            input_size: Number of input features
            output_size: Number of output features
            random_generator: NumPy random generator for weight initialization

        Returns:
            New Linear layer with Gaussian-initialized weights and zero bias
        """
        weights = random_generator.normal(0, 0.1, (output_size, input_size))
        bias = np.zeros(output_size)
        return Linear(weights, bias)


class Sequential(SecondOrderModule[np.ndarray]):
    """
    Sequential container for stacking multiple modules.

    This class allows chaining multiple modules together, where the output
    of one module becomes the input to the next. Supports forward, backward,
    and second-order backward passes. Assume that all modules have a single input 'x'.

    Args:
        modules: List of modules to be applied in sequence
    """

    def __init__(self, modules: list[SecondOrderModule[np.ndarray]]):
        """Initialize Sequential with a list of modules."""
        ...

    @property
    def parameters(self) -> Tree[np.ndarray]:
        """Return combined parameters from all modules."""
        ...

    def forward(self, **inputs: Unpack[_Singleton]) -> np.ndarray:
        """
        Forward pass through all modules in sequence.

        Args:
            x: Input array

        Returns:
            Output array after passing through all modules
        """
        ...

    def backward(
        self, gradient_outputs: np.ndarray, **inputs: Unpack[_Singleton]
    ) -> tuple[Tree[np.ndarray], dict[str, np.ndarray]]:
        """
        Backward pass through all modules in reverse order.
        Args:
            gradient_outputs: Gradient of loss with respect to final output
            x: Input from forward pass
        Returns:
            Tuple of (parameter_gradients, input_gradients)
        """
        ...

    def hessian(
        self,
        hessian_outputs: np.ndarray,
        gradient_outputs: np.ndarray,
        **inputs: Unpack[_Singleton],
    ) -> tuple[Tree[np.ndarray], dict[str, np.ndarray]]:
        """
        Second-order backward pass through all modules in reverse order.

        Args:
            hessian_outputs: Hessian of loss with respect to final output
            gradient_outputs: Gradient of loss with respect to final output
            x: Input from forward pass

        Returns:
            Tuple of (parameter_hessians, input_hessians)
        """
        ...

    def with_parameters(
        self,
        parameters: Tree[np.ndarray],
    ) -> Sequential:
        """Create new Sequential with different parameters for each module."""
        ...


class ReLU(SecondOrderModule[np.ndarray]):
    """
    Rectified Linear Unit (ReLU) activation function.

    Implements f(x) = max(0, x), which sets negative values to zero
    and preserves positive values. This is the most common activation
    function in deep learning due to its simplicity and effectiveness.
    """

    @property
    def parameters(self) -> Tree[np.ndarray]:
        """ReLU has no learnable parameters."""
        return {}

    def forward(self, **inputs: Unpack[_Singleton]) -> np.ndarray:
        """
        Forward pass: apply ReLU activation f(x) = max(0, x).

        Args:
            x: Input array

        Returns:
            Output array with negative values clipped to zero
        """
        ...

    def backward(
        self, gradient_outputs: np.ndarray, **inputs: Unpack[_Singleton]
    ) -> tuple[Tree[np.ndarray], dict[str, np.ndarray]]:
        """
        Backward pass: compute ReLU derivative.
        At x = 0, we conventionally use 0 (though mathematically undefined).

        Args:
            gradient_outputs: Gradient of loss with respect to output
            x: Input from forward pass

        Returns:
            Tuple of (empty parameter_gradients, input_gradients)
        """
        ...

    def hessian(
        self,
        hessian_outputs: np.ndarray,
        gradient_outputs: np.ndarray,
        **inputs: Unpack[_Singleton],
    ) -> tuple[Tree[np.ndarray], dict[str, np.ndarray]]:
        """
        Second-order backward pass.

        Returns:
            Tuple of (empty parameter_hessians, zero input_hessians)
        """
        ...

    def with_parameters(
        self,
        parameters: Tree[np.ndarray],
    ) -> ReLU:
        """Create new ReLU (parameters ignored since ReLU is parameter-free)."""
        return ReLU()

    @staticmethod
    def create() -> ReLU:
        """Factory method to create a ReLU activation function."""
        return ReLU()


class Newton(SecondOrderOptimizer[np.ndarray]):
    """
    Newton's optimization method using second-order information.

    Newton's method uses both first-order (gradient) and second-order (Hessian)
    information to find optimal parameter updates. The update rule is:
    θ_{t+1} = θ_t - H^{-1} * ∇f(θ_t)

    For stability, we approximate H^{-1} as 1/(H + ε) where ε is a small constant.

    Args:
        epsilon: Small constant for numerical stability (default: 1e-1)
    """

    def __init__(self, epsilon: float = 1e-1):
        """Initialize Newton optimizer with given epsilon for stability."""
        ...

    def step(
        self,
        parameters: Tree[np.ndarray],
        gradients: Tree[np.ndarray],
        second_order_gradients: Tree[np.ndarray],
    ) -> tuple[Tree[np.ndarray], SecondOrderOptimizer[np.ndarray]]:
        """
        Perform one Newton optimization step.

        Args:
            parameters: Current parameter values
            gradients: First-order gradients
            second_order_gradients: Second-order gradients (diagonal Hessian approximation)

        Returns:
            Tuple of (updated_parameters, new_optimizer_instance)
        """
        ...


class StochasticGradientDescent(Optimizer[np.ndarray]):
    """
    Stochastic Gradient Descent (SGD) optimizer.

    SGD is the fundamental optimization algorithm for neural networks.
    It updates parameters in the opposite direction of the gradient:
    θ_{t+1} = θ_t - α * ∇f(θ_t)

    Args:
        learning_rate: Step size for parameter updates (default: 0.01)
    """

    def __init__(self, learning_rate: float = 0.01):
        """Initialize SGD optimizer with given learning rate."""
        ...

    def step(
        self,
        parameters: Tree[np.ndarray],
        gradients: Tree[np.ndarray],
    ) -> tuple[Tree[np.ndarray], Optimizer[np.ndarray]]:
        """
        Perform one SGD optimization step.

        Args:
            parameters: Current parameter values
            gradients: Gradients with respect to parameters

        Returns:
            Tuple of (updated_parameters, new_optimizer_instance)
        """
        ...


class MeanSquaredError(SecondOrderLoss[np.ndarray]):
    """
    Mean Squared Error (MSE) loss function for regression tasks.

    MSE computes the average of squared differences between predictions and targets:
    L(y, ŷ) = (1/n) * Σ(y_i - ŷ_i)²

    This loss function is differentiable everywhere and commonly used for
    regression problems where we want to penalize large errors more heavily.
    """

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute mean squared error loss.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            MSE loss as a scalar value
        """
        ...

    def backward(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Compute gradient of MSE loss with respect to predictions.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Gradient of the loss with respect to predictions
        """
        ...

    def hessian(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Compute second-order derivative (Hessian) of MSE loss with respect to predictions.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Hessian of the loss with respect to predictions
        """
        ...


class CrossEntropy(SecondOrderLoss[np.ndarray]):
    """
    Cross-entropy loss function for multi-class classification.

    Cross-entropy loss combines softmax activation with negative log-likelihood:
    L(y, ŷ) = -Σ y_i * log(softmax(ŷ_i))

    This is the standard loss function for classification tasks where classes
    are mutually exclusive. The softmax ensures outputs sum to 1 (probabilities).
    """

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute cross-entropy loss with softmax.

        Args:
            predictions: Raw model outputs (logits)
            targets: One-hot encoded target labels

        Returns:
            Cross-entropy loss as a scalar value
        """
        ...

    def backward(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Compute gradient of cross-entropy loss with respect to predictions.

        Args:
            predictions: Raw model outputs (logits)
            targets: One-hot encoded target labels

        Returns:
            Gradient of the loss with respect to predictions
        """
        ...

    def hessian(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Compute second-order derivative (Hessian) of cross-entropy loss with respect to predictions.

        Args:
            predictions: Raw model outputs (logits)
            targets: One-hot encoded target labels

        Returns:
            Hessian of the loss with respect to predictions
        """
        ...
