from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

from ..tree.types import Tree


class Module[T](Protocol):
    """
    Protocol for a neural network module.

    Defines the interface for a module used in a neural network,
    including methods for forward and backward passes, as well as factory methods
    for creating new instances of the module.
    """

    @property
    def parameters(self) -> Tree[T]:
        """
        Returns the parameters of the module. Acts as a serializer for the module's parameters.

        Returns:
            Tree: A tree structure containing the parameters of the module.
        """
        ...

    def forward(self, **inputs: T) -> T:
        """
        Forward pass of the module.

        Args:
            inputs (dict[str, T]): Input data to the module.

        Returns:
            T: Output data from the module after processing the input.
        """
        ...

    def backward(
        self, gradient_outputs: T, **inputs: T
    ) -> tuple[Tree[T], dict[str, T]]:
        """
        Backward pass of the module implementing chain rule.

        Computes:
        1. dL/dθ = (dL/dy) * (dy/dθ) - gradients w.r.t. parameters
        2. dL/dx = (dL/dy) * (dy/dx) - gradients w.r.t. inputs

        where L is loss, y is output, θ is parameters, x is inputs.

        Args:
            gradient_outputs (T): Gradient of the loss w.r.t. output (dL/dy).
            inputs (dict[str, T]): Input data that was used in the forward pass.

        Returns:
            tuple[Tree[T], dict[str, T]]: A tuple containing:
                - A tree structure with gradients of the module's parameters (dL/dθ). Should be same with the structure of `self.parameters`.
                - Gradient of the inputs with respect to the loss (dL/dx). Should be a dictionary with the same keys as `inputs`.
        """
        ...

    def with_parameters(self, parameters: Tree[T]) -> Self:
        """
        Creates a new instance of the module with the specified parameters. Acts as a deserializer for the module's parameters.

        Args:
            parameters (Tree[T]): Parameters to set for the new module instance.

        Returns:
            Module[T]: A new instance of the module with the specified parameters.
        """
        ...


class SupportsHessian[T](Protocol):
    """
    Protocol for modules that support Hessian computation.

    Defines the interface for modules that can compute Hessian matrices,
    including methods for the Hessian backward pass.
    """

    def hessian(
        self, hessian_outputs: T, gradient_outputs: T, **inputs: T
    ) -> tuple[Tree[T], dict[str, T]]:
        """
        Hessian backward pass implementing second-order chain rule.

        Computes:
        1. d²L/dθ² = (d²L/dy²) * (dy/dθ)² + (dL/dy) * (d²y/dθ²) - Hessian w.r.t. parameters
        2. d²L/dx² = (d²L/dy²) * (dy/dx)² + (dL/dy) * (d²y/dx²) - Hessian w.r.t. inputs

        where L is loss, y is output, θ is parameters, x is inputs.

        Args:
            hessian_outputs (T): Hessian of the loss w.r.t. output (d²L/dy²).
            gradient_outputs (T): Gradient of the loss w.r.t. output (dL/dy).
            inputs (dict[str, T]): Input data that was used in the forward pass.

        Returns:
            tuple[Tree[T], dict[str, T]]: A tuple containing:
                - A tree structure with Hessian of the module's parameters (d²L/dθ²). Should be same with the structure of `self.parameters`.
                - Hessian of the inputs with respect to the loss (d²L/dx²). Should be a dictionary with the same keys as `inputs`.
        """
        ...


@runtime_checkable
class SecondOrderModule[T](Module[T], SupportsHessian[T], Protocol):
    """
    Protocol for a neural network module that supports second-order derivatives.

    Combines the basic Module protocol with the SupportsHessian protocol
    to define a module that can perform both first and second-order computations.
    """

    ...
