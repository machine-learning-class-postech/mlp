from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..tree.types import Tree


class Optimizer[T](Protocol):
    """
    Protocol for an optimizer in a machine learning framework.

    Defines the required interface for optimizers, including methods for
    returning new parameters along with the optimizer's next state.
    """

    def step(
        self, parameters: Tree[T], gradients: Tree[T]
    ) -> tuple[Tree[T], Optimizer[T]]:
        """
        Performs a single optimization step.

        Args:
            parameters (Tree[T]): Current parameters of the model.
            gradients (Tree[T]): Gradients of the loss with respect to the parameters.

        Returns:
            tuple[Tree[T], Optimizer[T]]: A tuple containing the updated parameters and the next state of the optimizer.
        """
        ...


@runtime_checkable
class SecondOrderOptimizer[T](Protocol):
    """
    Protocol for an optimizer that supports second-order derivatives.

    Extends the basic Optimizer protocol to include a method for performing
    optimization steps using second-order gradients.
    """

    def step(
        self, parameters: Tree[T], gradients: Tree[T], second_order_gradients: Tree[T]
    ) -> tuple[Tree[T], SecondOrderOptimizer[T]]:
        """
        Performs a single optimization step using second-order gradients.

        Args:
            parameters (Tree[T]): Current parameters of the model.
            gradients (Tree[T]): Gradients of the loss with respect to the parameters.
            second_order_gradients (Tree[T]): Second-order gradients of the loss with respect to the parameters.

        Returns:
            tuple[Tree[T], SecondOrderOptimizer[T]]: A tuple containing the updated parameters and the next state of the optimizer.
        """
        ...
