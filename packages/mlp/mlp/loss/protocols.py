from typing import Protocol, runtime_checkable


class Loss[T](Protocol):
    def forward(self, predictions: T, targets: T) -> float:
        """
        Computes the loss between predictions and targets.

        Args:
            predictions (T): The predicted values.
            targets (T): The ground truth values.

        Returns:
            float: The computed loss value.
        """
        ...

    def backward(self, predictions: T, targets: T) -> T:
        """
        Computes the gradient of the loss with respect to the predictions.
        This implements the first step of the chain rule: dL/dy where y is predictions.

        Args:
            predictions (T): The predicted values.
            targets (T): The ground truth values.

        Returns:
            T: The gradient of the loss with respect to the predictions.
        """
        ...


class SupportsHessian[T](Protocol):
    def hessian(self, predictions: T, targets: T) -> T:
        """
        Computes the second-order derivative (Hessian) of the loss with respect to the predictions.
        This implements the second-order chain rule component: d²L/dy² where y is predictions.

        Args:
            predictions (T): The predicted values.
            targets (T): The ground truth values.

        Returns:
            T: The Hessian of the loss with respect to the predictions.
        """
        ...


@runtime_checkable
class SecondOrderLoss[T](Loss[T], SupportsHessian[T], Protocol):
    """
    Protocol for a loss function that supports second-order derivatives.

    Combines the basic Loss protocol with the SupportsHessian protocol
    to define a loss that can perform both first and second-order computations.
    """
