from typing import Any

from mlp.data_loader.actors import DataLoader
from mlp.loss.protocols import Loss, SecondOrderLoss
from mlp.module.protocols import Module, SecondOrderModule
from mlp.optimizer.protocols import Optimizer, SecondOrderOptimizer

from .protocols import Trainer


class SupervisedLearningTrainer[T](Trainer[T, tuple[T, T]]):
    def __init__(
        self,
        initial_model: Module[T],
        initial_optimizer: Optimizer[T],
        loss: Loss[T],
    ):
        self._initial_model = initial_model
        self._initial_optimizer = initial_optimizer
        self._loss = loss

    def train(
        self,
        training_data: DataLoader[Any, tuple[T, T]],
        epochs: int,
    ) -> tuple[Module[T], list[float]]:
        model = self._initial_model
        optimizer = self._initial_optimizer

        losses: list[float] = []

        for _epoch in range(epochs):
            for inputs, targets in training_data:
                predictions = model.forward(x=inputs)
                gradients = self._loss.backward(
                    predictions=predictions, targets=targets
                )
                parameters_gradients, _ = model.backward(
                    gradient_outputs=gradients, x=inputs
                )
                next_parameters, optimizer = optimizer.step(
                    model.parameters, parameters_gradients
                )
                model = model.with_parameters(next_parameters)

                losses.append(
                    self._loss.forward(
                        predictions=model.forward(x=inputs), targets=targets
                    )
                )

        return model, losses


class SecondOrderSupervisedLearningTrainer[T](Trainer[T, tuple[T, T]]):
    def __init__(
        self,
        initial_model: SecondOrderModule[T],
        initial_optimizer: SecondOrderOptimizer[T],
        loss: SecondOrderLoss[T],
    ):
        self._initial_model = initial_model
        self._initial_optimizer = initial_optimizer
        self._loss = loss

    def train(
        self,
        training_data: DataLoader[Any, tuple[T, T]],
        epochs: int,
    ) -> tuple[SecondOrderModule[T], list[float]]:
        model = self._initial_model
        optimizer = self._initial_optimizer

        losses: list[float] = []

        for _epoch in range(epochs):
            for inputs, targets in training_data:
                predictions = model.forward(x=inputs)
                gradients = self._loss.backward(
                    predictions=predictions, targets=targets
                )
                hessians = self._loss.hessian(predictions=predictions, targets=targets)
                parameters_gradients, _ = model.backward(
                    gradient_outputs=gradients,
                    x=inputs,
                )
                parameters_hessians, _ = model.hessian(
                    gradient_outputs=gradients,
                    hessian_outputs=hessians,
                    x=inputs,
                )
                next_parameters, optimizer = optimizer.step(
                    model.parameters,
                    parameters_gradients,
                    parameters_hessians,
                )
                model = model.with_parameters(next_parameters)

                losses.append(
                    self._loss.forward(
                        predictions=model.forward(x=inputs), targets=targets
                    )
                )

        return model, losses
