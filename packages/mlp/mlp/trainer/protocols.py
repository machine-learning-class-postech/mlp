from typing import Any, Protocol

from mlp.data_loader.actors import DataLoader
from mlp.module.protocols import Module


class Trainer[T, Batch](Protocol):
    def train(
        self,
        training_data: DataLoader[Any, Batch],
        epochs: int,
    ) -> tuple[Module[T], list[float]]: ...
