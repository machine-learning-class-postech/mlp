from typing import Callable, Iterator

from ..dataset.protocols import Dataset
from ..sampler.protocols import Sampler


class DataLoader[T, Batch]:
    def __init__(
        self,
        dataset: Dataset[T],
        collate: Callable[[list[T]], Batch],
        sampler: Sampler[int] | Sampler[list[int]],
    ):
        self._dataset = dataset
        self._collate = collate
        self.sampler = sampler

    def __iter__(self) -> Iterator[Batch]:
        for index_or_indices in self.sampler:
            match index_or_indices:
                case int():
                    yield self._collate([self._dataset[index_or_indices]])
                case _:
                    yield self._collate([self._dataset[i] for i in index_or_indices])
