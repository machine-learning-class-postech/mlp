import math
from typing import Iterator, Sized

import numpy as np

from .protocols import Sampler


class SequentialSampler(Sampler[int]):
    def __init__(self, data_source: Sized) -> None:
        self._data_source = data_source

    def __iter__(self) -> Iterator[int]:
        for index in range(len(self._data_source)):
            yield index

    def __len__(self) -> int:
        return len(self._data_source)


class RandomSampler(Sampler[int]):
    def __init__(
        self, data_source: Sized, random_generator: np.random.Generator
    ) -> None:
        self._data_source = data_source
        self._random_generator = random_generator

    def __iter__(self) -> Iterator[int]:
        for index in self._random_generator.permutation(len(self._data_source)):
            yield index

    def __len__(self) -> int:
        return len(self._data_source)


class BatchSampler(Sampler[list[int]]):
    def __init__(
        self, sampler: Sampler[int], batch_size: int, drop_last: bool = False
    ) -> None:
        self._sampler = sampler
        self._batch_size = batch_size
        self._drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        batch: list[int] = []
        for index in self._sampler:
            batch.append(index)
            if len(batch) == self._batch_size:
                yield batch
                batch = []

        if not self._drop_last and batch:
            yield batch

    def __len__(self) -> int:
        if self._drop_last:
            return len(self._sampler) // self._batch_size
        else:
            return math.ceil(len(self._sampler) / self._batch_size)
