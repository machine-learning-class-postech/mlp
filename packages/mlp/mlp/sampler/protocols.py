from typing import Iterator, Protocol


class Sampler[T](Protocol):
    def __iter__(self) -> Iterator[T]: ...

    def __len__(self) -> int: ...
