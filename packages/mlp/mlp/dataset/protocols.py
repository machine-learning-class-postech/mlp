from typing import Protocol, TypeVar

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> T_co: ...
