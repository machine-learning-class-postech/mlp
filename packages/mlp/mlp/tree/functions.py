from typing import Callable, cast

import numpy as np

from .types import Tree


def zeros_like(
    tree: Tree[np.ndarray],
) -> Tree[np.ndarray]:
    if isinstance(tree, dict):
        return {key: np.zeros_like(value) for key, value in tree.items()}
    else:
        return [zeros_like(cast(Tree[np.ndarray], value)) for value in tree]


def tree_map[T, U](
    f: Callable[[T], U] | Callable[[T, T], U] | Callable[[T, T, T], U],
    *trees: Tree[T],
) -> Tree[U]:
    first_tree = next(iter(trees))
    assert all(type(first_tree) is type(tree) for tree in trees), (
        "All trees must have the same structure"
    )

    match first_tree:
        case dict():
            _trees = cast(list[dict[str, Tree[T]]], list(trees))
            return {
                key: tree_map(f, *(_tree[key] for _tree in _trees))
                for key in cast(dict[str, T], first_tree)
            }
        case list():
            _trees = cast(list[list[Tree[T]]], list(trees))
            return [
                tree_map(f, *(tree[i] for tree in _trees))
                for i in range(len(cast(list[T], first_tree)))
            ]
        case _:
            _trees = cast(list[T], list(trees))
            return f(*_trees)


def are_equal[T](
    a: Tree[T],
    b: Tree[T],
    using: Callable[[T, T], bool] | None = None,
) -> bool:
    assert type(a) is type(b), "Trees must have the same structure"

    match a, b:
        case dict(), dict():
            _a = cast(dict[str, Tree[T]], a)
            _b = cast(dict[str, Tree[T]], b)
            return set(_a.keys()) == set(_b.keys()) and all(
                are_equal(_a[key], _b[key], using) for key in _a.keys()
            )
        case list(), list():
            _a = cast(list[Tree[T]], a)
            _b = cast(list[Tree[T]], b)
            return len(_a) == len(_b) and all(
                are_equal(_a[i], _b[i], using) for i in range(len(_a))
            )
        case _:
            _a = cast(T, a)
            _b = cast(T, b)
            return using(_a, _b) if using else _a == _b
