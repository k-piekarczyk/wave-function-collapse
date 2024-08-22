from typing import TypeVar

import numpy as np

__all__ = ["set_intersection"]


T = TypeVar("T")


def set_intersection(set_1: set[T], set_2: set[T]) -> set[T]:
    return set(np.intersect1d(ar1=list(set_1), ar2=list(set_2)))
