from typing import Literal, NamedTuple

__all__ = ["PropagationInstruction"]


class PropagationInstruction(NamedTuple):
    target: tuple[int, int]
    direction: Literal["up", "down", "left", "right"]
    origin: tuple[int, int]
