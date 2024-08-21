from typing import TypedDict, NamedTuple


__all__ = ["LabelDescription", "PossibleLabel"]


class PossibleLabel(NamedTuple):
    label: str
    weight: float


class LabelDescription(TypedDict):
    up: list[PossibleLabel]
    down: list[PossibleLabel]
    left: list[PossibleLabel]
    right: list[PossibleLabel]
