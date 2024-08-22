from typing import TypedDict

__all__ = ["LabelDescription"]


class LabelDescription(TypedDict):
    up: set[str]
    down: set[str]
    left: set[str]
    right: set[str]
