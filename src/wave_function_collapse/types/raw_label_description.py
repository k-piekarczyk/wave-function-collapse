from typing import TypedDict


__all__ = ["RawLabelDescription"]


class RawLabelDescription(TypedDict):
    up: list[str]
    down: list[str]
    left: list[str]
    right: list[str]
