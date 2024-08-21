import numpy as np
import numpy.typing as npt

__all__ = ["rgb_array_to_string"]


def rgb_array_to_string(rgb_array: npt.NDArray[np.uint8]) -> str:
    return "#{:02x}{:02x}{:02x}".format(rgb_array[0], rgb_array[1], rgb_array[2])
