import numpy as np
import numpy.typing as npt

from numba import jit
from typing import Optional

from wave_function_collapse.utils.string_to_rgb_array import string_to_rgb_array

__all__ = ["transform_labels_to_avg_rgb_value"]


def transform_labels_to_avg_rgb_value(input_array: npt.NDArray[set[str]], output_array: Optional[npt.NDArray[np.uint8]] = None) -> npt.NDArray[np.uint8]:
    if output_array is None:
        np.zeros([input_array.shape[0], input_array.shape[1], 3], dtype=np.uint8)
    elif input_array.shape[0] != output_array.shape[0] and input_array.shape[1] != output_array.shape[1]:
        raise ValueError("`input_array` and `output_array` need to have the same dimensions")

    for y, x in np.ndindex((input_array.shape[0], input_array.shape[1])):
        labels: set[str] = set(input_array[y, x])

        rgb_values = np.asarray([string_to_rgb_array(label) for label in labels])
        rgb_avg_label_value = np.average(rgb_values, axis=0).astype(np.uint8)

        output_array[y, x] = rgb_avg_label_value

    return output_array
