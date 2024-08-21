import numpy as np
import numpy.typing as npt

__all__ = ["string_to_rgb_array"]


def string_to_rgb_array(rgb_hex: str) -> npt.NDArray[np.uint8]:
    return np.asarray([int(rgb_hex[1:3], 16), int(rgb_hex[3:5], 16), int(rgb_hex[5:7], 16)], dtype=np.uint8)
