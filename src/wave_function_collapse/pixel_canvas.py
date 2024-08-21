import tkinter as tk

import numpy as np
import numpy.typing as npt

from wave_function_collapse.utils import rgb_array_to_string

__all__ = ["PixelCanvas"]


class PixelCanvas:
    def __init__(
        self,
        root: tk.Tk,
        width: int,
        height: int,
        pixel_size: int,
        background_color: str = "black",
        offset: int = 0,
    ):
        self._width = width
        self._height = height
        self._pixel_size = pixel_size
        self._offset = offset

        self._canvas = tk.Canvas(root, bg=background_color, height=height * pixel_size, width=width * pixel_size)
        self._pixel_values: npt.NDArray[np.uint8] = np.zeros([height, width, 3], dtype=np.uint8)

        self._pixels: npt.NDArray[np.int64] = np.zeros([height, width], dtype=np.int64)
        for y, x in np.ndindex((self._height, self._width)):
            x0 = (x * self._pixel_size) + self._offset
            y0 = (y * self._pixel_size) + self._offset
            x1 = x0 + self._pixel_size - 1 + self._offset
            y1 = y0 + self._pixel_size - 1 + self._offset
            self._pixels[y, x] = self._canvas.create_rectangle(x0, y0, x1, y1, fill=background_color, outline="")
        self._canvas.pack()

    @property
    def canvas(self) -> tk.Canvas:
        return self._canvas

    @property
    def pixel_values(self) -> npt.NDArray[np.uint8]:
        """
        Keep in mind that the numpy array holding the values is row-major, meaning that access to pixels is
        in a `[y, x]` order.
        """
        return self._pixel_values

    def set_pixel_values(self, pixel_values: npt.NDArray[np.uint8]) -> None:
        expected_shape = (self._height, self._width, 3)
        if pixel_values.shape != expected_shape:
            raise ValueError(f"`new_pixel_values` should have shape of {expected_shape}")

        self._pixel_values[:] = pixel_values

    def update_pixels(self) -> None:
        for y, x in np.ndindex((self._height, self._width)):
            self._canvas.itemconfig(self._pixels[y, x].item(), fill=rgb_array_to_string(self._pixel_values[y, x]))
