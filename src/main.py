import tkinter as tk

import numpy as np

from wave_function_collapse.utils import string_to_rgb_array
from wave_function_collapse.pixel_canvas import PixelCanvas

pixel_size = 16
height, width = 32, 32
template_height, template_width = 6, 6

root = tk.Tk("Pixelart")

###################
# Template canvas #
###################
template_canvas = PixelCanvas(root=root, height=template_height, width=template_width, pixel_size=pixel_size, offset=2)

template_pixel_values_str = np.asarray([
    ["#000000", "#000000", "#000000", "#000000", "#000000", "#000000"],
    ["#000000", "#000000", "#00ff00", "#ff0000", "#000000", "#000000"],
    ["#000000", "#ff0000", "#34ebbd", "#000000", "#000000", "#000000"],
    ["#000000", "#000000", "#34ebbd", "#000000", "#000000", "#000000"],
    ["#000000", "#000000", "#ff0000", "#000000", "#000000", "#000000"],
    ["#000000", "#000000", "#000000", "#000000", "#000000", "#000000"],
], dtype=str)
template_pixel_values = np.zeros([template_height, template_width, 3], dtype=np.int64)

for y, x in np.ndindex((template_height, template_width)):
    template_pixel_values[y, x] = string_to_rgb_array(template_pixel_values_str[y, x].item())

template_canvas.set_pixel_values(pixel_values=template_pixel_values)
template_canvas.update_pixels()

########################
# Wave collapse canvas #
########################
pixel_canvas = PixelCanvas(root=root, height=height, width=width, pixel_size=pixel_size, offset=2)

root.mainloop()

# window_destroyed = False
#
#
# def on_destroyed(event):
#     if event.widget != root:
#         return
#
#     global window_destroyed
#     window_destroyed = True
#
#
# root.bind("<Destroy>", on_destroyed)
#
# while True:
#     if window_destroyed:
#         break
#
#     for y, x in np.ndindex((height, width)):
#         pixel_canvas.pixel_values[y, x] = np.random.randint(low=0, high=256, size=3)
#     pixel_canvas.update_pixels()
#
#     root.update()
