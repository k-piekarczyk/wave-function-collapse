import tkinter as tk

import numpy as np
import numpy.typing as npt
from numpy.polynomial.legendre import legmul

from wave_function_collapse.utils import string_to_rgb_array
from wave_function_collapse.types import RawLabelDescription, LabelDescription, PossibleLabel
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

#################
# Create labels #
#################
unique_labels = np.unique(template_pixel_values_str)
raw_labels_dict: dict[str, RawLabelDescription] = {label.item(): RawLabelDescription(up=[], down=[], left=[], right=[]) for label in unique_labels}

for y, x in np.ndindex((template_height, template_width)):
    if y - 1 >= 0:
        raw_labels_dict[template_pixel_values_str[y, x].item()]["up"].append(template_pixel_values_str[y - 1, x].item())

    if y + 1 < template_height:
        raw_labels_dict[template_pixel_values_str[y, x].item()]["down"].append(template_pixel_values_str[y + 1, x].item())

    if x - 1 >= 0:
        raw_labels_dict[template_pixel_values_str[y, x].item()]["left"].append(template_pixel_values_str[y, x - 1].item())

    if x + 1 < template_width:
        raw_labels_dict[template_pixel_values_str[y, x].item()]["right"].append(template_pixel_values_str[y, x + 1].item())

label_map: dict[str, LabelDescription] = {}
for label, raw_label_description in raw_labels_dict.items():
    label_description = LabelDescription(up=[], down=[], left=[], right=[])

    up_labels, up_counts = np.unique(raw_label_description["up"], return_counts=True)
    up_label_count_sum = np.sum(up_counts)
    for up_label, up_count in zip(list(up_labels), list(up_counts)):
        weight = up_count / up_label_count_sum
        label_description["up"].append(PossibleLabel(label=up_label, weight=weight))

    down_labels, down_counts = np.unique(raw_label_description["down"], return_counts=True)
    down_label_count_sum = np.sum(down_counts)
    for down_label, down_count in zip(list(down_labels), list(down_counts)):
        weight = down_count / down_label_count_sum
        label_description["down"].append(PossibleLabel(label=down_label, weight=weight))

    left_labels, left_counts = np.unique(raw_label_description["left"], return_counts=True)
    left_label_count_sum = np.sum(left_counts)
    for left_label, left_count in zip(list(left_labels), list(left_counts)):
        weight = left_count / left_label_count_sum
        label_description["left"].append(PossibleLabel(label=left_label, weight=weight))

    right_labels, right_counts = np.unique(raw_label_description["right"], return_counts=True)
    right_label_count_sum = np.sum(right_counts)
    for right_label, right_count in zip(list(right_labels), list(right_counts)):
        weight = right_count / right_label_count_sum
        label_description["right"].append(PossibleLabel(label=right_label, weight=weight))

    label_map[label] = label_description


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
