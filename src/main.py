import tkinter as tk

import numpy as np
import numpy.typing as npt

from wave_function_collapse.pixel_canvas import PixelCanvas
from wave_function_collapse.types import LabelDescription
from wave_function_collapse.utils import string_to_rgb_array, transform_labels_to_avg_rgb_value

pixel_size = 32
height, width = 12, 12
template_height, template_width = 6, 6

root = tk.Tk("Pixelart")

###################
# Template canvas #
###################
template_canvas = PixelCanvas(root=root, height=template_height, width=template_width, pixel_size=pixel_size, offset=2)

template_pixel_values_str = np.asarray(
    [
        ["#000000", "#000000", "#000000", "#000000", "#000000", "#000000"],
        ["#000000", "#000000", "#00ff00", "#ff0000", "#000000", "#000000"],
        ["#000000", "#ff0000", "#34ebbd", "#000000", "#000000", "#000000"],
        ["#000000", "#000000", "#34ebbd", "#000000", "#000000", "#000000"],
        ["#000000", "#000000", "#ff0000", "#000000", "#000000", "#000000"],
        ["#000000", "#000000", "#000000", "#000000", "#000000", "#000000"],
    ],
    dtype=str,
)
template_pixel_values = np.zeros([template_height, template_width, 3], dtype=np.uint8)

for y, x in np.ndindex((template_height, template_width)):
    template_pixel_values[y, x] = string_to_rgb_array(template_pixel_values_str[y, x].item())

template_canvas.set_pixel_values(pixel_values=template_pixel_values)
template_canvas.update_pixels()

#################
# Create labels #
#################
unique_labels, label_counts = np.unique(template_pixel_values_str, return_counts=True)

label_map: dict[str, LabelDescription] = {
    label.item(): LabelDescription(up=set(), down=set(), left=set(), right=set()) for label in unique_labels
}

label_count_sum = np.sum(label_counts)
label_weights_map: dict[str, float] = {}
for label, count in zip(list(unique_labels), list(label_counts)):
    label_weights_map[label] = count / label_count_sum

for y, x in np.ndindex((template_height, template_width)):
    if y - 1 >= 0:
        label_map[template_pixel_values_str[y, x].item()]["up"].add(template_pixel_values_str[y - 1, x].item())

    if y + 1 < template_height:
        label_map[template_pixel_values_str[y, x].item()]["down"].add(template_pixel_values_str[y + 1, x].item())

    if x - 1 >= 0:
        label_map[template_pixel_values_str[y, x].item()]["left"].add(template_pixel_values_str[y, x - 1].item())

    if x + 1 < template_width:
        label_map[template_pixel_values_str[y, x].item()]["right"].add(template_pixel_values_str[y, x + 1].item())

#####################
# Initialize labels #
#####################
possible_solution: npt.NDArray[set[str]] = np.array([set(unique_labels) for _ in range (height * width)]).reshape(height, width)

label_to_color_mapping = {label: string_to_rgb_array(label) for label in unique_labels}
average_label_pixel_values = np.zeros([height, width, 3], dtype=np.uint8)

transform_labels_to_avg_rgb_value(input_array=possible_solution, output_array=average_label_pixel_values)

########################
# Wave collapse canvas #
########################
pixel_canvas = PixelCanvas(root=root, height=height, width=width, pixel_size=pixel_size, offset=2)
pixel_canvas.set_pixel_values(pixel_values=average_label_pixel_values)
pixel_canvas.update_pixels()

#################################
# Wave Collapse algorithm setup #
#################################


################
# Window setup #
################
window_destroyed = False


def on_destroyed(event):
    if event.widget != root:
        return

    global window_destroyed
    window_destroyed = True


root.bind("<Destroy>", on_destroyed)

finished = False
solution_space_initialized = False
while not finished:
    if window_destroyed:
        break

    ###########################
    # Wave Collapse algorithm #
    ###########################
    if not solution_space_initialized:
        start_y, start_x = (np.random.randint(low=0, high=height), np.random.randint(low=0, high=width))

        start_labels = list(possible_solution[start_y, start_x])
        start_weights = [label_weights_map[label] for label in start_labels]

        possible_solution[start_y, start_x] = set(np.random.choice(start_labels, 1, p=start_weights))
        solution_space_initialized = True
    else:
        # Find the lowest entropy cell and select a label
        pass

    #


    #################
    # Window update #
    #################
    transform_labels_to_avg_rgb_value(input_array=possible_solution, output_array=average_label_pixel_values)

    pixel_canvas.set_pixel_values(pixel_values=average_label_pixel_values)
    pixel_canvas.update_pixels()
    root.update()
