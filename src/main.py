import tkinter as tk

import numpy as np
import numpy.typing as npt

from wave_function_collapse.pixel_canvas import PixelCanvas
from wave_function_collapse.types import LabelDescription, PropagationInstruction
from wave_function_collapse.utils import (
    set_intersection,
    string_to_rgb_array,
    transform_labels_to_avg_rgb_value,
)

pixel_size = 16
height, width = 16, 16
template_height, template_width = 9, 9

root = tk.Tk("Pixelart")

###################
# Template canvas #
###################
template_canvas = PixelCanvas(root=root, height=template_height, width=template_width, pixel_size=pixel_size, offset=2)

template_pixel_values_str = np.asarray(
    [
        ["#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000"],
        ["#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000"],
        ["#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000"],
        ["#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000"],
        ["#7311cc", "#ffffff", "#ffffff", "#ffffff", "#ffffff", "#ffffff", "#ffffff", "#ffffff", "#930055"],
        ["#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000"],
        ["#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000"],
        ["#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000"],
        ["#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000"],
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

top_boundary_condition: set[str] = set()
bottom_boundary_condition: set[str] = set()
left_boundary_condition: set[str] = set()
right_boundary_condition: set[str] = set()

for y, x in np.ndindex((template_height, template_width)):
    if y - 1 >= 0:
        label_map[template_pixel_values_str[y, x].item()]["up"].add(template_pixel_values_str[y - 1, x].item())
    else:
        top_boundary_condition.add(template_pixel_values_str[y, x].item())

    if y + 1 < template_height:
        label_map[template_pixel_values_str[y, x].item()]["down"].add(template_pixel_values_str[y + 1, x].item())
    else:
        bottom_boundary_condition.add(template_pixel_values_str[y, x].item())

    if x - 1 >= 0:
        label_map[template_pixel_values_str[y, x].item()]["left"].add(template_pixel_values_str[y, x - 1].item())
    else:
        left_boundary_condition.add(template_pixel_values_str[y, x].item())

    if x + 1 < template_width:
        label_map[template_pixel_values_str[y, x].item()]["right"].add(template_pixel_values_str[y, x + 1].item())
    else:
        right_boundary_condition.add(template_pixel_values_str[y, x].item())

top_left_corner_boundary_condition = set_intersection(top_boundary_condition, left_boundary_condition)
top_right_corner_boundary_condition = set_intersection(top_boundary_condition, right_boundary_condition)
bottom_left_corner_boundary_condition = set_intersection(bottom_boundary_condition, left_boundary_condition)
bottom_right_corner_boundary_condition = set_intersection(bottom_boundary_condition, right_boundary_condition)

#####################
# Initialize labels #
#####################
possible_solution: npt.NDArray[set[str]] = np.array([set(unique_labels) for _ in range(height * width)]).reshape(
    height, width
)
possible_solution[0, :] = top_boundary_condition
possible_solution[-1, :] = bottom_boundary_condition
possible_solution[:, 0] = left_boundary_condition
possible_solution[:, -1] = right_boundary_condition

possible_solution[0, 0] = top_left_corner_boundary_condition
possible_solution[0, -1] = top_right_corner_boundary_condition
possible_solution[-1, 0] = bottom_left_corner_boundary_condition
possible_solution[-1, -1] = bottom_right_corner_boundary_condition


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
        # Select a random coordinate
        le_y, le_x = (np.random.randint(low=0, high=height), np.random.randint(low=0, high=width))
        solution_space_initialized = True
    else:
        # Find the lowest entropy cell and select a label
        count_labels = lambda label_set: len(label_set)
        calculate_entropy = np.vectorize(count_labels)
        possible_solution_entropy = calculate_entropy(possible_solution)

        minimal_entropy_value = np.where(possible_solution_entropy > 1, possible_solution_entropy, np.inf).min()

        if minimal_entropy_value == np.inf:
            finished = True
            break

        minimal_entropy_indexes = np.where(possible_solution_entropy == minimal_entropy_value)
        index_selector = np.random.randint(low=0, high=len(minimal_entropy_indexes[0]))

        le_y = minimal_entropy_indexes[0][index_selector]
        le_x = minimal_entropy_indexes[1][index_selector]

    labels = list(possible_solution[le_y, le_x])
    weights = []
    weight_sum = 0
    for label in labels:
        weight = label_weights_map[label]
        weights.append(weight)
        weight_sum += weight

    if weight_sum < 1:
        modifier = 1 / weight_sum
        for i in range(len(weights)):
            weights[i] = weights[i] * modifier

    possible_solution[le_y, le_x] = set(np.random.choice(labels, 1, p=weights))

    # Propagate label changes
    propagation_queue: list[PropagationInstruction] = []
    processed_list: list[tuple[int, int]] = [(le_y, le_x)]

    if le_y - 1 >= 0:
        propagation_queue.append(PropagationInstruction(target=(le_y - 1, le_x), direction="up", origin=(le_y, le_x)))

    if le_y + 1 < height:
        propagation_queue.append(PropagationInstruction(target=(le_y + 1, le_x), direction="down", origin=(le_y, le_x)))

    if le_x - 1 >= 0:
        propagation_queue.append(PropagationInstruction(target=(le_y, le_x - 1), direction="left", origin=(le_y, le_x)))

    if le_x + 1 < width:
        propagation_queue.append(
            PropagationInstruction(target=(le_y, le_x + 1), direction="right", origin=(le_y, le_x))
        )

    while len(propagation_queue) > 0:
        propagation_instruction = propagation_queue.pop(0)
        target_y, target_x = propagation_instruction.target
        direction = propagation_instruction.direction
        origin = propagation_instruction.origin

        target_labels = set(possible_solution[target_y, target_x])
        origin_labels = set(possible_solution[origin])
        allowed_labels: set[str] = set()

        for label in origin_labels:
            allowed_labels.update(label_map[label][direction])

        intersection = set_intersection(target_labels, allowed_labels)

        if intersection == target_labels:
            processed_list.append((target_y, target_x))
            continue

        possible_solution[target_y, target_x] = intersection
        processed_list.append((target_y, target_x))

        neighbor_up = (target_y - 1, target_x)
        neighbor_down = (target_y + 1, target_x)
        neighbor_left = (target_y, target_x - 1)
        neighbor_right = (target_y, target_x + 1)

        if neighbor_up[0] >= 0 and neighbor_up not in processed_list:
            propagation_queue.append(
                PropagationInstruction(target=neighbor_up, direction="up", origin=(target_y, target_x))
            )

        if neighbor_down[0] < height and neighbor_down not in processed_list:
            propagation_queue.append(
                PropagationInstruction(target=neighbor_down, direction="down", origin=(target_y, target_x))
            )

        if neighbor_left[1] >= 0 and neighbor_left not in processed_list:
            propagation_queue.append(
                PropagationInstruction(target=neighbor_left, direction="left", origin=(target_y, target_x))
            )

        if neighbor_right[1] < width and neighbor_right not in processed_list:
            propagation_queue.append(
                PropagationInstruction(target=neighbor_right, direction="right", origin=(target_y, target_x))
            )

    #################
    # Window update #
    #################
    transform_labels_to_avg_rgb_value(input_array=possible_solution, output_array=average_label_pixel_values)

    pixel_canvas.set_pixel_values(pixel_values=average_label_pixel_values)
    pixel_canvas.update_pixels()
    root.update()

transform_labels_to_avg_rgb_value(input_array=possible_solution, output_array=average_label_pixel_values)

pixel_canvas.set_pixel_values(pixel_values=average_label_pixel_values)
pixel_canvas.update_pixels()
root.update()

print("Solved!")
root.mainloop()
