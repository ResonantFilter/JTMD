import torch
import torchvision
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal



def show_bias_image_grid(
    dataset: Dataset,
    num_classes_to_show: int,
    bias_logic: Literal['class_equals_bias', 'bias_is_zero'] = 'class_equals_bias'
):
    print(f"Searching for samples with bias logic: '{bias_logic}'...")

    
    found_samples = {i: {"aligned": None, "conflicting": None} for i in range(num_classes_to_show)}
    
    num_slots_to_fill = num_classes_to_show * 2
    filled_slots = 0

    
    for image, (class_label, bias_label), _ in dataset:
    
        if filled_slots >= num_slots_to_fill:
            break
            
        if class_label < num_classes_to_show:
    
            is_aligned = False
            if bias_logic == 'class_equals_bias':
                is_aligned = (class_label == bias_label)
            elif bias_logic == 'bias_is_zero':
                is_aligned = (bias_label == 0)
            
            sample_type = "aligned" if is_aligned else "conflicting"

    
            if found_samples[class_label][sample_type] is None:
                found_samples[class_label][sample_type] = image
                filled_slots += 1

    image_grid_list = []
    placeholder_image = torch.zeros_like(dataset[0][0]) # A black image as placeholder

    for i in range(num_classes_to_show):
        aligned_img = found_samples[i]['aligned']
        conflicting_img = found_samples[i]['conflicting']

        image_grid_list.append(aligned_img if aligned_img is not None else placeholder_image)
        image_grid_list.append(conflicting_img if conflicting_img is not None else placeholder_image)

    grid = torchvision.utils.make_grid(image_grid_list, nrow=2, padding=4)
    
    plt.figure(figsize=(8, num_classes_to_show * 2))
    np_grid = grid.permute(1, 2, 0).numpy()
    
    plt.imshow(np_grid)
    plt.title(f"Image Grid (Bias Logic: {bias_logic})", fontsize=16)
    plt.axis('off')
    
    ax = plt.gca()
    
    img_size = placeholder_image.shape[2]
    padding = 4
    
    ax.text((img_size + padding)/2, -10, 'Bias-Aligned', ha='center', va='bottom', fontsize=12)
    ax.text(img_size + padding + (img_size + padding)/2, -10, 'Bias-Conflicting', ha='center', va='bottom', fontsize=12)

    for i in range(num_classes_to_show):
        y_pos = i * (img_size + padding) + (img_size / 2)
        ax.text(-20, y_pos, f"Class {i}", ha='right', va='center', fontsize=12, rotation=0)

    plt.show()