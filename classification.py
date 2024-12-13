from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import display
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import cv2
import os
import shutil

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

# Define data transformations for the test set
data_transforms = {
    "test": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}

# List of class names
class_names = ['BACK', 'DEB', 'GLND', 'SCC', 'STR']

# Initialize the VGG19 model
model_ft = models.vgg19(pretrained=False)

# Replace the output layer to match the number of classes in the pre-trained model
num_features = model_ft.classifier[6].in_features
model_ft.classifier[6] = nn.Linear(num_features, 9)

# Load model parameters
model_param = torch.load('model/col_classification.pth')
model_ft.load_state_dict(model_param)

for param in model_ft.parameters():
    param.requires_grad = False  # Disable gradient calculation

# Replace the output layer for the dataset's specific number of classes
num_features = model_ft.classifier[6].in_features
model_ft.classifier[6] = nn.Linear(num_features, len(class_names))

# Transfer the model to the computation device
model_ft = model_ft.to(device)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Select the optimization method
optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Load model parameters
model_param = torch.load('model/classification.pth')
model_ft.load_state_dict(model_param)


# Function to generate image patches
def generate_patches(input_image_path, patch_path, patch_size):
    # List to store patch filenames and coordinates
    patch_info_list = []

    # Read the image
    img = cv2.imread(input_image_path)

    # Get image height and width
    h, w = img.shape[:2]

    # Calculate the number of patches
    n_patches_h = h // patch_size
    n_patches_w = w // patch_size

    # Generate patches
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            # Extract a patch
            patch = img[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]

            # Create an output filename
            output_file_name = os.path.splitext(os.path.basename(input_image_path))[0] + f"_patch_{i * n_patches_w + j}.png"
            output_file_path = os.path.join(patch_path, output_file_name)

            # Save the patch
            cv2.imwrite(output_file_path, patch)

            # Add patch filename and coordinates to the list
            patch_info = {
                "file_name": output_file_name,
                "top_left": (j * patch_size, i * patch_size),
                "bottom_right": ((j + 1) * patch_size, (i + 1) * patch_size)
            }
            patch_info_list.append(patch_info)

    return patch_info_list


# Function to get prediction probabilities for an image
def show_prediction(model, transform, image_path):
    img = Image.open(image_path)
    inputs = transform(img).unsqueeze(dim=0).to(device)

    with torch.no_grad():
        outputs = model(inputs)
        predicted_probs = torch.nn.functional.softmax(outputs, dim=1)[0].tolist()

    return predicted_probs


# Function to process the image and map predictions
def process_image(image_path, patch_info_list, folder_path, output_folder):
    # Load the image
    image = Image.open(image_path)
    width, height = image.size

    # Create an array for image mapping, initialized to white
    image_map = np.ones((height, width, 3), dtype=np.uint8) * 255

    # List to store average colors of 'BACK' class patches
    back_colors = []

    # Process each patch and display prediction results
    for patch_info in patch_info_list:
        patch_file_path = os.path.join(patch_path, patch_info["file_name"])
        predicted_probs = show_prediction(model_ft, data_transforms["test"], patch_file_path)

        # Find the class with the highest prediction probability
        max_prob_index = predicted_probs.index(max(predicted_probs))
        max_prob_class = class_names[max_prob_index]

        # Set the color for the mapping
        color = (0, 0, 0) if max_prob_class == 'BACK' else (255, 0, 0)

        if max_prob_class == 'BACK':
            image = Image.open(patch_file_path)
            image_array = np.array(image)
            back_color = np.mean(image_array, axis=(0, 1))
            back_colors.append(back_color)

        # Apply the color to the corresponding patch in the image map
        top_left = patch_info["top_left"]
        image_map[top_left[1]:top_left[1] + 128, top_left[0]:top_left[0] + 128] = color

        # Add the class with the highest probability to the patch info
        patch_info["max_prob_class"] = max_prob_class

    # Save the processed image
    output_image_path = os.path.join(output_folder, os.path.basename(image_path))
    image = Image.fromarray(image_map)
    image.save(output_image_path)

    # Delete patch files after processing
    for patch_info in patch_info_list:
        patch_file_path = os.path.join(folder_path, patch_info["file_name"])
        os.remove(patch_file_path)


# Path definitions
input_folder_path = "./images/SCC"
output_folder_path = "./images/classification"
patch_path = "./images/patch"
patch_size = 128

# Process all PNG images in the folder
for filename in os.listdir(input_folder_path):
    if filename.endswith(".png"):
        image_path = os.path.join(input_folder_path, filename)
        patch_info_list = generate_patches(image_path, patch_path, patch_size)
        process_image(image_path, patch_info_list, patch_path, output_folder_path)
