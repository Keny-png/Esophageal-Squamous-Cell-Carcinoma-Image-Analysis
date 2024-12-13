import torchvision
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display
import torchvision.transforms as transforms
import pandas as pd
from ultralytics import YOLO
from tqdm import trange


# Function to detect nuclei in immunostained images
def process_nuclear(image_rgb):
    # Define the stain matrix for H DAB
    stain_matrix = np.array([
        [0.650, 0.704, 0.286],
        [0.268, 0.570, 0.776],
        [0.711, 0.423, 0.561]
    ])

    # If the third row is all zeros, set it to the orthogonal complement of the other two
    if np.all(stain_matrix[2] == 0):
        stain_matrix[2] = np.cross(stain_matrix[0], stain_matrix[1])

    # Normalize each vector (column) in the matrix
    for i in range(3):
        norm = np.linalg.norm(stain_matrix[:, i])
        stain_matrix[:, i] /= norm

    # Invert the stain matrix
    inv_stain_matrix = np.linalg.pinv(stain_matrix)

    # Convert the RGB image into OD (Optical Density) space
    OD_image = -np.log((image_rgb.astype(np.float32) + 1) / 256)

    # Decompose the image in the new base
    deconvoluted_image = np.dot(OD_image.reshape((-1, 3)), inv_stain_matrix).reshape(image_rgb.shape)
    deconvoluted_image = np.clip(deconvoluted_image, 0, None)  # Clip negative values

    # Split into channels
    channel_1, channel_2, channel_3 = cv2.split(deconvoluted_image)

    # 1. Set low intensity pixels to 0 in the first channel
    non_zero_values = channel_1[channel_1 > 0]
    threshold_intensity = np.mean(non_zero_values)
    _, binary_image1 = cv2.threshold(channel_1, threshold_intensity, 255, cv2.THRESH_BINARY)

    # 2. Remove small regions in the first channel
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image1.astype(np.uint8))
    min_area = 0.001 * binary_image1.size
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            binary_image1[labels == i] = 0

    # 1. Set low intensity pixels to 0 in the second channel
    non_zero_values = channel_2[channel_1 > 0]
    threshold_intensity = np.mean(non_zero_values)
    _, binary_image2 = cv2.threshold(channel_2, threshold_intensity, 255, cv2.THRESH_BINARY)

    # 2. Remove small regions in the second channel
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image2.astype(np.uint8))
    min_area = 0.001 * binary_image2.size
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            binary_image2[labels == i] = 0

    # Identify the largest region in the first channel
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image1.astype(np.uint8))
    largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    largest_region_image1 = np.zeros_like(binary_image1)
    largest_region_image1[labels == largest_label] = 255

    # Identify the largest region in the second channel
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image2.astype(np.uint8))
    largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    largest_region_image2 = np.zeros_like(binary_image2)
    largest_region_image2[labels == largest_label] = 255

    largest_region_image1 = (largest_region_image1 / 255) * 2
    largest_region_image2 = (largest_region_image2 / 255)
    result = largest_region_image1 + largest_region_image2
    result[result == 1] = 0
    result[result == 2] = 255
    result[result == 3] = 0

    # Remove small regions under 20 pixels
    num_labels, labeled_result = cv2.connectedComponents(result.astype(np.uint8))
    label_areas = np.bincount(labeled_result.ravel())
    small_labels = np.where(label_areas <= 20)[0]
    for label in small_labels:
        result[labeled_result == label] = 0
    
    return result


# Load YOLO model
model = YOLO('/model/yolo_best.pt')

# Patch size and overlap definition
patch_size = 128
overlap = 64

# Folder path containing images
folder_path = "images/segment"
jpg_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
jpg_files = sorted(jpg_files)

# Process each image in the folder
for file_name in jpg_files:
    data_list = []
    img_path = os.path.join(folder_path, file_name)
    img = Image.open(img_path)

    # Get image dimensions
    img_width, img_height = img.size

    print(file_name)
    for y in trange(0, img_height, patch_size - overlap):
        for x in range(0, img_width, patch_size - overlap):
            patch = img.crop((x, y, x + patch_size, y + patch_size))

            # Skip patches with uniform color
            patch_color = np.array(patch).mean(axis=(0, 1))
            if np.all(patch_color == patch_color[0]):
                continue

            results = model(patch, imgsz=416, conf=0.001)
            tensor_label = results[0].boxes.shape
            if tensor_label[0] == 0:
                continue

            tensor_data = results[0].masks.data.numpy()
            tensor_scores = results[0].boxes.conf.numpy()
            tensor_boxes = results[0].boxes.xyxy.numpy()
            labels = tensor_label[0] - 1

            for i in range(labels):
                area = np.sum(tensor_data[i])
                scores = tensor_scores[i]
                boxes = tensor_boxes[i]

                xmin, ymin, xmax, ymax = boxes
                if xmin > 2 and xmax < 126 and ymin > 2 and ymax < 126 and scores > 0:
                    xmin, ymin, xmax, ymax = x + xmin, y + ymin, x + xmax, y + ymax
                    cropped_image = img.crop((xmin - 5, ymin - 5, xmax + 5, ymax + 5))
                    pixel_data = list(cropped_image.getdata())
                    white_pixel_count = sum(1 for pixel in pixel_data if pixel[0] == pixel[1] == pixel[2] == pixel[3])

                    data_list.append({'file_name': file_name, 'area': area, 'white_pixel': white_pixel_count, 'scores': scores, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})

    # Save data to CSV
    df = pd.DataFrame(data_list)
    df.to_csv(f'data/{file_name}.csv')
