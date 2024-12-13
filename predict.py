import os
import time
import copy
from collections import defaultdict
import torch
import shutil
import pandas as pd
from skimage import io, transform
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
import cv2
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torch.nn.functional as F
from PIL import Image
from torch import nn
import zipfile
import random
import torchvision.models as models
import segmentation_models_pytorch as smp
from skimage import io, transform
from skimage.measure import label, regionprops

# Set device to MPS if available, otherwise use CPU
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

TRAIN_PATH = './'


def load_ckp(checkpoint_path, model, optimizer):
    """
    Load model checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer): The optimizer to load state into.

    Returns:
        model, optimizer, start_epoch, valid_loss_min
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    valid_loss_min = checkpoint['valid_loss_min']
    return model, optimizer, start_epoch, valid_loss_min


# Model settings
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['building']
ACTIVATION = 'sigmoid'  # Could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cpu'

# Create a segmentation model with a pretrained encoder
model = smp.UnetPlusPlus(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

# Get preprocessing function for the specified encoder
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
valid_loss_min = float('inf')

best_model_path = 'model/UnetPlus.pt'

# Load the model checkpoint
model, optimizer, start_epoch, valid_loss_min = load_ckp(best_model_path, model, optimizer)


# Function for image data augmentation
def get_train_transform():
    return A.Compose(
        [
            # Resize (optional, depending on if resizing is already applied)
            # A.Resize(256, 256),
            # Normalization with default values from albumentations
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]
    )


# Custom Dataset class for test data
class LoadTestDataSet(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.image_folder = os.path.join(self.path, 'images/classification')
        self.image_files = [f for f in os.listdir(self.image_folder) if not f.startswith('.')]
        self.transforms = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_file)

        img = io.imread(image_path)[:, :, :3].astype('float32')
        img = transform.resize(img, (256, 256))

        augmented = self.transforms(image=img)
        img = augmented['image']
        return img, image_file


test_dataset = LoadTestDataSet(TRAIN_PATH, transform=get_train_transform())
test_loader = DataLoader(dataset=test_dataset, batch_size=10)


def remove_small_regions(mask, threshold_area):
    """
    Remove regions in the mask smaller than the threshold area.

    Args:
        mask (numpy.ndarray): Binary mask.
        threshold_area (int): Area threshold for removal.

    Returns:
        numpy.ndarray: Processed mask with small regions removed.
    """
    labeled_mask = label(mask)
    for region in regionprops(labeled_mask):
        if region.area <= threshold_area:
            labeled_mask[labeled_mask == region.label] = 0
        else:
            labeled_mask[labeled_mask == region.label] = 1
    return labeled_mask


def save_predicted_masks(model, output_folder):
    """
    Save predicted masks for test images.

    Args:
        model (torch.nn.Module): The trained segmentation model.
        output_folder (str): Path to the folder where masks will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with torch.no_grad():
        for data, filenames in test_loader:
            data = torch.autograd.Variable(data, volatile=True)
            output_masks = model(data)

            for idx in range(len(output_masks)):
                predicted_mask = output_masks[idx][0].cpu().numpy()
                filename = filenames[idx]

                # Binarize the mask
                predicted_mask = (predicted_mask > 0.5).astype(np.uint8)

                # Remove small regions with area below threshold
                predicted_mask = remove_small_regions(predicted_mask, threshold_area=100)

                # Invert the mask
                predicted_mask = 1 - predicted_mask

                # Remove small regions again with a smaller threshold
                predicted_mask = remove_small_regions(predicted_mask, threshold_area=50)

                # Invert the mask again
                predicted_mask = 1 - predicted_mask

                # Save the predicted mask
                mask_path = os.path.join(output_folder, filename)
                plt.imsave(mask_path, predicted_mask, cmap='gray')


# Specify the folder to save predicted masks
output_folder = 'images/predict'
save_predicted_masks(model, output_folder)
