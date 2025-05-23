# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:19:24 2025

@author: coldatoms
"""

#trying to plot gradient of pngs 

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
import os 

# Function to load images and calculate their gradient
def load_images_and_calculate_gradient(image_directory, file_pattern):
    image_paths = glob.glob(file_pattern)  # Get all matching image paths
    images = []

    # Loop over the image files and load them
    for image_path in image_paths:
        # print(f"Trying to load image from: {image_path}")
        try:
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            images.append(np.array(image))  # Convert to numpy array and add to list
        except FileNotFoundError:
            print(f"Warning: {image_path} not found. Skipping.")

    if not images:
        print(f"No images were loaded. Please check the file paths of {file_pattern}.")
        return None

    # Stack the images and compute the average
    average_image = np.mean(images, axis=0)

    # Calculate the gradient in both X and Y directions of the average image
    gradient_x = np.gradient(average_image, axis=1)  # Gradient along X-axis
    gradient_y = np.gradient(average_image, axis=0)  # Gradient along Y-axis

    # Calculate the magnitude of the gradient
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Normalize the gradient magnitude to [0, 255] range
    gradient_magnitude_normalized = (gradient_magnitude / gradient_magnitude.max()) * 255
    gradient_magnitude_normalized = gradient_magnitude_normalized.astype(np.uint8)

    return gradient_magnitude_normalized

# List of directories for multiple sets of images
image_directories = [
    r'E:\Data\2025\03 March2025\07March2025\D_shutter_closed_nolight\plots',
    r'E:\Data\2025\03 March2025\10March2025\A_TShots\plots',
]

# List to hold the gradient images from each directory
gradient_images = []

# Loop through each directory, calculate the gradient, and store it
for image_directory in image_directories:
    # Generate the file pattern for the images in the directory
    filedir = os.path.dirname(image_directory)
    folder = os.path.basename(os.path.dirname(image_directory))
    file_pattern = os.path.join(image_directory, folder + '_*_bg.png')  # Adjust the pattern as needed

    # Load images and calculate gradients for the current directory
    gradient_image = load_images_and_calculate_gradient(image_directory, file_pattern)

    if gradient_image is not None:
        gradient_images.append(gradient_image)
    else:
        print(f"Skipping directory {image_directory} due to missing images or errors.")

# If there are enough gradient images, process them
if len(gradient_images) > 1:
    # Create titles dynamically based on directory names
    dir_names = [os.path.basename(os.path.dirname(dir)) for dir in image_directories]  # Extract last part of directory paths

    # Show the gradient images for each set
    fig, axes = plt.subplots(1, len(gradient_images) + 1, figsize=(18, 6))

    # Loop through the gradient images and display them
    for i, grad_img in enumerate(gradient_images):
        axes[i].imshow(grad_img, cmap='gray')
        axes[i].set_title(f'Gradient Magnitude of {dir_names[i]}')
        axes[i].axis('off')

    # Subtract the gradients pairwise and show the difference
    for i in range(len(gradient_images) - 1):
        # Subtract the gradients of the two images
        gradient_difference = np.abs(gradient_images[i] - gradient_images[i+1])

        # Display the difference
        axes[-1].imshow(gradient_difference, cmap='hot')
        axes[-1].set_title(f'Gradient Difference ({dir_names[i]} - {dir_names[i+1]})')
        axes[-1].axis('off')

    plt.show()
else:
    if gradient_image_1 is None:
        print("Could not calculate the gradients for the first set of images.")

    if gradient_image_2 is None:
        print("Could not calculate the gradients for the second set of images.")
