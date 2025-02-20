# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:14:47 2024

@author: Habib
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import rasterio

from core.models.model_factory import Model
from core.data_provider import datasets_factory
from core.utils import preprocess

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
import warnings

warnings.filterwarnings("ignore")


class Configs:
    def __init__(self):
        self.is_training = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "predrnn"
        self.pretrained_model = "./pretrained/model.ckpt-80000"
        self.input_folder = os.getenv(
            "INPUT_FOLDER", "./input"
        )  # Ambil dari environment variable
        self.img_width = None
        self.img_height = None
        self.img_channel = 1
        self.input_length = 10
        self.total_length = 20
        self.num_hidden = "128,128,128,128"
        self.filter_size = 3
        self.stride = 1
        self.layer_norm = 0
        self.patch_size = 5
        self.batch_size = 1
        self.reverse_input = 1
        self.scheduled_sampling = 1
        self.reverse_scheduled_sampling = 0
        self.sampling_stop_iter = 100
        self.sampling_start_value = 1.0
        self.sampling_changing_rate = 0.00002
        self.display_interval = 1
        self.test_interval = 1
        self.snapshot_interval = 1
        self.num_save_samples = 1
        self.save_dir = "./checkpoints"
        self.gen_frm_dir = os.getenv(
            "OUTPUT_FOLDER", "./results"
        )  # Ambil dari environment variable
        self.save_output = 1
        self.lr = 0.0003
        self.beta1 = 0.9


def load_geotiff_images(folder_path):
    images = []
    image_paths = []

    # Get all files with .tif or .tiff extension in the folder
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith((".tif", ".tiff")):
            image_paths.append(os.path.join(folder_path, file_name))

    # Sort files to ensure consistent order
    image_paths.sort()

    # Read each GeoTIFF file
    for img_path in image_paths:
        print(f"Reading file: {img_path}")
        with rasterio.open(img_path) as dataset:
            img_array = dataset.read(1)  # Read the first band
            img_array[img_array > 200] = 200
            img_array[img_array < 0] = 0
            images.append(img_array)

    images = np.stack(images, axis=0)  # Shape: [sequence_length, img_height, img_width]
    return images


def min_max_normalize(images, min_value=0, max_value=200):
    images = (images - min_value) / (max_value - min_value)
    images = np.clip(images, 0, 1)  # Ensure values are within [0, 1]
    return images


def preprocess_geotiff_images(images, configs):
    # If img_channel = 1, add channel dimension
    if configs.img_channel == 1:
        images = images[
            ..., np.newaxis
        ]  # Shape: [sequence_length, img_height, img_width, 1]
    elif configs.img_channel > 1:
        pass  # Adjust if using more than 1 channel

    return images.astype(np.float32)


def save_geotiff(data_array, output_path, reference_image):
    with rasterio.open(reference_image) as src:
        profile = src.profile
        # Update profile to ensure compatibility
        profile.update(dtype=rasterio.float32, count=1, nodata=0)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data_array.astype(rasterio.float32), 1)


def main():
    # Get configurations
    configs = Configs()

    # Create result directory if it doesn't exist
    if not os.path.exists(configs.gen_frm_dir):
        os.makedirs(configs.gen_frm_dir)

    # Move model to the appropriate device
    device = torch.device(configs.device)

    # Read and process GeoTIFF data
    input_folder = configs.input_folder  # Replace with your input folder path
    images = load_geotiff_images(input_folder)

    # Ensure enough images
    input_length = configs.input_length  # 10
    total_length = configs.total_length  # 30
    if images.shape[0] < input_length:
        raise ValueError(
            f"Not enough images. Found {images.shape[0]}, required {input_length}"
        )

    # Use the first 'input_length' images as input
    input_images = images[:input_length]

    # Ensure image dimensions
    img_height, img_width = input_images.shape[1], input_images.shape[2]
    configs.img_width = img_width
    configs.img_height = img_height

    # Initialize model after setting image dimensions
    model = Model(configs)
    model.load(configs.pretrained_model)
    model.network.eval()
    model.network.to(device)

    # Normalize images with Min-Max Scaler
    min_value = 0
    max_value = 200
    input_images = min_max_normalize(
        input_images, min_value=min_value, max_value=max_value
    )

    # Process images to match model input shape
    input_images = preprocess_geotiff_images(
        input_images, configs
    )  # Shape: [input_length, img_height, img_width, img_channel]

    # Create frames_tensor with shape [batch_size, total_length, img_height, img_width, img_channel]
    frames_tensor = np.zeros(
        (
            configs.batch_size,
            configs.total_length,
            img_height,
            img_width,
            configs.img_channel,
        ),
        dtype=np.float32,
    )

    # Copy input images into frames_tensor
    frames_tensor[0, :input_length] = input_images

    # Optionally, replicate the last input frame to fill the rest
    for t in range(input_length, total_length):
        frames_tensor[0, t] = input_images[-1]

    # Preprocessing data (reshape into patches)
    frames_tensor = preprocess.reshape_patch(frames_tensor, configs.patch_size)
    frames_tensor = torch.from_numpy(frames_tensor).float().to(device)  # Ensure float32

    # Create real_input_flag
    real_input_flag = np.zeros(
        (
            configs.batch_size,
            configs.total_length - configs.input_length,
            configs.img_height // configs.patch_size,
            configs.img_width // configs.patch_size,
            configs.patch_size**2 * configs.img_channel,
        )
    ).astype(np.float32)
    real_input_flag[:, : configs.input_length - 1, :, :] = 1.0
    real_input_flag = torch.from_numpy(real_input_flag).float().to(device)

    # Perform prediction
    with torch.no_grad():
        output = model.network(frames_tensor, real_input_flag)
        output_data = output[0]  # Extract next_frames from the tuple
        output_data = output_data.cpu().numpy()

    # Postprocessing data
    output_data = preprocess.reshape_patch_back(output_data, configs.patch_size)

    # Get predicted frames (after input_length)
    pred_frames = output_data[:, configs.input_length - 1 :, :, :, :]
    pred_frames[pred_frames < 0.01] = np.nan

    # Ensure predicted frames are within [0, 1]
    pred_frames = np.clip(pred_frames, 0, 1)

    # Define colormap and normalization
    cmap_colors = ["green", "yellow", "orange", "red"]
    cmap = ListedColormap(cmap_colors)
    cmap.set_bad(color="none")
    bounds = [0, 0.1, 0.5, 0.8, 1.0]
    denorm = np.array([0, 20, 100, 160, 200])
    norm = BoundaryNorm(bounds, cmap.N)

    # Latitude and Longitude bounds
    min_lat, max_lat = -9, -4
    min_lon, max_lon = 105, 110

    # Create latitude and longitude arrays
    latitudes = np.linspace(max_lat, min_lat, img_height)  # Ensure correct order
    longitudes = np.linspace(min_lon, max_lon, img_width)

    # Create coordinate meshgrid
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    # Save predicted frames as GeoTIFF and plot
    for i in range(pred_frames.shape[1]):
        frame = pred_frames[0, i, :, :, 0]  # Get first batch and first channel

        # Ensure values are within [0, 1]
        frame = np.clip(frame, 0, 1)

        # Save frame as GeoTIFF
        output_path = os.path.join(
            configs.gen_frm_dir, f"predicted_frame_{i+1:04d}.tiff"
        )
        # Use one of the input files as reference
        reference_image = os.path.join(input_folder, os.listdir(input_folder)[0])
        save_geotiff(frame, output_path, reference_image=reference_image)

        # Prepare data for plotting
        data = frame  # Data is already in [0, 1]

        # Create figure and axis with map projection
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Add map features
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS)
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

        # Plot data using pcolormesh
        mesh = ax.pcolormesh(
            lon_grid, lat_grid, data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree()
        )

        # Add colorbar
        cbar = plt.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.7, pad=0.05)
        cbar.set_ticklabels(denorm.astype(int))
        cbar.set_label("mm/10 menit")

        # Add title
        plt.title(f"Predicted Frame {i+1}")

        # Save plot
        plot_output_path = os.path.join(
            configs.gen_frm_dir, f"predicted_frame_{i+1:04d}.png"
        )
        plt.savefig(plot_output_path, dpi=300)
        plt.close()

    print("Prediction completed. Results saved in folder:", configs.gen_frm_dir)
    return pred_frames, input_images


if __name__ == "__main__":
    output, input_data = main()
