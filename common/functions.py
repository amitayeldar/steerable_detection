#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 12:50:30 2024

@author: Amitay Eldar and Keren Mor
"""
# import os
# use_gpu=0
# if use_gpu==0:
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU before TensorFlow is imported
#     import tensorflow as tf  # Now import TensorFlow
import numpy as np
import os.path
from aspire.image import *
from aspire.basis import Coef, FBBasis2D
from aspire.image import Image as AspireImage
# import matplotlib.pyplot as plt
# from numba import jit, prange
import tensorflow as tf
from scipy import signal
from matplotlib.patches import Circle
import mrcfile
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import logging
import os
import pandas as pd
from scipy.sparse.linalg import eigsh
import torch.nn.functional as F
import torch
# Redirect logging to nowhere
logging.getLogger('aspire').handlers = [logging.FileHandler(os.devnull)]

def downsample(image, size_out):
    """
    Use Fourier methods to change the sample interval and/or aspect ratio
    of any dimensions of the input image.
    """
    x = np.fft.fftshift(np.fft.fft2(image))
    # crop x:
    nx, ny = x.shape
    nsx = int(np.floor(nx/2) - np.floor(size_out[1]/2))
    nsy = int(np.floor(ny/2) - np.floor(size_out[0]/2))
    fx = x[nsx : nsx + size_out[1], nsy : nsy + size_out[0]]
    output = np.fft.ifft2(np.fft.ifftshift(fx)) * (np.prod(size_out) / np.prod(image.shape))
    return output.real


def extract_noise_patches_and_coor_return_min_variance_tf(noise_img_scaled, contamination_mask, box_sz,
                                                          num_of_patches=30):
    """
    Extract noise patches of size box_sz x box_sz from a micrograph, starting from the patches with the lowest variance.

    Returns:
    noise_patches    : ndarray : Array of extracted noise patches.
    patch_coordinates : list   : List of (row, col) center coordinates for the patches.
    """

    # Define half_box_sz depending on whether box_sz is even or odd

    if box_sz % 2 == 0:
        half_box_sz = box_sz // 2
    else:
        half_box_sz = (box_sz // 2) + 1

    x = 1
    # Create a kernel for averaging
    kernel = np.ones((box_sz, box_sz)) / (box_sz * box_sz)
    # Create a kernel for averaging

    # Convert image to TensorFlow tensor
    noise_img_scaled_tf = tf.convert_to_tensor(noise_img_scaled, dtype=tf.float32)

    # Create an averaging kernel (box filter)
    kernel = np.ones((box_sz, box_sz), dtype=np.float32) / (box_sz * box_sz)
    kernel_tf = tf.convert_to_tensor(kernel, dtype=tf.float32)

    # Reshape to match TensorFlow's `conv2d` format
    noise_img_scaled_tf = tf.expand_dims(tf.expand_dims(noise_img_scaled_tf, axis=0), axis=-1)  # (1, H, W, 1)
    kernel_tf = tf.expand_dims(tf.expand_dims(kernel_tf, axis=-1), axis=-1)  # (box_sz, box_sz, 1, 1)

    # Perform convolution for mean (on CPU)
    mean_img_tf = tf.nn.conv2d(noise_img_scaled_tf, kernel_tf, strides=[1, 1, 1, 1], padding='SAME')

    # Perform convolution for mean of squared image (for variance calculation)
    mean_squared_img_tf = tf.nn.conv2d(tf.square(noise_img_scaled_tf), kernel_tf, strides=[1, 1, 1, 1], padding='SAME')

    # Compute the mean of each patch using convolution
    # mean_img = convolve2d(noise_img_scaled, kernel, mode='valid')

    # Compute the mean of the squared image (required for variance calculation)
    # mean_squared_img = convolve2d(noise_img_scaled ** 2, kernel, mode='same', boundary='symm')
    # Convert image to TensorFlow tensor
    # noise_img_scaled_tf = tf.convert_to_tensor(noise_img_scaled, dtype=tf.float32)

    # # Create an averaging kernel (box filter)
    # kernel = np.ones((box_sz, box_sz), dtype=np.float32) / (box_sz * box_sz)
    # kernel_tf = tf.convert_to_tensor(kernel, dtype=tf.float32)

    # # Reshape to match TensorFlow's `conv2d` format
    # noise_img_scaled_tf = tf.expand_dims(tf.expand_dims(noise_img_scaled_tf, axis=0), axis=-1)  # (1, H, W, 1)
    # kernel_tf = tf.expand_dims(tf.expand_dims(kernel_tf, axis=-1), axis=-1)  # (box_sz, box_sz, 1, 1)

    # # Perform convolution for mean (uses GPU)
    # mean_img_tf = tf.nn.conv2d(noise_img_scaled_tf, kernel_tf, strides=[1, 1, 1, 1], padding='SAME')

    # # Perform convolution for mean of squared image (for variance calculation)
    # mean_squared_img_tf = tf.nn.conv2d(tf.square(noise_img_scaled_tf), kernel_tf, strides=[1, 1, 1, 1], padding='SAME')

    # Convert results back to NumPy
    mean_img_np = mean_img_tf.numpy().squeeze()  # Remove extra dimensions
    mean_squared_img_np = mean_squared_img_tf.numpy().squeeze()
    # Variance formula: E[x^2] - (E[x])^2
    Y_var = mean_squared_img_np - mean_img_np ** 2

    # Exclude boundaries by setting variance to infinity
    Y_var[:half_box_sz, :] = np.inf
    Y_var[:, :half_box_sz] = np.inf
    Y_var[-half_box_sz:, :] = np.inf
    Y_var[:, -half_box_sz:] = np.inf

    noise_patches = []
    patch_coordinates = []  # Store coordinates
    cnt_p = 0

    while cnt_p < num_of_patches:
        p_min = np.min(Y_var)  # Get the minimum variance now
        if p_min == np.inf:
            break
        i_row, i_col = np.unravel_index(np.argmin(Y_var), Y_var.shape)  # Index of the minimum variance value
        # Extract patch, ensuring it remains box_sz x box_sz
        if box_sz % 2 == 0:
            if np.any(contamination_mask[
                max(0, i_row - half_box_sz): min(i_row + half_box_sz, noise_img_scaled.shape[0]),
                max(0, i_col - half_box_sz): min(i_col + half_box_sz, noise_img_scaled.shape[1])
            ] == 1):
                Y_var[max(0, i_row - half_box_sz): min(i_row + half_box_sz, noise_img_scaled.shape[0]),
                max(0, i_col - half_box_sz): min(i_col + half_box_sz, noise_img_scaled.shape[1])] = np.inf
                continue
            else:
                patch = noise_img_scaled[
                    max(0, i_row - half_box_sz): min(i_row + half_box_sz, noise_img_scaled.shape[0]),
                    max(0, i_col - half_box_sz): min(i_col + half_box_sz, noise_img_scaled.shape[1])]
        else:
            if np.any(contamination_mask[
                max(0, i_row - half_box_sz): min(i_row + half_box_sz - 1, noise_img_scaled.shape[0]),
                max(0, i_col - half_box_sz): min(i_col + half_box_sz - 1, noise_img_scaled.shape[1])
            ] == 1):
                Y_var[max(0, i_row - half_box_sz): min(i_row + half_box_sz - 1, noise_img_scaled.shape[0]),
                max(0, i_col - half_box_sz): min(i_col + half_box_sz - 1, noise_img_scaled.shape[1])] = np.inf
                continue
            else:
                patch = noise_img_scaled[
                    max(0, i_row - half_box_sz): min(i_row + half_box_sz - 1, noise_img_scaled.shape[0]),
                    max(0, i_col - half_box_sz): min(i_col + half_box_sz - 1, noise_img_scaled.shape[1])]

        # Skip invalid or out-of-bound patches
        if patch.shape != (box_sz, box_sz):
            Y_var[i_row, i_col] = np.inf  # Mask the invalid location
            continue

        # Append patch and coordinates
        noise_patches.append(patch)
        patch_coordinates.append((i_row, i_col))
        cnt_p += 1

        # Mask the region to avoid overlap with rDelAlgorithm
        row_start = max(0, i_row - box_sz)
        row_end = min(Y_var.shape[0], i_row + box_sz + 1)
        col_start = max(0, i_col - box_sz)
        col_end = min(Y_var.shape[1], i_col + box_sz + 1)
        Y_var[row_start:row_end, col_start:col_end] = np.inf  # Set the area to infinity

    return np.array(noise_patches), patch_coordinates

def extract_patches_from_coordinates(Y, object_coord_dir, microName, patch_size, obj_sz_real, obj_sz_down_scaled, flipud=False, downsample_fn=None):
    """
    Extract and downsample patches from the image based on coordinates in .star or .box files.

    Parameters:
    Y                 : ndarray : The input image (2D array, micrograph).
    object_coord_dir  : str     : Directory containing coordinate files.
    microName         : str     : Base name of the micrograph file.
    patch_size        : int     : Size of the square patch to extract.
    obj_sz_real       : int     : Original patch size.
    obj_sz_down_scaled: int     : Target size to downsample patches.
    flipud            : bool    : Whether the micrograph was flipped vertically.
    downsample_fn     : callable: Function to downsample images.

    Returns:
    patches : ndarray : Array of downsampled patches.
    """
    # Step 1: Locate the coordinate file
    mgScale = obj_sz_down_scaled / obj_sz_real
    coord_path = None
    file_type = None
    for ext in ['.star', '.box']:
        potential_file = os.path.join(object_coord_dir, f"{microName}{ext}")
        if os.path.exists(potential_file):
            coord_path = potential_file
            file_type = 'star' if ext == '.star' else 'box'
            print(f"[DEBUG] Found coordinate file: {coord_path} (type: {file_type})")
            break

    if not coord_path:
        raise FileNotFoundError(f"No coordinate file (.star or .box) found for {microName} in {object_coord_dir}")

    # Load .star file coordinates
    def load_star(file_path):
        coordinates = []
        with open(file_path, 'r') as f:
            inside_loop = False
            for line in f:
                if line.startswith('loop_'):
                    inside_loop = True
                    continue
                if inside_loop:
                    tokens = re.split(r'\s+', line.strip())
                    if len(tokens) >= 2:
                        try:
                            x = float(tokens[0])
                            y = float(tokens[1])
                            coordinates.append((x, y))
                        except ValueError:
                            continue
        return np.array(coordinates)

    # Load .box file coordinates
    def load_box(file_path):
        boxes = np.loadtxt(file_path, delimiter='\t', usecols=(0, 1, 2, 3))
        return [(x + w / 2, y + h / 2) for x, y, w, h in boxes]  # Convert top-left to center coordinates

    # Step 2: Load coordinates
    if file_type == "star":
        coordinates = load_star(coord_path)
        #coordinates = int(mgScale * coordinates)
    else:  # file_type == "box"
        coordinates = load_box(coord_path)
        scaled_coordinates = [(int(x * mgScale), int(y * mgScale)) for x, y in coordinates]
        coordinates = scaled_coordinates
    print(f"[DEBUG] Loaded {len(coordinates)} coordinates from {file_type} file: {coord_path}")

    # Step 3: Flip y-coordinates if necessary
    if flipud:
        print("[DEBUG] Flipping y-coordinates due to np.flipud...")
        coordinates = [(x, Y.shape[0] - y) for x, y in coordinates]

    # Step 4: Extract patches
    half_patch = patch_size / 2
    patches = []

    for i, (x, y) in enumerate(coordinates):
        # Define patch boundaries
        row_start = int(y - half_patch)
        row_end = int(y + half_patch)
        col_start = int(x - half_patch)
        col_end = int(x + half_patch)

        # Handle edges by clipping
        row_start = max(0, row_start)
        row_end = min(Y.shape[0], row_end)
        col_start = max(0, col_start)
        col_end = min(Y.shape[1], col_end)

        # Extract the patch
        patch = Y[row_start:row_end, col_start:col_end]

        # Check and append only valid patches
        if patch.shape == (patch_size, patch_size):
            # Downsample the patch
            mgScale = obj_sz_down_scaled / obj_sz_real
            dSampleSz = (int(np.floor(mgScale * patch.shape[0])), int(np.floor(mgScale * patch.shape[1])))
            #patch_downsampled = downsample_fn(patch, (dSampleSz[0], dSampleSz[1]))
            patches.append(patch)
        else:
            print(f"[DEBUG] Skipped patch {i} with shape {patch.shape} (not {patch_size}x{patch_size}).")

    print(f"Extracted and downsampled {len(patches)} valid patches from {coord_path}.")

    # Step 5: Visualization
    print("[DEBUG] Visualizing downsampled patches on the micrograph...")
    plt.figure(figsize=(10, 10))
    plt.imshow(Y, cmap='gray', origin='upper')

    for x, y in coordinates:
        x_top_left = x - half_patch
        y_top_left = y - half_patch
        rect = plt.Rectangle((x_top_left, y_top_left), patch_size, patch_size,
                             linewidth=1, edgecolor='green', facecolor='none')
        plt.gca().add_patch(rect)

    plt.title("Debug: Extracted Patch Boundaries")
    plt.show()

    return np.array(patches)


def extract_patches_from_any(Y, object_coord_dir, microName, patch_size, mgScale,contamination_mask):
    """
    Extract and downsample patches from the image based on coordinates in `.csv`, `.box`, or `.star` files.

    Parameters:
    Y            : ndarray : The input **downsampled** micrograph.
    object_coord_dir : str : Directory containing coordinate files.
    microName    : str     : Base name of the micrograph file.
    patch_size   : int     : Size of the square patch to extract.
    mgScale      : float   : Downsampling factor (obj_sz_down_scaled / obj_sz_real).

    Returns:
    patches : ndarray : Array of extracted patches.
    """

    # Step 1: Locate the coordinate file
    coord_path = None
    file_type = None
    for ext in ['.csv', '.box', '.star']:
        potential_file = os.path.join(object_coord_dir, f"{microName}{ext}")
        if os.path.exists(potential_file):
            coord_path = potential_file
            file_type = ext[1:]  # Remove the dot to store file type
            break

    if not coord_path:
        print(f"⚠ Skipping {microName}: No coordinate file (.csv, .box, .star) found in {object_coord_dir}")
        return []

    # Step 2: Load coordinates
    def load_csv(file_path):
        """ Load `.csv` file with (x_center, y_center, diameter). """
        data = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=(0, 1, 2))  # Read X, Y, Diameter
        return [(x, y, d, d) for x, y, d in data]  # Treat diameter as width=height

    def load_star(file_path):
        """ Load `.star` file with (x_center, y_center). """
        coordinates = []
        with open(file_path, 'r') as f:
            inside_loop = False
            for line in f:
                if line.startswith('loop_'):
                    inside_loop = True
                    continue
                if inside_loop:
                    tokens = re.split(r'\s+', line.strip())
                    if len(tokens) >= 2:
                        try:
                            x = float(tokens[0])
                            y = float(tokens[1])
                            coordinates.append((x, y, patch_size, patch_size))  # Assume patch_size as width=height
                        except ValueError:
                            continue
        return coordinates

    def load_box(file_path):
        """ Load `.box` file with top-left (x, y, width, height), convert to center. """
        boxes = np.loadtxt(file_path, usecols=(0, 1, 2, 3))  # Read x, y, width, height
        return [(x + w / 2, y + h / 2, w, h) for x, y, w, h in boxes]  # Convert top-left to center

    # Load the correct file type
    if file_type == "csv":
        coordinates = load_csv(coord_path)
    elif file_type == "star":
        coordinates = load_star(coord_path)
    else:  # file_type == "box"
        coordinates = load_box(coord_path)

    # ✅ Step 3: **Scale the coordinates**
    scaled_coordinates = [(x * mgScale, y * mgScale, w * mgScale, h * mgScale) for x, y, w, h in coordinates]

    # ✅ Step 4: **Flip Y-coordinates ONLY for `.csv` and `.star`, NOT for `.box`**
    image_height_down_scaled = Y.shape[0]  # Use Y.shape instead of extra parameter
    if file_type in ["csv", "star"]:
        flipped_coordinates = [(x, image_height_down_scaled - y, w, h) for x, y, w, h in scaled_coordinates]
    else:  # If `.box`, don't flip Y
        flipped_coordinates = scaled_coordinates

    # ✅ Step 5: **Extract patches using top-left corner**
    half_patch = patch_size / 2
    patches = []

    for i, (x, y, w, h) in enumerate(flipped_coordinates):
        # Define patch boundaries
        row_start = int(y - half_patch)
        row_end = int(y + half_patch)
        col_start = int(x - half_patch)
        col_end = int(x + half_patch)

        # Handle edges by clipping
        row_start = max(0, row_start)
        row_end = min(Y.shape[0], row_end)
        col_start = max(0, col_start)
        col_end = min(Y.shape[1], col_end)

        # Extract the patch
        if np.any(contamination_mask[row_start:row_end, col_start:col_end] == 1):
            continue
        else:
            patch = Y[row_start:row_end, col_start:col_end]

        # Validate patch size
        if patch.shape == (patch_size, patch_size):
            patches.append(patch)

    print(f"✅ Extracted {len(patches)} valid patches from {coord_path}.")

    # ✅ Step 6: Visualize Patches on Micrograph
    plt.figure(figsize=(10, 10))
    plt.imshow(Y, cmap='gray', origin='upper')

    # Draw extracted patches
    for x, y, w, h in flipped_coordinates:
        rect = plt.Rectangle((x - w / 2, y - h / 2), w, h, linewidth=1, edgecolor='green', facecolor='none')
        plt.gca().add_patch(rect)

    plt.title(f"Debug: Extracted Patch Boundaries ({file_type.upper()})")
    plt.show()

    return np.array(patches)


def fourier_bessel_pca_per_angle(noise_patches, fb_basis):
    num_img = noise_patches.shape[0]
    eigen_vectors_per_ang_lst = []
    mean_noise_per_ang_lst = []
    # get coeff_per_angular
    coeff_per_img = []
    coeff_per_angle = []
    for n_img in range(num_img):
        img = Image(noise_patches[n_img].astype(np.float32))
        coeff_per_img.append(fb_basis.evaluate_t(img))

    for n_ang in range(fb_basis.ell_max + 1):
        if n_ang == 0:
            s = 0
            e = fb_basis.k_max[0]
            coeff_arr = np.zeros((fb_basis.k_max[n_ang], num_img))
            for n_img in range(num_img):
                coeff_tmp = coeff_per_img[n_img]._data
                coeff_arr[:, n_img] = coeff_tmp[0][s:e]
            mean_noise_per_ang_lst.append(np.mean(coeff_arr, axis=1))
            cov_per_angle = np.cov(coeff_arr)
            eigen_values_per_angle, eigen_vectors_per_angle = np.linalg.eigh(cov_per_angle)
            sorted_indices = np.argsort(eigen_values_per_angle)[::-1]
            eigen_values_per_angle = eigen_values_per_angle[sorted_indices]
            eigen_vectors_per_angle = eigen_vectors_per_angle[:, sorted_indices]
            total_norm = np.sum(eigen_values_per_angle)
            cumulative_norm = np.cumsum(eigen_values_per_angle)
            l2_ratios = cumulative_norm / total_norm
            k = np.argmax(l2_ratios >= 0.9) + 1
            eigen_vectors_per_ang_lst.append(eigen_vectors_per_angle[:, :k])
        else:
            for i in range(2):
                coeff_arr = np.zeros((fb_basis.k_max[n_ang], num_img))
                s = e
                e += fb_basis.k_max[n_ang]
                for n_img in range(num_img):
                    coeff_tmp = coeff_per_img[n_img]._data
                    coeff_arr[:, n_img] = coeff_tmp[0][s:e]
                mean_noise_per_ang_lst.append(np.mean(coeff_arr, axis=1))
                cov_per_angle = np.cov(coeff_arr)
                eigen_values_per_angle, eigen_vectors_per_angle = np.linalg.eigh(cov_per_angle)
                sorted_indices = np.argsort(eigen_values_per_angle)[::-1]
                eigen_values_per_angle = eigen_values_per_angle[sorted_indices]
                eigen_vectors_per_angle = eigen_vectors_per_angle[:, sorted_indices]
                total_norm = np.sum(eigen_values_per_angle)
                cumulative_norm = np.cumsum(eigen_values_per_angle)
                l2_ratios = cumulative_norm / total_norm
                k = np.argmax(l2_ratios >= 0.9) + 1
                eigen_vectors_per_ang_lst.append(eigen_vectors_per_angle[:, :k])
    return eigen_vectors_per_ang_lst, mean_noise_per_ang_lst
def fourier_bessel_pca_per_angle_alpha(noise_patches, fb_basis):
    num_img = noise_patches.shape[0]
    eigen_vectors_per_ang_lst = []
    eigen_values_per_ang_lst = []
    mean_noise_per_ang_lst = []
    # get coeff_per_angular
    coeff_per_img = []
    coeff_per_angle = []
    for n_img in range(num_img):
        img = Image(noise_patches[n_img].astype(np.float32))
        coeff_per_img.append(fb_basis.evaluate_t(img))

    for n_ang in range(fb_basis.ell_max + 1):
        if n_ang == 0:
            s = 0
            e = fb_basis.k_max[0]
            coeff_arr = np.zeros((fb_basis.k_max[n_ang], num_img))
            for n_img in range(num_img):
                coeff_tmp = coeff_per_img[n_img]._data
                coeff_arr[:, n_img] = coeff_tmp[0][s:e]
            mean_noise_per_ang_lst.append(np.mean(coeff_arr, axis=1))
            cov_per_angle = np.cov(coeff_arr, bias=True)
            eigen_values_per_angle, eigen_vectors_per_angle = np.linalg.eigh(cov_per_angle)
            sorted_indices = np.argsort(eigen_values_per_angle)[::-1]
            eigen_values_per_angle = eigen_values_per_angle[sorted_indices]
            eigen_vectors_per_angle = eigen_vectors_per_angle[:, sorted_indices]
            # check pca
            # mean_norm = 0
            # for n in range(coeff_arr.shape[1]):
            #     mean_norm += np.linalg.norm(coeff_arr[:,n] - np.mean(coeff_arr, axis=1), axis=0)**2
            # mean_norm = mean_norm/coeff_arr.shape[1]
            total_norm = np.sum(eigen_values_per_angle)
            # print((total_norm-mean_norm)/mean_norm)
            cumulative_norm = np.cumsum(eigen_values_per_angle)
            l2_ratios = cumulative_norm / total_norm
            k = np.argmax(l2_ratios >= 0.999) + 1
            eigen_vectors_per_ang_lst.append(eigen_vectors_per_angle[:, :k])
            eigen_values_per_ang_lst.append(eigen_values_per_angle)
        else:
            for i in range(2):
                coeff_arr = np.zeros((fb_basis.k_max[n_ang], num_img))
                s = e
                e += fb_basis.k_max[n_ang]
                for n_img in range(num_img):
                    coeff_tmp = coeff_per_img[n_img]._data
                    coeff_arr[:, n_img] = coeff_tmp[0][s:e]
                mean_noise_per_ang_lst.append(np.mean(coeff_arr, axis=1))
                cov_per_angle = np.cov(coeff_arr, bias=True)
                eigen_values_per_angle, eigen_vectors_per_angle = np.linalg.eigh(cov_per_angle)
                sorted_indices = np.argsort(eigen_values_per_angle)[::-1]
                eigen_values_per_angle = eigen_values_per_angle[sorted_indices]
                eigen_vectors_per_angle = eigen_vectors_per_angle[:, sorted_indices]
                mean_norm = 0
                for n in range(coeff_arr.shape[1]):
                    mean_norm += np.linalg.norm(coeff_arr[:, n] - np.mean(coeff_arr, axis=1)) ** 2
                mean_norm = mean_norm / coeff_arr.shape[1]
                total_norm = np.sum(eigen_values_per_angle)
                print((total_norm - mean_norm) / mean_norm)
                cumulative_norm = np.cumsum(eigen_values_per_angle)
                l2_ratios = cumulative_norm / total_norm
                k = np.argmax(l2_ratios >= 0.999) + 1
                eigen_vectors_per_ang_lst.append(eigen_vectors_per_angle[:, :k])
                eigen_values_per_ang_lst.append(eigen_values_per_angle)
    return eigen_vectors_per_ang_lst, eigen_values_per_ang_lst, mean_noise_per_ang_lst
def compute_the_steerable_images_alpha(objects, obj_sz, fb_basis, eigen_vectors_per_ang_lst, eigen_values_per_ang_lst, mean_noise_per_ang_lst):
    # compute the radial Fourier component to each class average
    # get the coefficients of the class averages in the FB basis
    steerable_euclidian_l = np.zeros((obj_sz, obj_sz, 1 + 2 * (fb_basis.ell_max), objects.shape[0]))
    denoised_images = np.zeros(objects.shape)
    for n_img in range(objects.shape[0]):
        img = Image(objects[n_img].astype(np.float32))
        img_fb_coeff = fb_basis.evaluate_t(img)
        img_fb_coeff_denoise = img_fb_coeff.copy()
        v_0 = img_fb_coeff.copy()
        v_0._data[0, fb_basis.k_max[0]:] = 0
        v_0._data[0, :fb_basis.k_max[0]] = v_0._data[0, :fb_basis.k_max[0]] - mean_noise_per_ang_lst[0]
        # denoise the image by removing the noise component
        Q = eigen_vectors_per_ang_lst[0]
        object_coeff_eigen_noise = Q.T @ v_0._data[0, :fb_basis.k_max[0]]
        projected_obj = np.zeros(object_coeff_eigen_noise.shape)
        projected_noise = np.zeros(object_coeff_eigen_noise.shape)
        for n in range(object_coeff_eigen_noise.shape[0]-1):
            projected_obj[n] = np.sum(object_coeff_eigen_noise[n+1:]**2)
            projected_noise[n] = np.sum(eigen_values_per_ang_lst[0][n + 1:])
        projected_diff = projected_obj - projected_noise
        start_index = 0
        first_index = np.argmax(projected_diff[start_index:] > 0) if np.any(projected_diff[start_index:] > 0) else -1
        first_index = start_index + first_index
        if first_index == -1:
            v_0._data[0, :fb_basis.k_max[0]] = 0
        else:
            k = np.argmax(projected_diff[first_index:]) + first_index
            Q = Q[:, :k + 1]
            v_0._data[0, :fb_basis.k_max[0]] = v_0._data[0, :fb_basis.k_max[0]] - Q @ (
                    Q.T @ v_0._data[0, :fb_basis.k_max[0]])
            # print(k, 0)


        img_fb_coeff_denoise._data[0, :fb_basis.k_max[0]] = v_0._data[0, :fb_basis.k_max[0]]
        v_0_img = fb_basis.evaluate(v_0).asnumpy()[0]
        steerable_euclidian_l[:, :, 0, n_img] = fb_basis.evaluate(v_0)
        coeff_k_index_start = fb_basis.k_max[0]
        for m in range(1, fb_basis.ell_max + 1):
            l_idx = 2 * m - 1
            k_idx = fb_basis.k_max[m]
            coeff_k_index_end_cos = coeff_k_index_start + k_idx
            coeff_k_index_end_sin = coeff_k_index_end_cos + k_idx
            vcos = img_fb_coeff.copy()
            vcos._data[0, :coeff_k_index_start] = 0
            vcos._data[0, coeff_k_index_end_cos:] = 0
            # denoise the image by removing the noise component
            vcos._data[0, coeff_k_index_start:coeff_k_index_end_cos] = vcos._data[0,
                                                                       coeff_k_index_start:coeff_k_index_end_cos] - mean_noise_per_ang_lst[l_idx]
            Q = eigen_vectors_per_ang_lst[l_idx]
            object_coeff_eigen_noise = Q.T @ vcos._data[0, coeff_k_index_start:coeff_k_index_end_cos]
            projected_obj = np.zeros(object_coeff_eigen_noise.shape)
            projected_noise = np.zeros(object_coeff_eigen_noise.shape)
            for n in range(object_coeff_eigen_noise.shape[0] - 1):
                projected_obj[n] = np.sum(object_coeff_eigen_noise[n + 1:] ** 2)
                projected_noise[n] = np.sum(eigen_values_per_ang_lst[l_idx][n + 1:])
            projected_diff = projected_obj - projected_noise
            start_index = 0
            first_index = np.argmax(projected_diff[start_index:] > 0) if np.any(projected_diff[start_index:] > 0) else -1
            first_index = start_index + first_index
            if first_index == -1:
                vcos._data[0, coeff_k_index_start:coeff_k_index_end_cos] = 0
            else:
                k = np.argmax(projected_diff[first_index:])+first_index
                Q = Q[:, :k + 1]
                vcos._data[0, coeff_k_index_start:coeff_k_index_end_cos] = vcos._data[0,
                                                                           coeff_k_index_start:coeff_k_index_end_cos] - Q @ (
                                                                                   Q.T @ vcos._data[0,
                                                                                         coeff_k_index_start:coeff_k_index_end_cos])
                # print(k,l_idx)



            img_fb_coeff_denoise._data[0, coeff_k_index_start:coeff_k_index_end_cos] = vcos._data[0,
                                                                                       coeff_k_index_start:coeff_k_index_end_cos]
            vsin = img_fb_coeff.copy()
            vsin._data[0, :coeff_k_index_end_cos] = 0
            vsin._data[0, coeff_k_index_end_sin:] = 0
            # denoise the image by removing the noise component
            vsin._data[0, coeff_k_index_end_cos:coeff_k_index_end_sin] = vsin._data[0,
                                                                         coeff_k_index_end_cos:coeff_k_index_end_sin] - mean_noise_per_ang_lst[l_idx + 1]
            Q = eigen_vectors_per_ang_lst[l_idx + 1]
            object_coeff_eigen_noise = Q.T @ vsin._data[0, coeff_k_index_end_cos:coeff_k_index_end_sin]
            projected_obj = np.zeros(object_coeff_eigen_noise.shape)
            projected_noise = np.zeros(object_coeff_eigen_noise.shape)
            for n in range(object_coeff_eigen_noise.shape[0] - 1):
                projected_obj[n] = np.sum(object_coeff_eigen_noise[n + 1:] ** 2)
                projected_noise[n] = np.sum(eigen_values_per_ang_lst[l_idx][n + 1:])
            projected_diff = projected_obj - projected_noise
            start_index = 0
            first_index = np.argmax(projected_diff[start_index:] > 0) if np.any(projected_diff[start_index:] > 0) else -1
            first_index = start_index + first_index
            if first_index == -1:
                vsin._data[0, coeff_k_index_end_cos:coeff_k_index_end_sin] = 0
            else:
                k = np.argmax(projected_diff[first_index:]) + first_index
                Q = Q[:, :k + 1]
                vsin._data[0, coeff_k_index_end_cos:coeff_k_index_end_sin] = vsin._data[0,
                                                                             coeff_k_index_end_cos:coeff_k_index_end_sin] - Q @ (
                                                                                     Q.T @ vsin._data[0,
                                                                                           coeff_k_index_end_cos:coeff_k_index_end_sin])
                # print(k, l_idx)


            img_fb_coeff_denoise._data[0, coeff_k_index_end_cos:coeff_k_index_end_sin] = vsin._data[0,
                                                                                         coeff_k_index_end_cos:coeff_k_index_end_sin]

            vcos_img = fb_basis.evaluate(vcos).asnumpy()[0]
            vsin_img = fb_basis.evaluate(vsin).asnumpy()[0]
            steerable_euclidian_l[:, :, l_idx, n_img] = vcos_img
            steerable_euclidian_l[:, :, l_idx + 1, n_img] = vsin_img
            # steerable_euclidian_l[:,:,l_idx,n_img] = vcos_img + 1j*vsin_img
            # steerable_euclidian_l[:, :, l_idx+1, n_img] = vcos_img - 1j * vsin_img
            coeff_k_index_start = coeff_k_index_end_sin
        denoised_images[n_img, :, :] = fb_basis.evaluate(img_fb_coeff_denoise).asnumpy()[0]
    return steerable_euclidian_l, denoised_images
def compute_the_steerable_images(objects, obj_sz, fb_basis, eigen_vectors_per_ang_lst, mean_noise_per_ang_lst):
    # compute the radial Fourier component to each class average
    # get the coefficients of the class averages in the FB basis
    steerable_euclidian_l = np.zeros((obj_sz, obj_sz, 1 + 2 * (fb_basis.ell_max), objects.shape[0]))
    denoised_images = np.zeros(objects.shape)
    for n_img in range(objects.shape[0]):
        img = Image(objects[n_img].astype(np.float32))
        img_fb_coeff = fb_basis.evaluate_t(img)
        img_fb_coeff_denoise = img_fb_coeff.copy()
        v_0 = img_fb_coeff.copy()
        v_0._data[0, fb_basis.k_max[0]:] = 0
        # denoise the image by removing the noise component
        Q = eigen_vectors_per_ang_lst[0]
        v_0._data[0, :fb_basis.k_max[0]] = v_0._data[0, :fb_basis.k_max[0]] - mean_noise_per_ang_lst[0]
        v_0._data[0, :fb_basis.k_max[0]] = v_0._data[0, :fb_basis.k_max[0]] - Q @ (
                    Q.T @ v_0._data[0, :fb_basis.k_max[0]])
        img_fb_coeff_denoise._data[0, :fb_basis.k_max[0]] = v_0._data[0, :fb_basis.k_max[0]]
        v_0_img = fb_basis.evaluate(v_0).asnumpy()[0]
        steerable_euclidian_l[:, :, 0, n_img] = fb_basis.evaluate(v_0)
        coeff_k_index_start = fb_basis.k_max[0]
        for m in range(1, fb_basis.ell_max + 1):
            l_idx = 2 * m - 1
            k_idx = fb_basis.k_max[m]
            coeff_k_index_end_cos = coeff_k_index_start + k_idx
            coeff_k_index_end_sin = coeff_k_index_end_cos + k_idx
            vcos = img_fb_coeff.copy()
            vcos._data[0, :coeff_k_index_start] = 0
            vcos._data[0, coeff_k_index_end_cos:] = 0
            # denoise the image by removing the noise component
            vcos._data[0, coeff_k_index_start:coeff_k_index_end_cos] = vcos._data[0,
                                                                       coeff_k_index_start:coeff_k_index_end_cos] - \
                                                                       mean_noise_per_ang_lst[l_idx]
            Q = eigen_vectors_per_ang_lst[l_idx]
            vcos._data[0, coeff_k_index_start:coeff_k_index_end_cos] = vcos._data[0,
                                                                       coeff_k_index_start:coeff_k_index_end_cos] - Q @ (
                                                                                   Q.T @ vcos._data[0,
                                                                                         coeff_k_index_start:coeff_k_index_end_cos])
            img_fb_coeff_denoise._data[0, coeff_k_index_start:coeff_k_index_end_cos] = vcos._data[0,
                                                                                       coeff_k_index_start:coeff_k_index_end_cos]
            vsin = img_fb_coeff.copy()
            vsin._data[0, :coeff_k_index_end_cos] = 0
            vsin._data[0, coeff_k_index_end_sin:] = 0
            # denoise the image by removing the noise component
            vsin._data[0, coeff_k_index_end_cos:coeff_k_index_end_sin] = vsin._data[0,
                                                                         coeff_k_index_end_cos:coeff_k_index_end_sin] - \
                                                                         mean_noise_per_ang_lst[l_idx + 1]
            Q = eigen_vectors_per_ang_lst[l_idx + 1]
            vsin._data[0, coeff_k_index_end_cos:coeff_k_index_end_sin] = vsin._data[0,
                                                                         coeff_k_index_end_cos:coeff_k_index_end_sin] - Q @ (
                                                                                     Q.T @ vsin._data[0,
                                                                                           coeff_k_index_end_cos:coeff_k_index_end_sin])
            img_fb_coeff_denoise._data[0, coeff_k_index_end_cos:coeff_k_index_end_sin] = vsin._data[0,
                                                                                         coeff_k_index_end_cos:coeff_k_index_end_sin]

            vcos_img = fb_basis.evaluate(vcos).asnumpy()[0]
            vsin_img = fb_basis.evaluate(vsin).asnumpy()[0]
            steerable_euclidian_l[:, :, l_idx, n_img] = vcos_img
            steerable_euclidian_l[:, :, l_idx + 1, n_img] = vsin_img
            # steerable_euclidian_l[:,:,l_idx,n_img] = vcos_img + 1j*vsin_img
            # steerable_euclidian_l[:, :, l_idx+1, n_img] = vcos_img - 1j * vsin_img
            coeff_k_index_start = coeff_k_index_end_sin
        denoised_images[n_img, :, :] = fb_basis.evaluate(img_fb_coeff_denoise).asnumpy()[0]
    return steerable_euclidian_l, denoised_images

def compute_the_steerable_basis(steerable_euclidian_l):
    # Initialize an empty list to accumulate Q matrices
    steerable_basis_vectors_list = []

    for l in range(steerable_euclidian_l.shape[2]):
        l_images = steerable_euclidian_l[:, :, l, :]
        Q = qr_factorization_matrices(l_images)
        steerable_basis_vectors_list.append(Q)  # Add each Q matrix to the list

    # Concatenate all Q matrices horizontally at the end
    steerable_basis_vectors = np.hstack(steerable_basis_vectors_list)

    return steerable_basis_vectors

def qr_factorization_matrices(matrices):
    """
    Apply compact QR factorization to a 3D numpy array of matrices.

    :param matrices: 3D numpy array where each slice along the third dimension is a matrix to be orthogonalized.
    :return: Orthogonal matrix Q from compact QR factorization.
    """
    if not isinstance(matrices, np.ndarray) or matrices.ndim != 3:
        raise ValueError("Input must be a 3D numpy array.")

    # Flatten each matrix and stack them into a single 2D array
    num_matrices = matrices.shape[2]
    flattened_matrices = np.array([matrices[:, :, i].flatten() for i in range(num_matrices)]).T

    # Apply QR factorization (compact version)
    Q, R = np.linalg.qr(flattened_matrices)
    Q_dom = dominant_vectors(Q, R)
    return Q_dom

def dominant_vectors(Q, R):
    # Calculate the absolute maximum value on the diagonal of R
    max_diag_value = np.max(np.abs(np.diag(R)))

    # Set the threshold based on this maximum value
    adapted_threshold = 0.01 * max_diag_value

    # Determine dominant vectors based on this threshold
    dominant_indices = np.abs(np.diag(R)) > adapted_threshold
    return Q[:, dominant_indices]


def sort_steerable_basis_by_obj_then_snr(basis_full, objects, noise_patches, max_dimension):
    """
    Sorts steerable basis vectors based on projected SNR with sanity check.

    Parameters:
    basis_full : Full set of basis vectors (2D array)
    objects : Flattened objects (2D array)
    noise_patches : Flattened noise patches (2D array)
    max_dimension : Maximum dimension to sort

    Returns:
    sorted_basis_vectors : Sorted basis vectors (2D array)
    projected_snr_per_dim : Projected SNR per dimension (list)
    projected_snr_min_img : Indices of minimum SNR images (list)
    """
    ### first sort by object norm
    # Calculate coefficients (square of projections)
    coeff_objects = np.square((basis_full.T) @ objects)
    # coeff_noise = np.square((basis_full.T) @ noise_patches)

    # Full projected obj_norm
    projected_norm_objects = coeff_objects.sum(axis=0)
    min_idx_projected_obj = np.argmin(projected_norm_objects)

    projected_snr_per_dim = []
    projected_obj_per_dim = []

    sorted_basis_lst = []
    sorted_basis_idx = []
    projected_norm_objects = np.zeros(projected_norm_objects.shape)
    basis_idx = []

    # Initialize a mask to track used indices
    active_mask = np.ones(basis_full.shape[1], dtype=bool)

    # Loop to select max_dimension basis vectors
    for dim in range(max_dimension):
        cumulative_obj_norm = coeff_objects[:, min_idx_projected_obj] + projected_norm_objects[min_idx_projected_obj]
        cumulative_obj_norm[~active_mask] = -np.inf
        max_idx_candidate = np.argmax(cumulative_obj_norm)
        basis_idx.append(max_idx_candidate)
        # Update norms
        projected_norm_objects += coeff_objects[max_idx_candidate, :]

        # Append the sorted basis vector
        sorted_basis_lst.append(basis_full[:, max_idx_candidate])
        sorted_basis_idx.append(max_idx_candidate)

        # Mark this index as used
        active_mask[max_idx_candidate] = False
        #Recompute projected SNR

        min_idx_projected_snr = np.argmin(projected_norm_objects)
        projected_obj_per_dim.append(projected_norm_objects[min_idx_projected_snr])

    ### second sort by projected snr
    sorted_basis_vectors = np.column_stack(sorted_basis_lst)
    coeff_objects_sorted_basis = np.square((sorted_basis_vectors.T) @ objects)
    coeff_noise_sorted_basis = np.square((sorted_basis_vectors.T) @ noise_patches)
    for dim in range(max_dimension):
        numerator = np.sum(coeff_objects_sorted_basis[:dim+1,:], axis=0)
        denominator = np.mean(np.sum(coeff_noise_sorted_basis[:dim+1,:], axis=0))
        projected_snr_per_dim.append(np.min(numerator / denominator))


    num_of_basis = 0
    while num_of_basis <=10:
        num_of_basis = np.argmax(projected_snr_per_dim)
        if num_of_basis <=10:
            projected_snr_per_dim[num_of_basis] = 0

    return sorted_basis_vectors, num_of_basis, projected_snr_per_dim, basis_idx

def plot_patches_on_micrograph(Y, noise_patches_coor, patch_size, microName, output_folder=None):
    """
    Plot all extracted patches on the micrograph and optionally save the plot with the micrograph name.

    Parameters:
    Y : np.array
        The micrograph image (2D array).
    noise_patches_coor : list of tuples
        List of (row, col) coordinates for the center of each patch.
    patch_size : int
        Size of the square patches.
    microName : str
        Name of the micrograph (used in the output filename).
    output_folder : str, optional
        Path to the folder where the plot will be saved. If None, the plot will not be saved.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(Y, cmap='gray', origin='upper')

    for i_row, i_col in noise_patches_coor:
        # Draw each patch as a rectangle
        rect = patches.Rectangle(
            (i_col - patch_size // 2, i_row - patch_size // 2),  # Top-left corner
            patch_size,  # Width
            patch_size,  # Height
            linewidth=1.5,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

    plt.title(f"Patches on Micrograph: {microName}")
    plt.axis('off')

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
        filename = f"patches_{microName}.png"
        output_path = os.path.join(output_folder, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

def projected_noise_simulation_from_noise_patches_tf(noise_samples, basis):
    """
    Simulates projected noise using given noise samples and basis functions.

    Parameters:
    noise_samples : Array of noise samples.
    basis         : Array of basis functions.
    num_of_exp_noise : Number of noise experiments.

    Returns:
    S_z : Simulated noise projections.
    """
    num_of_exp_noise = noise_samples.shape[1]
    noise_samples = noise_samples.astype(np.float32)
    basis = basis.astype(np.float32)

    sz = int(np.sqrt(noise_samples.shape[0]))  # Image size
    noise_imgs = np.reshape(noise_samples, (num_of_exp_noise, sz, sz, 1)).astype(np.float32)  # Reshape noise samples to 4D (batch, height, width, channels)

    # Initialize output tensor S_z
    sz_pn = tf.nn.conv2d(
        tf.reshape(noise_samples[:, 0], (1, sz, sz, 1)),
        tf.reshape(basis[:, :, 0], (basis.shape[0], basis.shape[1], 1, 1)),
        strides=[1, 1, 1, 1],
        padding='VALID'
    ).shape[1]
    S_z_n = np.zeros((num_of_exp_noise, sz_pn, sz_pn), dtype=np.float32)
    # Flip basis stack
    flipped_basis_stack = tf.convert_to_tensor(
        np.stack([np.flip(np.flip(basis[:, :, j], 0), 1) for j in range(basis.shape[2])]),
        dtype=tf.float32
    )

    flipped_basis_stack = tf.reshape(flipped_basis_stack, [basis.shape[0], basis.shape[1], 1, basis.shape[2]])

    # Reshape the array to add the channel dimension: (num_of_exp_noise, sz, sz, 1)
    noise_imgs_tf = tf.convert_to_tensor(noise_imgs, dtype=tf.float32)
    conv_result = tf.nn.conv2d(noise_imgs_tf, flipped_basis_stack, strides=[1, 1, 1, 1], padding='VALID')**2
    conv_result = tf.reduce_sum(conv_result, axis=-1)
    S_z_n += tf.squeeze(conv_result).numpy()
    S_z_n = np.transpose(S_z_n, (1, 2, 0))
    return S_z_n

def projected_noise_simulation_from_noise_patches_para_fast_tf(noise_samples, basis, num_of_exp_noise):
    """
    Simulates projected noise using given noise samples and basis functions in batches,
    distributing computation across multiple GPUs.
    """

    noise_samples = noise_samples.astype(np.float32)
    basis = basis.astype(np.float32)

    sz = int(np.sqrt(noise_samples.shape[0]))  # Image size

    # ✅ Dynamically determine batch size based on available GPU memory
    def get_optimal_batch_size(base_size=1000):
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return base_size  # Default batch size for CPU

        try:
            for gpu in gpus:
                details = tf.config.experimental.get_memory_info(gpu.name)
                free_memory = details['current']  # Get available memory
                batch_size = max(1, int(free_memory / (sz * sz * 4)))  # Compute batch size
                print(f"Adjusted batch size: {batch_size}")
                return batch_size
        except:
            pass  # If memory info isn't available, fallback to base size
        return base_size

    batch_size = get_optimal_batch_size()
    num_batches = (num_of_exp_noise + batch_size - 1) // batch_size  # Compute number of batches

    # List available GPUs
    x=1
    gpus = tf.config.list_physical_devices('GPU')
    num_gpus = len(gpus)

    if num_gpus == 0:
        raise RuntimeError("No GPUs found! Consider using a CPU-based approach.")

    print(f"Using {num_gpus} GPUs for parallel processing. Batch size: {batch_size}")

    # Compute output size
    sz_pn = tf.nn.conv2d(
        tf.reshape(noise_samples[:, 0], (1, sz, sz, 1)),
        tf.reshape(basis[:, :, 0], (basis.shape[0], basis.shape[1], 1, 1)),
        strides=[1, 1, 1, 1],
        padding='VALID'
    ).shape[1]

    # Initialize output tensor
    S_z_n = np.zeros((num_of_exp_noise, sz_pn, sz_pn), dtype=np.float32)

    # Flip basis stack
    flipped_basis_stack = tf.convert_to_tensor(
        np.stack([np.flip(np.flip(basis[:, :, j], 0), 1) for j in range(basis.shape[2])]),
        dtype=tf.float32
    )

    flipped_basis_stack = tf.reshape(flipped_basis_stack, [basis.shape[0], basis.shape[1], 1, basis.shape[2]])

    # Function to process a batch
    @tf.function
    def process_batch(noise_imgs_tf, flipped_basis_stack):
        conv_result = tf.nn.conv2d(noise_imgs_tf, flipped_basis_stack, strides=[1, 1, 1, 1], padding='VALID') ** 2
        conv_result = tf.reduce_sum(conv_result, axis=-1)
        return conv_result

    results = [None] * num_batches

    # Process batches
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, num_of_exp_noise)

        gpu_idx = batch_idx % num_gpus
        device_name = f"/GPU:{gpu_idx}"

        print(f"Processing batch {batch_idx+1}/{num_batches} on {device_name}")

        with tf.device(device_name):
            noise_batch = noise_samples[:, start:end]
            noise_imgs = np.reshape(noise_batch, (end - start, sz, sz, 1)).astype(np.float32)
            noise_imgs_tf = tf.convert_to_tensor(noise_imgs, dtype=tf.float32)

            results[batch_idx] = process_batch(noise_imgs_tf, flipped_basis_stack)

            # Clear GPU memory
            del noise_imgs_tf
            tf.keras.backend.clear_session()

    # Collect results
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, num_of_exp_noise)
        S_z_n[start:end, :, :] += results[batch_idx].numpy()

    return np.transpose(S_z_n, (1, 2, 0))  # Match expected output shape


def projected_noise_simulation_from_noise_patches_scipy(noise_samples, basis):
    noise_samples = noise_samples.astype(np.float32)
    basis = basis.astype(np.float32)
    conv_sz_valid = int(np.sqrt(noise_samples.shape[0])) - basis.shape[0] + 1
    img_sz = int(np.sqrt(noise_samples.shape[0]))
    S_z = np.zeros((conv_sz_valid, conv_sz_valid,noise_samples.shape[1]), dtype=np.float32)
    flipped_basis = np.zeros(basis.shape)
    for n in range(basis.shape[2]):
        flipped_basis[:,:,n] = np.flip(np.flip(basis[:,:,n], axis=0), axis=1)
    for i in range(noise_samples.shape[1]):
        for j in range(basis.shape[2]):
            S_z[:, :, i] += signal.convolve2d(np.reshape(noise_samples[:, i], (img_sz, img_sz)), flipped_basis[:, :, j],mode='valid') ** 2
    return S_z


def projected_noise_simulation_from_noise_patches_torch(noise_samples, basis, device='cuda'):
    noise_samples_np = noise_samples.astype(np.float32)
    basis_np = basis.astype(np.float32)
    conv_sz_valid = int(np.sqrt(noise_samples_np.shape[0])) - basis_np.shape[0] + 1
    img_sz = int(np.sqrt(noise_samples_np.shape[0]))

    flipped_basis_np = np.zeros_like(basis_np)
    for n in range(basis_np.shape[2]):
        flipped_basis_np[:, :, n] = np.flip(basis_np[:, :, n], axis=(0, 1))

    S_z_torch = torch.zeros((conv_sz_valid, conv_sz_valid, noise_samples.shape[1]), dtype=torch.float32, device=device)

    for i in range(noise_samples_np.shape[1]):
        if i % 100 == 0:
            print(f"Processing noise sample {i+1}/{noise_samples_np.shape[1]}")
        reshaped_noise = np.reshape(noise_samples_np[:, i], (img_sz, img_sz))
        reshape_noise_torch = torch.tensor(reshaped_noise, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        for j in range(basis_np.shape[2]):
            kernel_torch = torch.tensor(flipped_basis_np[:, :, j], dtype=torch.float32, device=device).unsqueeze(
                0).unsqueeze(0)  # Add channel dims
            kernel_torch_flipped = torch.flip(kernel_torch, dims=[2, 3])
            # Perform 2D convolution using torch (valid mode, no padding)
            convolved_torch = F.conv2d(reshape_noise_torch, kernel_torch_flipped, padding=0).squeeze()
            S_z_torch[:, :, i] += torch.pow(convolved_torch, 2)
    return S_z_torch.detach().cpu().numpy()

def peak_algorithm_cont_mask_tf(img, basis, sideLengthAlgorithm, contamination_mask=None, obj_sz_down_scaled=None,
                             contamination_threshold=0.5, debug=False):
    """
    Identify peaks in an image using convolution with basis functions, avoiding contaminated areas if a mask is provided.

    Parameters:
    img                : Input image (2D NumPy array).
    basis              : Array of basis functions (3D NumPy array).
    sideLengthAlgorithm: Length parameter for peak extraction.
    contamination_mask : (Optional) Downsampled binary contamination mask (2D NumPy array).
    obj_sz_down_scaled : (Optional) Size of object patch to consider when checking contamination.
    contamination_threshold : (float) Threshold for contamination mask.
    debug              : (bool) Enable debugging visualization.

    Returns:
    peaks     : List of peak values.
    peaks_loc : List of peak locations (coordinates).
    S         : Scoring map from convolution.
    """
    img = img.astype(np.float32)

    basis = basis.astype(np.float32)
    num_of_basis_functions = basis.shape[2]
    rDelAlgorithm = round(sideLengthAlgorithm // 2)

    peaks = []
    peaks_loc = []
    obj_sz = basis.shape[0]

    # Perform convolution and sum over basis functions
    img_shape = tf.reshape(img, [1, img.shape[0], img.shape[1], 1])
    img_shape = tf.cast(img_shape, tf.float32)
    S = tf.zeros_like(img_shape, dtype=tf.float32)
    flipped_basis_list = [np.flip(np.flip(basis[:, :, j], 0), 1) for j in range(num_of_basis_functions)]
    for flipped_basis in flipped_basis_list:
        flipped_basis_shape = tf.reshape(flipped_basis, [flipped_basis.shape[0], flipped_basis.shape[1], 1, 1])
        flipped_basis_shape = tf.cast(flipped_basis_shape, tf.float32)
        S += tf.nn.conv2d(img_shape, flipped_basis_shape, strides=[1, 1, 1, 1], padding='SAME') ** 2

    S = tf.squeeze(S).numpy()
    scoringMat = S.copy()
    scoringMat[:obj_sz // 2, :] = 0
    scoringMat[:, :obj_sz // 2] = 0
    scoringMat[-obj_sz // 2:, :] = 0
    scoringMat[:, -obj_sz // 2:] = 0
    rows, cols = np.ogrid[:scoringMat.shape[0], :scoringMat.shape[1]]
    while True:
        pMax = np.max(scoringMat)
        if pMax <= 0:
            break
        I = np.argmax(scoringMat)
        i_row, i_col = np.unravel_index(I, scoringMat.shape)

        # Check for contamination if mask is provided
        if contamination_mask is not None and obj_sz_down_scaled is not None:
            half_patch = obj_sz_down_scaled // 2
            row_start = max(0, i_row - half_patch)
            row_end = min(contamination_mask.shape[0], i_row + half_patch)
            col_start = max(0, i_col - half_patch)
            col_end = min(contamination_mask.shape[1], i_col + half_patch)

            # Skip contaminated regions
            if np.any(contamination_mask[row_start:row_end, col_start:col_end] > contamination_threshold):
                row_start = max(0, i_row - rDelAlgorithm)
                row_end = min(scoringMat.shape[0], i_row + rDelAlgorithm + 1)
                col_start = max(0, i_col - rDelAlgorithm)
                col_end = min(scoringMat.shape[1], i_col + rDelAlgorithm + 1)
                scoringMat[row_start:row_end, col_start:col_end] = 0
                continue

        # If not contaminated, save peak information

        mask = (rows - i_row) ** 2 + (cols - i_col) ** 2 <= rDelAlgorithm ** 2
        scoringMat[mask] = 0
        peaks.append(pMax)
        peaks_loc.append([i_row, i_col])


    return np.array(peaks), np.array(peaks_loc), S

def peak_algorithm_cont_mask_scipy(img, basis, sideLengthAlgorithm, contamination_mask=None, obj_sz_down_scaled=None,
                             contamination_threshold=0.5, debug=False):
    """
    Identify peaks in an image using convolution with basis functions, avoiding contaminated areas if a mask is provided.

    Parameters:
    img                : Input image (2D NumPy array).
    basis              : Array of basis functions (3D NumPy array).
    sideLengthAlgorithm: Length parameter for peak extraction.
    contamination_mask : (Optional) Downsampled binary contamination mask (2D NumPy array).
    obj_sz_down_scaled : (Optional) Size of object patch to consider when checking contamination.
    contamination_threshold : (float) Threshold for contamination mask.
    debug              : (bool) Enable debugging visualization.

    Returns:
    peaks     : List of peak values.
    peaks_loc : List of peak locations (coordinates).
    S         : Scoring map from convolution.
    """
    img = img.astype(np.float32)

    basis = basis.astype(np.float32)
    num_of_basis_functions = basis.shape[2]
    rDelAlgorithm = round(sideLengthAlgorithm // 2)

    peaks = []
    peaks_loc = []
    obj_sz = basis.shape[0]

    # Perform convolution and sum over basis functions
    S = np.zeros_like(img, dtype=np.float32)
    flipped_basis = np.zeros(basis.shape)
    for n in range(basis.shape[2]):
        flipped_basis[:,:,n] = np.flip(np.flip(basis[:,:,n], 0), 1)
    for n in range(basis.shape[2]):
        # S += signal.fftconvolve(img, flipped_basis[:, :, n], mode='same') ** 2
        S += signal.convolve2d(img, flipped_basis[:, :, n], mode='same') ** 2


    scoringMat = S.copy()
    scoringMat[:obj_sz // 2, :] = 0
    scoringMat[:, :obj_sz // 2] = 0
    scoringMat[-obj_sz // 2:, :] = 0
    scoringMat[:, -obj_sz // 2:] = 0
    rows, cols = np.ogrid[:scoringMat.shape[0], :scoringMat.shape[1]]
    while True:
        pMax = np.max(scoringMat)
        if pMax <= 0:
            break
        I = np.argmax(scoringMat)
        i_row, i_col = np.unravel_index(I, scoringMat.shape)

        # Check for contamination if mask is provided
        if contamination_mask is not None and obj_sz_down_scaled is not None:
            half_patch = obj_sz_down_scaled // 2
            row_start = max(0, i_row - half_patch)
            row_end = min(contamination_mask.shape[0], i_row + half_patch)
            col_start = max(0, i_col - half_patch)
            col_end = min(contamination_mask.shape[1], i_col + half_patch)

            # Skip contaminated regions
            if np.any(contamination_mask[row_start:row_end, col_start:col_end] > contamination_threshold):
                row_start = max(0, i_row - rDelAlgorithm)
                row_end = min(scoringMat.shape[0], i_row + rDelAlgorithm + 1)
                col_start = max(0, i_col - rDelAlgorithm)
                col_end = min(scoringMat.shape[1], i_col + rDelAlgorithm + 1)
                scoringMat[row_start:row_end, col_start:col_end] = 0
                continue

        # If not contaminated, save peak information

        mask = (rows - i_row) ** 2 + (cols - i_col) ** 2 <= rDelAlgorithm ** 2
        scoringMat[mask] = 0
        peaks.append(pMax)
        peaks_loc.append([i_row, i_col])


    return np.array(peaks), np.array(peaks_loc), S


def peak_algorithm_cont_mask_torch(img, basis, sideLengthAlgorithm, contamination_mask=None, obj_sz_down_scaled=None,
                                   contamination_threshold=0.5, debug=False, device='cuda'):
    """
    Identify peaks in an image using convolution with basis functions, avoiding contaminated areas if a mask is provided.

    Parameters:
    img                : Input image (2D NumPy array).
    basis              : Array of basis functions (3D NumPy array).
    sideLengthAlgorithm: Length parameter for peak extraction.
    contamination_mask : (Optional) Downsampled binary contamination mask (2D NumPy array).
    obj_sz_down_scaled : (Optional) Size of object patch to consider when checking contamination.
    contamination_threshold : (float) Threshold for contamination mask.
    debug              : (bool) Enable debugging visualization.

    Returns:
    peaks     : List of peak values.
    peaks_loc : List of peak locations (coordinates).
    S         : Scoring map from convolution.
    """
    img = img.astype(np.float32)

    basis = basis.astype(np.float32)
    num_of_basis_functions = basis.shape[2]
    rDelAlgorithm = round(sideLengthAlgorithm // 2)

    peaks = []
    peaks_loc = []
    obj_sz = basis.shape[0]
    # Perform convolution and sum over basis functions
    # S_np = np.zeros_like(img, dtype=np.float32)
    flipped_basis = np.zeros(basis.shape)
    for n in range(basis.shape[2]):
        flipped_basis[:, :, n] = np.flip(np.flip(basis[:, :, n], 0), 1)

    # for n in range(basis.shape[2]):
    #     # S += signal.fftconvolve(img, flipped_basis[:, :, n], mode='same') ** 2
    #     S_np += signal.convolve2d(img, flipped_basis[:, :, n], mode='same') ** 2

    # Perform convolution and sum over basis functions

    S_torch = torch.zeros(img.shape, dtype=torch.float32, device=device)
    img_torch = torch.tensor(img, device=device)

    for n in range(basis.shape[2]):
        kernel_torch = torch.tensor(flipped_basis[:, :, n], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(
            0)
        # S += signal.fftconvolve(img, flipped_basis[:, :, n], mode='same') ** 2
        convolved_torch = F.conv2d(img_torch.unsqueeze(0).unsqueeze(0), kernel_torch, padding='same').squeeze()
        S_torch += torch.pow(convolved_torch, 2)

    S = S_torch.detach().cpu().numpy()
    # relative_error = np.linalg.norm(S_np - S) / np.linalg.norm(S_np)
    # print(relative_error)
    scoringMat = S.copy()
    scoringMat[:obj_sz // 2, :] = 0
    scoringMat[:, :obj_sz // 2] = 0
    scoringMat[-obj_sz // 2:, :] = 0
    scoringMat[:, -obj_sz // 2:] = 0
    rows, cols = np.ogrid[:scoringMat.shape[0], :scoringMat.shape[1]]
    while True:
        pMax = np.max(scoringMat)
        if pMax <= 0:
            break
        I = np.argmax(scoringMat)
        i_row, i_col = np.unravel_index(I, scoringMat.shape)

        # Check for contamination if mask is provided
        if contamination_mask is not None and obj_sz_down_scaled is not None:
            half_patch = obj_sz_down_scaled // 2
            row_start = max(0, i_row - half_patch)
            row_end = min(contamination_mask.shape[0], i_row + half_patch)
            col_start = max(0, i_col - half_patch)
            col_end = min(contamination_mask.shape[1], i_col + half_patch)

            # Skip contaminated regions
            if np.any(contamination_mask[row_start:row_end, col_start:col_end] > contamination_threshold):
                row_start = max(0, i_row - rDelAlgorithm)
                row_end = min(scoringMat.shape[0], i_row + rDelAlgorithm + 1)
                col_start = max(0, i_col - rDelAlgorithm)
                col_end = min(scoringMat.shape[1], i_col + rDelAlgorithm + 1)
                scoringMat[row_start:row_end, col_start:col_end] = 0
                continue

        # If not contaminated, save peak information

        mask = (rows - i_row) ** 2 + (cols - i_col) ** 2 <= rDelAlgorithm ** 2
        scoringMat[mask] = 0
        peaks.append(pMax)
        peaks_loc.append([i_row, i_col])

    return np.array(peaks), np.array(peaks_loc), S

def test_function_real_data(z_max, Y_peaks):
    """
    Estimate the test function using the law of large numbers with an indicator function.

    Parameters:
    z_max   : Array of maximum projection values.
    Y_peaks : Array of peak values.

    Returns:
    test_val : Array of test values for each peak.
    """

    test_val = np.zeros(Y_peaks.shape[0])

    for i in range(Y_peaks.shape[0]):
        test_val[i] = np.mean(z_max > Y_peaks[i])

    return test_val

def BH(p_val, alpha, M_L):
    """
    Perform the Benjamini-Hochberg procedure to control the false discovery rate.

    Parameters:
    p_val : List or array of p-values sorted in ascending order.
    alpha : Desired false discovery rate.
    M_L   : Number of hypotheses.

    Returns:
    K : Index of the largest p-value that passes the BH criterion.
    """

    K = -1
    for l in range(len(p_val)):
        # Check the BH condition
        if p_val[l] > (l * alpha) / M_L :
            K = l
            break

    return K

def coords_output(Y_peaks_loc, addr_coords, microName, mgScale, mgBigSz, K, patchSzPickBox):
    """
    Writes particle coordinates to .star and .box files for given peaks.

    Parameters:
    Y_peaks_loc : Array of peak locations.
    addr_coords : Directory path for output files.
    microName   : Name of the micrograph.
    mgScale     : Scaling factor for micrograph.
    mgBigSz     : Size of the micrograph (tuple or list).
    K           : Number of peaks to output.
    """
    # Prepare file paths
    star_file_path = os.path.join(addr_coords, f"{microName}.star")
    box_file_path = os.path.join(addr_coords, f"{microName}.box")

    # Open files for writing
    with open(star_file_path, 'w') as particlesCordinateStar, open(box_file_path, 'w') as particlesCordinateBox:
        # Write headers for the .star file
        particlesCordinateStar.write("data_\n\nloop_\n")
        particlesCordinateStar.write("_rlnCoordinateX #1\n")
        particlesCordinateStar.write("_rlnCoordinateY #2\n\n")

        for i in range(K):
            i_colPatch = Y_peaks_loc[i, 1]
            i_rowPatch = Y_peaks_loc[i, 0]

            # Calculate coordinates and write to .star file
            x_star = (1 / mgScale) * i_colPatch
            y_star = (mgBigSz[0] + 1) - (1 / mgScale) * i_rowPatch
            particlesCordinateStar.write(f"{x_star:.0f}\t{y_star:.0f}\n")

            # Calculate coordinates and write to .box file
            x_box = (1 / mgScale) * i_colPatch - patchSzPickBox // 2
            #y_box = (1 / mgScale) * i_rowPatch - patchSzPickBox // 2
            y_box = (mgBigSz[0] + 1) - (1 / mgScale) * i_rowPatch - patchSzPickBox // 2
            particlesCordinateBox.write(f"{x_box:.0f}\t{y_box:.0f}\t{patchSzPickBox:.0f}\t{patchSzPickBox:.0f}\n")

def plot_and_save(image, circles, obj_sz_down_scaled, filename, output_folder):
    """
    Plot an image with optional circles and save it to a file, keeping the image size the same.

    Parameters:
    image : 2D array
        The image to be plotted.
    circles : list of tuples
        Each tuple contains the (x, y) coordinates of a circle to be drawn.
    obj_sz_down_scaled : int
        The size of the object after downscaling, used to determine circle radius.
    filename : str
        The name of the file to save the plot as.
    output_folder : str
        The directory where the file will be saved.
    """
    # Set the figure size to match the image size exactly
    fig, ax = plt.subplots(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)

    # Display the image without axis
    ax.imshow(image, cmap='gray', aspect='equal')
    ax.axis('off')

    # Draw the circles if provided
    if circles is not None and len(circles) > 0:
        for (x, y) in circles:
            circle = Circle((x, y), radius=np.ceil(obj_sz_down_scaled / 2), color='red', fill=False, linewidth=0.5)
            ax.add_patch(circle)

    # Save the plot without cropping the image
    file_path = os.path.join(output_folder, filename)
    plt.savefig(file_path, pad_inches=0, dpi=100)
    plt.close(fig)
    #plt.show()

def csv_to_box(input_csv, output_box, box_size, mgScale, image_height):
    """
    Convert CSV to BOX format for EMAN2, ensuring coordinates align with the micrograph.

    Parameters:
    input_csv : str : Path to input CSV file
    output_box : str : Path to output BOX file
    box_size : int : Box size for particles
    mgScale : float : Scaling factor for micrograph
    image_height : int : Height of the flipped micrograph
    """
    # Load the CSV file (assuming columns: 'X-Coordinate', 'Y-Coordinate')
    df = pd.read_csv(input_csv)

    # Open the BOX file for writing
    with open(output_box, 'w') as f:
        for _, row in df.iterrows():
            # Scale and adjust for EMAN2 format
            x = row['X-Coordinate'] * mgScale
            y = row['Y-Coordinate'] * mgScale

            # Correct for flip by subtracting from image height
            y_corrected = (image_height - y) - box_size

            # Calculate top-left corner of the box
            x_topleft = x - box_size // 2
            y_topleft = y_corrected - box_size // 2

            # Write to .box file
            f.write(f"{x_topleft:.0f}\t{y_topleft:.0f}\t{box_size}\t{box_size}\n")

def process_all_csvs_to_box(input_dir, output_dir, box_size, mgScale, image_height):
    """
    Process all CSV files in a directory and convert them to BOX files for EMAN2.

    Parameters:
    input_dir : str : Directory containing input CSV files
    output_dir : str : Directory to save the output BOX files
    box_size : int : Box size for particles
    mgScale : float : Scaling factor for micrograph
    image_height : int : Height of the flipped micrograph
    """
    os.makedirs(output_dir, exist_ok=True)
    for csv_file in os.listdir(input_dir):
        if csv_file.endswith('.csv'):
            microName = os.path.splitext(csv_file)[0]
            input_csv_path = os.path.join(input_dir, csv_file)
            output_box_path = os.path.join(output_dir, f"{microName}.box")
            csv_to_box(input_csv_path, output_box_path, box_size, mgScale, image_height)


def plot_image_stack(image_stack):
    """
    Plots a stack of 2D images in a grid layout.

    Parameters:
    - image_stack: A 3D numpy array of images, where the shape is (n, height, width).
                   n is the number of images, and each image is a 2D array.
    """
    n = image_stack.shape[0]  # Number of images

    # Determine grid size (rows, cols) based on the number of images
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    # Create figure and axes for the subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    # Flatten the axes array to easily iterate over it
    axes = axes.flatten()

    # Loop through each image and plot it
    for i in range(n):
        axes[i].imshow(image_stack[i], cmap='gray')
        axes[i].axis('off')  # Turn off axis labels
        axes[i].set_title(f'Image {i + 1}')

    # Remove any unused subplots if n is not a perfect square
    for i in range(n, len(axes)):
        fig.delaxes(axes[i])

    # Adjust the layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()

