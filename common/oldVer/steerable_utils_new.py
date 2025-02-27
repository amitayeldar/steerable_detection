#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 12:50:30 2024

@author: kerenmor
"""

## This is the main code used as a proof of cnמcept
import os

from scipy.spatial.distance import squareform, pdist

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import convolve
import os.path
import os.path
from aspire.image import *
from aspire.image import Image as AspireImage
# from scipy.signal import correlate2d
from aspire.basis import FBBasis2D
import os
# from scipy.ndimage import rotate
# from scipy.interpolate import interp1d
from numpy.fft import fft2, ifft2
# from scipy.linalg import eigh
# from scipy.signal import convolve
# from scipy.signal import convolve2d
# import scipy.linalg
# from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from numba import jit, prange
# import tensorflow as tf
import pickle
from matplotlib.patches import Circle
# from scipy.spatial.distance import pdist, squareform
# from scipy.signal import convolve2d

# from scipy.signal import wiener
# from scipy.linalg import toeplitz
# import cupy as cp
# from cupyx.scipy.signal import correlate2d  # GPU-accelerated correlation
# from cupyx.scipy.linalg import toeplitz  # GPU-accelerated Toeplitz
import numpy as np
from scipy.signal import convolve2d
import torch
import torch.nn.functional as F
import gc
# Set device to MPS if available
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
import os
import mrcfile
import re
import numpy as np
from sklearn.decomposition import TruncatedSVD

# import cupy as cp
import numpy as np
# from cupyx.scipy.ndimage import convolve
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def fourier_bessel_pca_per_angle(noise_patches,fb_basis):
    num_img = noise_patches.shape[0]
    eigen_vectors_per_ang_lst = []
    mean_noise_per_ang_lst = []
    # get coeff_per_angular
    coeff_per_img = []
    coeff_per_angle = []
    for n_img in range(num_img):
        img = Image(noise_patches[n_img].astype(np.float32))
        coeff_per_img.append(fb_basis.evaluate_t(img))

    for n_ang in range(fb_basis.ell_max+1):
        if n_ang == 0:
            s = 0
            e = fb_basis.k_max[0]
            coeff_arr = np.zeros((fb_basis.k_max[n_ang], num_img))
            for n_img in range(num_img):
                coeff_tmp = coeff_per_img[n_img]._data
                coeff_arr[:,n_img] = coeff_tmp[0][s:e]
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
            eigen_vectors_per_ang_lst.append(eigen_vectors_per_angle[:,:k])
        else:
            for i in range(2):
                coeff_arr = np.zeros((fb_basis.k_max[n_ang], num_img))
                s = e
                e += fb_basis.k_max[n_ang]
                for n_img in range(num_img):
                    coeff_tmp = coeff_per_img[n_img]._data
                    coeff_arr[:, n_img] = coeff_tmp[0][s:e]
                mean_noise_per_ang_lst.append(np.mean(coeff_arr,axis=1))
                cov_per_angle = np.cov(coeff_arr)
                eigen_values_per_angle, eigen_vectors_per_angle = np.linalg.eigh(cov_per_angle)
                sorted_indices = np.argsort(eigen_values_per_angle)[::-1]
                eigen_values_per_angle = eigen_values_per_angle[sorted_indices]
                eigen_vectors_per_angle = eigen_vectors_per_angle[:, sorted_indices]
                total_norm = np.sum(eigen_values_per_angle)
                cumulative_norm = np.cumsum(eigen_values_per_angle)
                l2_ratios = cumulative_norm / total_norm
                k = np.argmax(l2_ratios >= 0.9) + 1
                eigen_vectors_per_ang_lst.append(eigen_vectors_per_angle[:,:k])
    return eigen_vectors_per_ang_lst,mean_noise_per_ang_lst

def sort_steerable_basis_fixed(basis_full, objects, noise_patches, max_dimension):
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

    # Calculate coefficients (square of projections)
    coeff_objects = np.square((basis_full.T) @ objects)
    coeff_noise = np.square((basis_full.T) @ noise_patches)

    # Full projected SNR
    projected_norm_objects = coeff_objects.sum(axis=0)
    projected_norm_noise = coeff_noise.sum(axis=0)
    plt.plot(np.sort(projected_norm_objects)[::-1])
    plt.plot(np.sort(projected_norm_noise)[::-1])
    plt.show()
    projected_snr = projected_norm_objects / projected_norm_noise.mean()
    min_idx_projected_snr = np.argmin(projected_snr)
    # projected_norm_objects = np.zeros((117))
    # for n in range(117):
    #     projected_norm_objects[n] = np.linalg.norm((basis_full.T) @ objects[:, n])**2
    # plt.plot(np.sort(projected_norm_objects)[::-1])
    # plt.show()
    # Initialize lists and variables
    projected_snr_per_dim = []
    sorted_basis_lst = []
    sorted_basis_idx = []
    projected_norm_objects = np.zeros(projected_norm_objects.shape)
    projected_norm_noise = np.zeros(projected_norm_noise.shape)
    basis_idx = []

    # Initialize a mask to track used indices
    active_mask = np.ones(basis_full.shape[1], dtype=bool)

    # Loop to select max_dimension basis vectors
    for dim in range(max_dimension):
        
        numerator = coeff_objects[:, min_idx_projected_snr] + projected_norm_objects[min_idx_projected_snr]
        denominator = np.zeros(coeff_noise.shape[0])

        # Update the denominator for all noise patches
        for n in range(coeff_noise.shape[1]):
            denominator += coeff_noise[:, n] + projected_norm_noise[n]
        denominator = denominator / coeff_noise.shape[1]

        ratio = numerator / denominator
        # ratio = numerator
        # Only consider active indices for max selection
        ratio[~active_mask] = -np.inf
        max_idx_candidate = np.argmax(ratio)
        basis_idx.append(max_idx_candidate)
        # Update norms
        projected_norm_objects += coeff_objects[max_idx_candidate, :]
        projected_norm_noise += coeff_noise[max_idx_candidate, :]

        # Append the sorted basis vector
        sorted_basis_lst.append(basis_full[:, max_idx_candidate])
        sorted_basis_idx.append(max_idx_candidate)

        # Mark this index as used
        active_mask[max_idx_candidate] = False

        # Sanity check
        # sorted_basis_vectors = np.column_stack(sorted_basis_lst)
        # projected_norm_objects_check = np.square((sorted_basis_vectors.T) @ objects).sum(axis=0)
        # projected_norm_noise_check = np.square((sorted_basis_vectors.T) @ noise_patches).sum(axis=0)
        # # projected_norm_objects_check = np.square((basis_full[:, ~active_mask].T) @ objects).sum(axis=0)
        # # projected_norm_noise_check = np.square((basis_full[:, ~active_mask].T) @ noise_patches).sum(axis=0)
        # diff_num = (projected_norm_objects_check - projected_norm_objects) / projected_norm_objects
        # diff_den = (projected_norm_noise_check - projected_norm_noise) / projected_norm_noise
        # print(f"Sanity Check objects (Iteration {dim + 1}):\n", diff_num)
        # print(f"Sanity Check noise (Iteration {dim + 1}):\n", diff_den)
        
        #Recompute projected SNR
        projected_snr = projected_norm_objects / projected_norm_noise.mean()
        min_idx_projected_snr = np.argmin(projected_snr)
        # max_idx_projected_snr = np.argmax(projected_snr)
        projected_snr_per_dim.append(projected_snr[min_idx_projected_snr])
        

    # Stack sorted basis vectors
    sorted_basis_vectors = np.column_stack(sorted_basis_lst)

    return sorted_basis_vectors, projected_snr_per_dim,projected_norm_objects,projected_norm_noise, basis_idx

def plot_patches_with_coordinates(Y, noise_patches_coor, patch_size):
    """
    Plots a micrograph with coordinates marked and boxes drawn around them to represent patches.
    
    Parameters:
    Y (ndarray): The micrograph image to be plotted.
    noise_patches_coor (ndarray or list): Array or list of tuples (x, y) coordinates for the center of each patch.
    patch_size (int): The size of the patch (side length of the square box).
    """
    # Convert noise_patches_coor to a NumPy array if it’s a list of tuples
    noise_patches_coor = np.array(noise_patches_coor)
    
    # Plot the micrograph image
    plt.figure(figsize=(8, 8))
    plt.imshow(Y, cmap='gray')  # You can use other colormaps if needed
    
    # Plot the coordinates on top of the image
    plt.scatter(noise_patches_coor[:, 0], noise_patches_coor[:, 1], c='red', s=10, label='Coordinates')
    
    # Add rectangles around each coordinate (as the center of the patch)
    for (x, y) in noise_patches_coor:
        # Create a rectangle with bottom-left corner adjusted to center around (x, y)
        rect = Rectangle((x - patch_size // 2, y - patch_size // 2), patch_size, patch_size,
                         linewidth=1, edgecolor='blue', facecolor='none', linestyle='--')
        plt.gca().add_patch(rect)
    
    # Optional: Add labels and title
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Micrograph with Noise Patch Coordinates and Boxes')
    plt.legend()
    
    # Show the plot
    plt.show()
    
def projected_noise_simulation_from_noise_patches_para_cup(noise_samples, basis, num_of_exp_noise):
    """
    Simulates projected noise using given noise samples and basis functions in batches,
    distributing computation across multiple GPUs using CuPy.
    """

    # 🔹 **Keep `noise_samples` on CPU, move only batches**
    x = 1
    noise_samples_cpu = noise_samples.astype(np.float32)

    # List available GPUs
    num_gpus = cp.cuda.runtime.getDeviceCount()
    if num_gpus == 0:
        raise RuntimeError("No GPUs found! Consider using a CPU-based approach.")

    print(f"Using {num_gpus} GPUs for parallel processing.")

    # 🔹 **Move `basis` to GPU**
    basis_gpu = cp.asarray(basis, dtype=cp.float32)

    # 🔹 **Compute output size**
    sz = int(np.sqrt(noise_samples.shape[0]))  # Image size
    test_conv = convolve(cp.reshape(cp.asarray(noise_samples_cpu[:, 0]), (sz, sz)), 
                         cp.reshape(basis_gpu[:, :, 0], (basis.shape[0], basis.shape[1])), mode='constant')
    sz_pn = test_conv.size  # **Flattened size of convolved output (should be 16641)**

    print(f"Output shape: ({num_of_exp_noise}, {sz_pn})")

    # 🔹 **Use disk storage to avoid memory crash**
    S_z_n = np.memmap('S_z_n.dat', dtype=np.float32, mode='w+', shape=(num_of_exp_noise, sz_pn))

    # 🔹 **Reduce memory use by processing small batches**
    max_batch_size_per_gpu = max(1, num_of_exp_noise // (10 * num_gpus))
    print(f"Using batch size: {max_batch_size_per_gpu}")

    # 🔹 **Preallocate memory for flipped basis (on GPU)**
    flipped_basis_stack = cp.stack([cp.flip(cp.flip(basis_gpu[:, :, j], 0), 1) for j in range(basis.shape[2])])

    # 🔹 **Function to process a batch on a given GPU**
    def process_batch(start, end, device_id):
        """Processes a batch on a given GPU, ensuring basis is moved to the correct GPU."""
        with cp.cuda.Device(device_id):
            # 🔹 **Ensure `flipped_basis_stack` is on the correct GPU**
            flipped_basis_stack_gpu = flipped_basis_stack.copy()

            # Move batch to GPU
            noise_batch_gpu = cp.asarray(noise_samples_cpu[:, start:end])  # Move only batch to GPU
            noise_imgs_gpu = cp.reshape(noise_batch_gpu, (end - start, sz, sz))

            # Perform convolution for each image
            conv_results = cp.array([
                convolve(noise_imgs_gpu[i], flipped_basis_stack_gpu[j], mode='constant')**2
                for i in range(end - start) for j in range(basis.shape[2])
            ])

            # Sum over all basis functions and flatten
            conv_results = cp.sum(conv_results.reshape(end - start, basis.shape[2], sz_pn), axis=1)

            # 🔹 **Move results back to CPU and free GPU memory**
            conv_results_cpu = cp.asnumpy(conv_results)
            del noise_batch_gpu, noise_imgs_gpu, conv_results, flipped_basis_stack_gpu
            cp.get_default_memory_pool().free_all_blocks()

            return conv_results_cpu

    # 🔹 **Process in smaller batches**
    for batch_idx in range((num_of_exp_noise + max_batch_size_per_gpu - 1) // max_batch_size_per_gpu):
        start = batch_idx * max_batch_size_per_gpu
        end = min((batch_idx + 1) * max_batch_size_per_gpu, num_of_exp_noise)

        # Process only one batch at a time to avoid memory overload
        batch_result = process_batch(start, end, batch_idx % num_gpus)

        # Store result in file-backed array instead of RAM
        S_z_n[start:end] = batch_result

    # 🔹 **Flush memory-mapped file and close it**
    S_z_n.flush()

    return S_z_n


def compute_top_eigen_svd(patches, num_components=15):
    """
    Compute the top eigenvectors and eigenvalues using Truncated SVD,
    ensuring the data is centered before applying SVD.

    Parameters:
    - patches: (15, 129, 129) array where each patch is 129x129
    - num_components: Number of top eigenvectors to extract (default: 15)

    Returns:
    - eigenvalues: (15,) array of top 15 eigenvalues
    - eigenvectors: (129*129, 15) array of top 15 eigenvectors
    """
    # Reshape patches from (15, 129, 129) to (15, 16641) for SVD
    patches_reshaped = patches.reshape(patches.shape[0], -1)  # (15, 16641)
    
    # Center the data: Subtract the mean image across patches
    mean_patch = np.mean(patches_reshaped, axis=0, keepdims=True)  # (1, 16641)
    patches_centered = patches_reshaped - mean_patch

    # Apply Truncated SVD to extract the top principal components
    svd = TruncatedSVD(n_components=num_components)
    transformed_patches = svd.fit_transform(patches_centered)  # Ensures singular values are computed

    # The top eigenvalues are approximated as the square of the singular values
    top_eigenvalues = svd.singular_values_**2 / (patches.shape[0] - 1)
    
    # The top eigenvectors are the right singular vectors
    top_eigenvectors = svd.components_.T  # (16641, 15)

    return top_eigenvalues, top_eigenvectors

def generate_gaussian_samples(eigenvalues, eigenvectors, num_samples=50):
    """
    Generate multiple samples from a Gaussian process estimated from noise patches.

    Parameters:
    - eigenvalues: (15,) array of top 15 eigenvalues
    - eigenvectors: (N, 15) array of top 15 eigenvectors
    - num_samples: Number of noise samples to generate

    Returns:
    - noise_samples: (N, num_samples) array of generated Gaussian samples
    """
    # Compute the square root of eigenvalues (15x15 diagonal matrix)
    Lambda_sqrt = np.diag(np.sqrt(eigenvalues))
    
    # Generate standard normal random vectors (15 x num_samples)
    Z = np.random.randn(len(eigenvalues), num_samples)
    
    # Compute correlated noise in reduced dimension
    Y = Lambda_sqrt @ Z
    
    # Transform back to the original space
    noise_samples = eigenvectors @ Y

    return noise_samples

def extract_patches(micrograph, coord_path, box_size=None, file_type="star"):
    """
    Extract patches from the micrograph image based on coordinates in .star or .box files.
    
    Parameters:
    micrograph : ndarray : The micrograph image as a NumPy array.
    coord_path : str : Path to the .star or .box file.
    box_size : int : Size of the square patch to extract (required for .star files, ignored for .box files).
    file_type : str : Type of file, either 'star' or 'box'.
    
    Returns:
    patches : list : A list of extracted patches as NumPy arrays.
    """
    if file_type not in ["star", "box"]:
        raise ValueError("Invalid file_type. Must be 'star' or 'box'.")

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

    def load_box(file_path):
        boxes = np.loadtxt(file_path, delimiter='\t', usecols=(0, 1, 2, 3))
        return boxes  # Array of [x, y, w, h]

    # Load coordinates
    if file_type == "star":
        if box_size is None:
            raise ValueError("box_size must be specified for .star files.")
        coordinates = load_star(coord_path)
    else:  # file_type == "box"
        coordinates = load_box(coord_path)

    # Extract patches
    patches = []
    for i, coord in enumerate(coordinates):
        if file_type == "star":
            # For .star files, coord is (x, y)
            x, y = int(round(coord[0])), int(round(coord[1]))
            half_box = box_size // 2
            x_start = max(0, x - half_box)
            x_end = min(micrograph.shape[1], x + half_box)
            y_start = max(0, y - half_box)
            y_end = min(micrograph.shape[0], y + half_box)
            patch = micrograph[y_start:y_end, x_start:x_end]
        else:  # file_type == "box"
            # For .box files, coord is (x, y, w, h)
            x, y, w, h = map(int, coord)
            x_start = max(0, x)
            x_end = min(micrograph.shape[1], x + w)
            y_start = max(0, y)
            y_end = min(micrograph.shape[0], y + h)
            patch = micrograph[y_start:y_end, x_start:x_end]

        # If the patch is not the exact desired size, pad with zeros (only for star patches)
        if file_type == "star" and patch.shape != (box_size, box_size):
            patch_padded = np.zeros((box_size, box_size), dtype=patch.dtype)
            patch_padded[:patch.shape[0], :patch.shape[1]] = patch
            patch = patch_padded

        patches.append(patch)

    print(f"Extracted {len(patches)} patches from {coord_path}.")
    return patches


def read_coordinates(coord_file):
    """
    Read coordinates from .star, .csv, or .box files.

    Parameters:
        coord_file : str : Path to the coordinate file.

    Returns:
        coordinates : ndarray : Array of (x, y) coordinates.
        file_type   : str     : 'star', 'csv', or 'box' based on the file type.
    """
    coordinates = []

    # Detect file type
    if coord_file.endswith('.star'):
        file_type = 'star'
    elif coord_file.endswith('.csv'):
        file_type = 'csv'
    elif coord_file.endswith('.box'):
        file_type = 'box'
    else:
        raise ValueError("Unsupported file type. Only .star, .csv, and .box files are supported.")

    print(f"Reading file: {coord_file} (type: {file_type})")

    with open(coord_file, 'r') as f:
        inside_loop = False  # Flag for .star files

        for line_number, line in enumerate(f, 1):
            print(f"Raw line {line_number}: {line!r}")  # Debug: Print raw line
            line = line.strip()

            # Parsing for .star files
            if file_type == 'star':
                if not line or line.startswith(('data_', 'loop_', '_')):
                    if line.startswith('loop_'):
                        inside_loop = True
                        print(f"'loop_' detected on line {line_number}. Starting to process data rows.")
                    else:
                        print(f"Skipping header/metadata line {line_number}: {line}")
                    continue

                if inside_loop:
                    tokens = re.split(r'\s+', line)  # Split on any whitespace
                    print(f"Tokens from line {line_number}: {tokens}")  # Debug: Show tokens
                    if len(tokens) >= 2:
                        try:
                            x = int(float(tokens[0]))  # Column #1
                            y = int(float(tokens[1]))  # Column #2
                            coordinates.append((x, y))
                            print(f"Valid coordinate: ({x}, {y})")
                        except ValueError as e:
                            print(f"Error parsing line {line_number}: {line.strip()}, Error: {e}")
                            continue

            # Parsing for .csv files
            elif file_type == 'csv':
                tokens = line.split(',')
                print(f"Tokens from line {line_number}: {tokens}")  # Debug: Show tokens
                if len(tokens) >= 2:
                    try:
                        x = int(float(tokens[0]))  # First column
                        y = int(float(tokens[1]))  # Second column
                        coordinates.append((x, y))
                        print(f"Valid coordinate: ({x}, {y})")
                    except ValueError as e:
                        print(f"Error parsing line {line_number}: {line.strip()}, Error: {e}")
                        continue

            # Parsing for .box files
            elif file_type == 'box':
                tokens = line.split()
                print(f"Tokens from line {line_number}: {tokens}")  # Debug: Show tokens
                if len(tokens) >= 4:
                    try:
                        # Box file: top-left (x, y) and dimensions (w, h)
                        x = int(tokens[0])
                        y = int(tokens[1])
                        w = int(tokens[2])
                        h = int(tokens[3])
                        # Convert top-left (x, y) to center coordinates
                        x_center = x + w // 2
                        y_center = y + h // 2
                        coordinates.append((x_center, y_center))
                        print(f"Valid coordinate (box): ({x_center}, {y_center})")
                    except ValueError as e:
                        print(f"Error parsing line {line_number}: {line.strip()}, Error: {e}")
                        continue

    # Check for empty coordinates
    if not coordinates:
        print(f"No valid coordinates found in {coord_file}.")
    else:
        print(f"Total coordinates parsed: {len(coordinates)}")

    return np.array(coordinates), file_type







def check_orthonormal_columns(matrix, tol=1e-6):
    """
    Check if the columns of a given matrix are orthonormal.

    :param matrix: 2D numpy array representing the matrix.
    :param tol: Tolerance for checking orthonormality.
    :return: True if columns are orthonormal, False otherwise.
    """
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")

    # Compute the dot product of the matrix with its transpose
    dot_product = np.dot(matrix.T, matrix)

    # Check if the dot product is close to the identity matrix
    identity_matrix = np.eye(matrix.shape[1])
    orthonormal = np.allclose(dot_product, identity_matrix, atol=tol)

    if not orthonormal:
        print("Dot product matrix (should be identity):\n", dot_product)

    return orthonormal


def save_coortodist_data(coortodistData, filename):
    with open(filename, 'wb') as file:
        pickle.dump(coortodistData, file)

# Function to load coortodistData
def load_coortodist_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
#def ensure_positive_definite(matrix):
 #   """Attempt Cholesky decomposition and adjust matrix to be positive definite if necessary."""
  #  try:
   #     _ = scipy.linalg.cholesky(matrix, lower=True)
    #    return matrix
    #except np.linalg.LinAlgError:
        # Adding a small value to the diagonal
     #   return make_positive_definite(matrix)
    
#def ensure_positive_definite(matrix, epsilon=1e-6, max_attempts=10):
#     """
#     Ensure the input matrix is positive definite by attempting to add a small value to the diagonal.

#     Parameters:
#     matrix : ndarray
#         Input matrix to check.
#     epsilon : float, optional
#         Initial value to add to the diagonal if the matrix is not positive definite.
#     max_attempts : int, optional
#         Maximum number of attempts to adjust the matrix.

#     Returns:
#     matrix : ndarray
#         Adjusted positive definite matrix.

#     Raises:
#     LinAlgError:
#         If the matrix cannot be made positive definite after max_attempts.
#     """
#     for attempt in range(max_attempts):
#         try:
#             _ = scipy.linalg.cholesky(matrix, lower=True)
#             return matrix
#         except np.linalg.LinAlgError:
#             diag_increment = epsilon * (10 ** attempt)  # Increase epsilon exponentially
#             matrix += np.eye(matrix.shape[0]) * diag_increment
#             print(f"Matrix adjusted with diagonal increment: {diag_increment:.2e}")
    
#     raise np.linalg.LinAlgError("Matrix could not be made positive definite after multiple attempts.")


def ensure_positive_definite_sqrt(cov_matrix):
    """
    Ensures the covariance matrix is positive semi-definite and returns its square root.
    
    If more than 10% of eigenvalues are negative, the function raises an error.

    Parameters:
        cov_matrix (cp.ndarray): Input covariance matrix (H*W, H*W).

    Returns:
        cov_matrix_sqrt (cp.ndarray): Square root of the adjusted covariance matrix.
    """
    # Free GPU memory
    cp.cuda.Device(6).use()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()

    cov_matrix = cp.asarray(cov_matrix, dtype=cp.float32)

    try:
        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = cp.linalg.eigh(cov_matrix)
    except AttributeError:
        print("⚠️ CuPy failed on GPU. Switching to CPU...")
        cov_matrix_cpu = cp.asnumpy(cov_matrix)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix_cpu)
        eigvals, eigvecs = cp.asarray(eigvals), cp.asarray(eigvecs)

    # Count negative eigenvalues
    num_negative = cp.sum(eigvals < 0)
    total_eigenvalues = eigvals.size

    # If more than 10% of eigenvalues are negative, raise an error
    if num_negative > 0.1 * total_eigenvalues:
        raise ValueError(f"❌ Too many negative eigenvalues! ({num_negative}/{total_eigenvalues})")

    # Find the smallest positive eigenvalue
    min_positive = cp.min(eigvals[eigvals > 0]) if cp.any(eigvals > 0) else 0

    # Replace negative eigenvalues with the smallest positive eigenvalue
    eigvals = cp.where(eigvals < 0, min_positive, eigvals)

    # Compute the square root of the eigenvalues
    eigvals_sqrt = cp.sqrt(eigvals)

    # Compute the square root of the covariance matrix
    cov_matrix_sqrt = eigvecs @ cp.diag(eigvals_sqrt) @ eigvecs.T

    return cov_matrix_sqrt


# def ensure_positive_definite_sqrt(cov_matrix):
#     """
#     Ensures the covariance matrix is positive semi-definite and returns its square root.

#     If more than 10% of eigenvalues are negative, the function raises an error.

#     Parameters:
#         cov_matrix (tf.Tensor or np.ndarray): Input covariance matrix (H*W, H*W).

#     Returns:
#         cov_matrix_sqrt (np.ndarray): Square root of the adjusted covariance matrix.
#     """
#     # Convert input to TensorFlow tensor if it's a NumPy array
#     cov_matrix = tf.convert_to_tensor(cov_matrix, dtype=tf.float32)

#     # Compute eigenvalues and eigenvectors
#     eigvals, eigvecs = tf.linalg.eigh(cov_matrix)

#     # Count negative eigenvalues
#     num_negative = tf.reduce_sum(tf.cast(eigvals < 0, tf.int32))
#     total_eigenvalues = tf.size(eigvals)

#     # If more than 10% of eigenvalues are negative, raise an error
#     if tf.greater(num_negative, tf.cast(0.1 * tf.cast(total_eigenvalues, tf.float32), tf.int32)):
#         raise ValueError(f"❌ Too many negative eigenvalues! ({num_negative.numpy()}/{total_eigenvalues.numpy()})")

#     # Find the smallest positive eigenvalue
#     min_positive = tf.reduce_min(tf.boolean_mask(eigvals, eigvals > 0))

#     # Replace negative eigenvalues with the smallest positive eigenvalue
#     eigvals = tf.where(eigvals < 0, min_positive, eigvals)

#     # Compute the square root of the eigenvalues
#     eigvals_sqrt = tf.sqrt(eigvals)

#     # Compute the square root of the covariance matrix
#     cov_matrix_sqrt = eigvecs @ tf.linalg.diag(eigvals_sqrt) @ tf.transpose(eigvecs)

#     # Convert the result back to NumPy array
#     return cov_matrix_sqrt.numpy()

def make_positive_definite(matrix, epsilon=1e-6):
    """Ensure the matrix is positive definite by adding a small value to the diagonal."""
    return matrix + epsilon * np.eye(matrix.shape[0])
def convolve_and_sum1(img, basis, num_of_basis_functions):
    img_sz = img.shape[0]
    S = np.zeros((img_sz, img_sz, num_of_basis_functions))
    for m in range(num_of_basis_functions):
        # Use np.flip only once outside the loop
        kernel = np.flip(np.flip(basis[:, :, m], axis=0), axis=1)
        S[:, :, m] = convolve2d(img, kernel, mode='same') ** 2
    return np.sum(S, axis=2)
def convolve_and_sum(img, basis, num_of_basis_functions):
    img_sz = img.shape[0]
    S = torch.zeros((img_sz, img_sz, num_of_basis_functions), device=device)

    # Precompute flipped kernels and make a copy to avoid negative strides
    flipped_kernels = np.flip(np.flip(basis, axis=0), axis=1).copy()

    # Move data to device
    img_tensor = torch.tensor(img, device=device, dtype=torch.float32)
    flipped_kernels_tensor = torch.tensor(flipped_kernels, device=device, dtype=torch.float32)

    for m in range(num_of_basis_functions):
        kernel = flipped_kernels_tensor[:, :, m]
        convolved = torch.nn.functional.conv2d(img_tensor.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding='same')
        S[:, :, m] = convolved.squeeze() ** 2

    return torch.sum(S, axis=2).cpu().numpy()
def save_fig(directory, filename, fig):
    """
    Saves a Matplotlib figure to the specified directory.

    Parameters:
    directory : Path to the directory where the figure will be saved.
    filename  : Name of the file.
    fig       : Matplotlib figure object.
    """
    filepath = os.path.join(directory, filename)
    fig.savefig(filepath, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
def coords_output2(Y_peaks_loc, addr_coords, microName, mgScale, mgBigSz, K, patchSzPickBox=300):
    """
    Writes particle coordinates to .star and .box files.

    Parameters:
    Y_peaks_loc : ndarray
        Array of peak locations.
    addr_coords : str
        Directory address where files will be saved.
    microName : str
        Name of the micrograph.
    mgScale : float
        Scaling factor for micrograph.
    mgBigSz : tuple
        Size of the micrograph (height, width).
    K : int
        Number of peaks to process.
    patchSzPickBox : int, optional
        Size of the patch for picking box. Default is 300.
    """

    # Ensure the directory exists
    os.makedirs(addr_coords, exist_ok=True)

    # Open files for writing
    particlesCordinateStar_path = os.path.join(addr_coords, f'{microName}.star')
    particlesCordinateBox_path = os.path.join(addr_coords, f'{microName}.box')

    with open(particlesCordinateStar_path, 'w') as particlesCordinateStar, \
            open(particlesCordinateBox_path, 'w') as particlesCordinateBox:

        # Format Relion star file
        particlesCordinateStar.write('data_\n\n')
        particlesCordinateStar.write('loop_\n')
        particlesCordinateStar.write('_rlnCoordinateX #1\n')
        particlesCordinateStar.write('_rlnCoordinateY #2\n')

        for i in range(K):
            i_colPatch = Y_peaks_loc[i, 1]
            i_rowPatch = Y_peaks_loc[i, 0]
            x_star = (1 / mgScale) * i_colPatch
            y_star = (mgBigSz[0] + 1) - (1 / mgScale) * i_rowPatch
            x_box = x_star - np.floor(patchSzPickBox / 2)
            y_box = y_star - np.floor(patchSzPickBox / 2)

            particlesCordinateStar.write(f'{x_star:.0f}\t{y_star:.0f}\n')
            particlesCordinateBox.write(f'{x_box:.0f}\t{y_box:.0f}\t{patchSzPickBox:.0f}\t{patchSzPickBox:.0f}\n')
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



def coords_output_csv(Y_peaks_loc, addr_coords, microName, mgScale, mgBigSz, K, patchSzPickBox, ground_truth_csv=None):
    """
    Writes particle coordinates to .csv (and optionally .star/.box) files for given peaks.
    
    Parameters:
    Y_peaks_loc : Array of peak locations.
    addr_coords : Directory path for output files.
    microName   : Name of the micrograph.
    mgScale     : Scaling factor for micrograph.
    mgBigSz     : Size of the micrograph (tuple or list).
    K           : Number of peaks to output.
    patchSzPickBox : Size of the box around each particle.
    ground_truth_csv : Path to the ground truth .csv file (optional).
    """
    # Prepare file paths
    csv_file_path = os.path.join(addr_coords, f"{microName}.csv")
    star_file_path = os.path.join(addr_coords, f"{microName}.star")
    box_file_path = os.path.join(addr_coords, f"{microName}.box")
    
    # Initialize ground truth data if provided
    ground_truth = None
    if ground_truth_csv and os.path.exists(ground_truth_csv):
        ground_truth = pd.read_csv(ground_truth_csv)
    
    # Open files for writing
    with open(star_file_path, 'w') as particlesCordinateStar, open(box_file_path, 'w') as particlesCordinateBox:
        # Write headers for the .star file
        particlesCordinateStar.write("data_\n\nloop_\n")
        particlesCordinateStar.write("_rlnCoordinateX #1\n")
        particlesCordinateStar.write("_rlnCoordinateY #2\n\n")

        # Prepare .csv output
        output_data = []

        for i in range(K):
            i_colPatch = Y_peaks_loc[i, 1]
            i_rowPatch = Y_peaks_loc[i, 0]

            # Calculate coordinates for .star and .csv
            x_coord = (1 / mgScale) * i_colPatch
            y_coord = (mgBigSz[0] + 1) - (1 / mgScale) * i_rowPatch
            output_data.append([x_coord, y_coord])

            # Write to .star file
            particlesCordinateStar.write(f"{x_coord:.0f}\t{y_coord:.0f}\n")

            # Calculate coordinates for .box file
            x_box = x_coord - patchSzPickBox // 2
            y_box = y_coord - patchSzPickBox // 2
            particlesCordinateBox.write(f"{x_box:.0f}\t{y_box:.0f}\t{patchSzPickBox:.0f}\t{patchSzPickBox:.0f}\n")
        
        # Write to .csv
        output_df = pd.DataFrame(output_data, columns=["x", "y"])
        output_df.to_csv(csv_file_path, index=False)
    
    if ground_truth is not None:
        # Compare with ground truth
        comparison = pd.merge(output_df, ground_truth, on=["x", "y"], how="inner")
        print(f"Matched particles: {len(comparison)} / {K}")
        comparison.to_csv(os.path.join(addr_coords, f"{microName}_matched.csv"), index=False)



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

def extract_non_overlapping_patches(image, patch_size, step):
    """
    Extract patches of size (patch_size x patch_size) from the input image.
    The next patch's top-left corner is step pixels apart.
    Returns an array of patches and the patch top-left corners (i, j).
    """
    patches = []
    centers = []
    img_height, img_width = image.shape
    patch_height, patch_width = patch_size

    # Compute how many patches we can fit along each dimension
    n_patches_y = (img_height - patch_height) // step + 1
    n_patches_x = (img_width - patch_width) // step + 1

    # Loop over the image using the calculated number of patches
    for i in range(n_patches_y):
        for j in range(n_patches_x):
            start_y = i * step
            start_x = j * step
            patch = image[start_y:start_y + patch_height, start_x:start_x + patch_width]
            patches.append(patch)
            centers.append((start_y, start_x))  # Store the top-left corner of each patch

    return np.array(patches), centers



# def compute_projection_norms(patches, basis_vectors):
#     """
#     Computes the norm of the projection of each patch onto the given 30 basis vectors.
#     Patches are of shape (64, 64), and basis_vectors is of shape (64, 64, 30).
#     """
#     n_patches = patches.shape[0]
#     n_basis = basis_vectors.shape[2]  # Number of basis vectors (30)

#     projection_norms = np.zeros(n_patches)
#     for i in range(n_patches):
#         patch_flattened = patches[i].flatten()

#         # For each basis vector, compute the projection
#         inner_products = np.zeros(n_basis)
#         for j in range(n_basis):
#             basis_flattened = basis_vectors[:, :, j].flatten()
#             inner_products[j] = np.dot(patch_flattened, basis_flattened)

#         # Compute the norm of the projection onto the basis set
#         sum_of_squares = np.sum(inner_products ** 2)
#         projection_norms[i] = np.sqrt(sum_of_squares)

#     return projection_norms



def peak_algorithm_cont_mask(img, basis, sideLengthAlgorithm, contamination_mask=None, obj_sz_down_scaled=None, contamination_threshold=0.5, debug=False):
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
    # Ensure using CPU (change to 'cuda' if GPU is available)
    device = torch.device('cpu')

    # Convert image and basis to PyTorch tensors and move to device (CPU)
    img_tensor = torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    basis = torch.tensor(basis, dtype=torch.float32, device=device)
    num_of_basis_functions = basis.shape[2]
    rDelAlgorithm = round(sideLengthAlgorithm // 2)

    peaks = []
    peaks_loc = []
    obj_sz = basis.shape[0]

    # Perform convolution and sum over basis functions
    S = torch.zeros_like(img_tensor, dtype=torch.float32, device=device)

    flipped_basis_list = [torch.flip(basis[:, :, j], dims=[0, 1]) for j in range(num_of_basis_functions)]
    for flipped_basis in flipped_basis_list:
        flipped_basis_shape = flipped_basis.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        conv_result = F.conv2d(img_tensor, flipped_basis_shape, stride=1, padding='same') ** 2
        S += conv_result

    S = S.squeeze().cpu().numpy()
    scoringMat = S.copy()
    scoringMat[:obj_sz // 2, :] = 0
    scoringMat[:, :obj_sz // 2] = 0
    scoringMat[-obj_sz // 2:, :] = 0
    scoringMat[:, -obj_sz // 2:] = 0

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
        rows, cols = np.ogrid[:scoringMat.shape[0], :scoringMat.shape[1]]
        mask = (rows - i_row) ** 2 + (cols - i_col) ** 2 <= rDelAlgorithm ** 2
        if debug:
            mask_circ = ((rows - i_row) ** 2 + (cols - i_col) ** 2 >= rDelAlgorithm ** 2 - 50) & \
                        ((rows - i_row) ** 2 + (cols - i_col) ** 2 <= rDelAlgorithm ** 2)
            img_copy = img.copy()
            img_copy[mask_circ] = 3000

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

    num_of_basis_functions = basis.shape[2]
    rDelAlgorithm = round(sideLengthAlgorithm // 2)
    obj_sz = basis.shape[0]
    basis = basis.astype(np.float32)
    img = img.astype(np.float32)

    # Pre-allocate the scoring map
    S = np.zeros(img.shape, dtype=np.float32)

    # Perform convolution using fftconvolve from SciPy
    for j in range(num_of_basis_functions):
        # Flip the basis function horizontally and vertically
        flipped_basis = np.flip(np.flip(basis[:, :, j], axis=0), axis=1)

        # Perform convolution and sum of squared convolutions
        conv_result = fftconvolve(img, flipped_basis, mode='same')
        S += conv_result ** 2

    scoringMat = S.copy()

    # Zero out the edges to avoid boundary artifacts
    scoringMat[:obj_sz // 2, :] = 0
    scoringMat[:, :obj_sz // 2] = 0
    scoringMat[-obj_sz // 2:, :] = 0
    scoringMat[:, -obj_sz // 2:] = 0

    peaks = []
    peaks_loc = []

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
        rows, cols = np.ogrid[:scoringMat.shape[0], :scoringMat.shape[1]]
        mask = (rows - i_row) ** 2 + (cols - i_col) ** 2 <= rDelAlgorithm ** 2
        if debug:
            mask_circ = ((rows - i_row) ** 2 + (cols - i_col) ** 2 >= rDelAlgorithm ** 2 - 50) & \
                        ((rows - i_row) ** 2 + (cols - i_col) ** 2 <= rDelAlgorithm ** 2)
            img_copy = img.copy()
            img_copy[mask_circ] = 3000

        scoringMat[mask] = 0
        peaks.append(pMax)
        peaks_loc.append([i_row, i_col])

    return np.array(peaks), np.array(peaks_loc), S

def peak_algorithm_cont_mask_torch_swap(img, basis, sideLengthAlgorithm, contamination_mask=None, obj_sz_down_scaled=None,
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
    # Use GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert image and basis to PyTorch tensors
    img_tensor = torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    basis_tensor = torch.tensor(basis, dtype=torch.float32, device=device)

    num_of_basis_functions = basis_tensor.shape[2]
    rDelAlgorithm = round(sideLengthAlgorithm // 2)
    obj_sz = basis_tensor.shape[0]

    # Pre-allocate the scoring map and move it to the device
    S = torch.zeros_like(img_tensor, dtype=torch.float32, device=device)

    # Perform convolution using in-place operations to save memory
    for j in range(num_of_basis_functions):
        flipped_basis = torch.flip(basis_tensor[:, :, j], dims=[0, 1])
        flipped_basis = flipped_basis.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        # Use in-place operation for convolution result
        conv_result = F.conv2d(img_tensor, flipped_basis, stride=1, padding='same')
        conv_result.pow_(2)  # In-place square operation
        S.add_(conv_result)  # In-place addition

    # Move the scoring map back to CPU and convert to NumPy
    S = S.squeeze().cpu().numpy()
    scoringMat = S.copy()

    # Clear the cache to free up GPU memory
    torch.cuda.empty_cache()

    # Zero out the edges to avoid boundary artifacts
    scoringMat[:obj_sz // 2, :] = 0
    scoringMat[:, :obj_sz // 2] = 0
    scoringMat[-obj_sz // 2:, :] = 0
    scoringMat[:, -obj_sz // 2:] = 0

    peaks = []
    peaks_loc = []

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
        rows, cols = np.ogrid[:scoringMat.shape[0], :scoringMat.shape[1]]
        mask = (rows - i_row) ** 2 + (cols - i_col) ** 2 <= rDelAlgorithm ** 2
        if debug:
            mask_circ = ((rows - i_row) ** 2 + (cols - i_col) ** 2 >= rDelAlgorithm ** 2 - 50) & \
                        ((rows - i_row) ** 2 + (cols - i_col) ** 2 <= rDelAlgorithm ** 2)
            img_copy = img.copy()
            img_copy[mask_circ] = 3000

        scoringMat[mask] = 0
        peaks.append(pMax)
        peaks_loc.append([i_row, i_col])

    # Clear cache one last time before returning results
    torch.cuda.empty_cache()

    return np.array(peaks), np.array(peaks_loc), S


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
        rows, cols = np.ogrid[:scoringMat.shape[0], :scoringMat.shape[1]]
        mask = (rows - i_row) ** 2 + (cols - i_col) ** 2 <= rDelAlgorithm ** 2
        if debug:
            mask_circ = ((rows - i_row) ** 2 + (cols - i_col) ** 2 >= rDelAlgorithm ** 2 - 50) & \
                        ((rows - i_row) ** 2 + (cols - i_col) ** 2 <= rDelAlgorithm ** 2)
            img_copy = img_no_mean.copy()
            img_copy[mask_circ] = 3000

        scoringMat[mask] = 0
        peaks.append(pMax)
        peaks_loc.append([i_row, i_col])

    return np.array(peaks), np.array(peaks_loc), S

def peak_algorithm_cont_mask_tf_old(img, basis, sideLengthAlgorithm, contamination_mask=None, obj_sz_down_scaled=None, contamination_threshold=0.5, debug=False):
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
        rows, cols = np.ogrid[:scoringMat.shape[0], :scoringMat.shape[1]]
        mask = (rows - i_row) ** 2 + (cols - i_col) ** 2 <= rDelAlgorithm ** 2
        if debug:
            mask_circ = ((rows - i_row) ** 2 + (cols - i_col) ** 2 >= rDelAlgorithm ** 2 - 50) & \
                        ((rows - i_row) ** 2 + (cols - i_col) ** 2 <= rDelAlgorithm ** 2)
            img_copy = img_no_mean.copy()
            img_copy[mask_circ] = 3000

        scoringMat[mask] = 0
        peaks.append(pMax)
        peaks_loc.append([i_row, i_col])

    return np.array(peaks), np.array(peaks_loc), S

def peak_algorithm_boxes(img, basis, sideLengthAlgorithm):
    """
    Identify peaks in an image using convolution with basis functions.

    Parameters:
    img                : Input image.
    basis              : Array of basis functions.
    sideLengthAlgorithm: Length parameter for peak extraction.

    Returns:
    peaks     : List of peak values.
    peaks_loc : List of peak locations (coordinates).
    S         : Scoring map from convolution.
    """

    num_of_basis_functions = basis.shape[2]

    img_sz_max = max(img.shape[0],img.shape[1])
    rDelAlgorithm = round(sideLengthAlgorithm // 2)
    #peaks_size = (img_sz_max // rDelAlgorithm) ** 2
    # Convert the result explicitly to an integer before using as array size
    #peaks_size = round(peaks_size)
    # Initialize lists to store peaks and their locations
    peaks = []
    peaks_loc = []
    obj_sz = basis.shape[0]
    basis_vectors = np.reshape(basis, (basis.shape[0]**2, basis.shape[2]))
    # S = convolve_and_sum(img, basis, num_of_basis_functions)
    img_shape = tf.reshape(img, [1, img.shape[0], img.shape[1], 1])
    img_shape = tf.cast(img_shape, tf.float32)
    flipped_basis = np.flip(np.flip(basis[:, :, -1], 0), 1)
    flipped_basis_shape = tf.reshape(flipped_basis,[flipped_basis.shape[0], flipped_basis.shape[1], 1, 1 ])
    flipped_basis_shape = tf.cast(flipped_basis_shape, tf.float32)
    S= tf.nn.conv2d(img_shape, flipped_basis_shape,strides=[1,1,1,1], padding='SAME') **2
    for j in range(basis.shape[2]-1):
        flipped_basis = np.flip(np.flip(basis[:, :, j], 0), 1)
        flipped_basis_shape = tf.reshape(flipped_basis,[flipped_basis.shape[0], flipped_basis.shape[1], 1, 1 ])
        flipped_basis_shape = tf.cast(flipped_basis_shape, tf.float32)
        S += tf.nn.conv2d(img_shape, flipped_basis_shape,strides=[1,1,1,1], padding='SAME') ** 2
    S = tf.squeeze(S).numpy()
    # Loop over each possible center point
    # B = basis.shape[0]
    # N = img_sz
    # half_B = int(B // 2)  # Half the patch size to find boundaries around the center
    # S_n = np.zeros((img_sz, img_sz))
    # for i in range(half_B, N - half_B):
    #     for j in range(half_B, N - half_B):
    #         # Extract the patch centered at (i, j)
    #         patch = img[i - half_B:i + half_B, j - half_B:j + half_B]
    #         patch_flattened = patch.flatten()
    #         inner_products = np.dot(basis_vectors.T,patch_flattened)
    #         sum_of_squares = np.sum(inner_products ** 2)
    #         projection_norm = np.sqrt(sum_of_squares)
    #         S_n[i,j] = projection_norm
    # Populate matrix S using the projection norms
    # for idx, (i, j) in enumerate(centers):
    #     S[i, j] = projection_norms[idx]
    scoringMat = S.copy()
    scoringMat[:obj_sz // 2, :] = 0
    scoringMat[:, :obj_sz // 2] = 0
    scoringMat[-obj_sz // 2:, :] = 0
    scoringMat[:, -obj_sz // 2:] = 0


    idxRow = np.arange(scoringMat.shape[0])
    idxCol = np.arange(scoringMat.shape[1])
    cnt = 0
    pMax = 1

    while pMax > 0:
        # Find the maximum value and its location in the scoring matrix
        pMax = np.max(scoringMat)
        if pMax <= 0:
            break
        I = np.argmax(scoringMat)
        i_row, i_col = np.unravel_index(I, scoringMat.shape)


        # Define the bounding box limits for row and column indices
        row_start = max(i_row - rDelAlgorithm, 0)
        row_end = min(i_row + rDelAlgorithm, scoringMat.shape[0])
        col_start = max(i_col - rDelAlgorithm, 0)
        col_end = min(i_col + rDelAlgorithm, scoringMat.shape[1])

        # Zero out the region corresponding to the bounding box
        scoringMat[row_start:row_end+1, col_start:col_end+1] = 0

        # Debug print: count zeroed-out pixels
        num_zeroed_out = np.sum(scoringMat == 0)
        print(f"Number of zeroed-out pixels: {num_zeroed_out}")

        cnt += 1

        # Append peak value and location
        peaks.append(pMax)
        peaks_loc.append([i_row, i_col])

    # Convert lists to NumPy arrays
    peaks = np.array(peaks)
    peaks_loc = np.array(peaks_loc)

    return peaks, peaks_loc, S


def peak_algorithm(img, basis, sideLengthAlgorithm):
    """
    Identify peaks in an image using convolution with basis functions.

    Parameters:
    img                : Input image.
    basis              : Array of basis functions.
    sideLengthAlgorithm: Length parameter for peak extraction.

    Returns:
    peaks     : List of peak values.
    peaks_loc : List of peak locations (coordinates).
    S         : Scoring map from convolution.
    """

    num_of_basis_functions = basis.shape[2]

    img_sz_max = max(img.shape[0], img.shape[1])
    rDelAlgorithm = round(sideLengthAlgorithm // 2)

    peaks = []
    peaks_loc = []
    obj_sz = basis.shape[0]
    basis_vectors = np.reshape(basis, (basis.shape[0] ** 2, basis.shape[2]))

    # Perform convolution and sum over basis functions
    img_shape = tf.reshape(img, [1, img.shape[0], img.shape[1], 1])
    img_shape = tf.cast(img_shape, tf.float32)
    flipped_basis = np.flip(np.flip(basis[:, :, -1], 0), 1)
    flipped_basis_shape = tf.reshape(flipped_basis, [flipped_basis.shape[0], flipped_basis.shape[1], 1, 1])
    flipped_basis_shape = tf.cast(flipped_basis_shape, tf.float32)
    S = tf.nn.conv2d(img_shape, flipped_basis_shape, strides=[1, 1, 1, 1], padding='SAME') ** 2
    for j in range(basis.shape[2] - 1):
        flipped_basis = np.flip(np.flip(basis[:, :, j], 0), 1)
        flipped_basis_shape = tf.reshape(flipped_basis, [flipped_basis.shape[0], flipped_basis.shape[1], 1, 1])
        flipped_basis_shape = tf.cast(flipped_basis_shape, tf.float32)
        S += tf.nn.conv2d(img_shape, flipped_basis_shape, strides=[1, 1, 1, 1], padding='SAME') ** 2
    S = tf.squeeze(S).numpy()

    # Copy the scoring matrix and apply boundary conditions
    scoringMat = S.copy()
    img_copy = img.copy()
    scoringMat[:obj_sz // 2, :] = 0
    scoringMat[:, :obj_sz // 2] = 0
    scoringMat[-obj_sz // 2:, :] = 0
    scoringMat[:, -obj_sz // 2:] = 0

    cnt = 0
    pMax = 1

    while pMax > 0:
        # Find the maximum value and its location in the scoring matrix
        pMax = np.max(scoringMat)
        if pMax <= 0:
            break
        I = np.argmax(scoringMat)
        i_row, i_col = np.unravel_index(I, scoringMat.shape)

        # Create a circular mask to delete a "ball" around the peak
        rows, cols = np.ogrid[:scoringMat.shape[0], :scoringMat.shape[1]]
        mask = (rows - i_row) ** 2 + (cols - i_col) ** 2 <= rDelAlgorithm ** 2
        mask_circ = ((rows - i_row) ** 2 + (cols - i_col) ** 2 >= rDelAlgorithm ** 2 - 50) & \
                    ((rows - i_row) ** 2 + (cols - i_col) ** 2 <= rDelAlgorithm ** 2)
        # Zero out the circular region (ball) in the scoring matrix
        scoringMat[mask] =0
        img_copy[mask_circ] = 3000

        cnt += 1

        # Append peak value and location
        peaks.append(pMax)
        peaks_loc.append([i_row, i_col])

    # Convert lists to NumPy arrays
    peaks = np.array(peaks)
    peaks_loc = np.array(peaks_loc)

    return peaks, peaks_loc, S




def names_sub_folder(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]


def projected_noise_simulation_from_noise_patches(noise_samples, basis, num_of_exp_noise):
    """
    Simulates projected noise using given noise samples and basis functions in batches.

    Parameters:
    noise_samples : Array of noise samples.
    basis         : Array of basis functions.
    num_of_exp_noise : Number of noise experiments.

    Returns:
    S_z : Simulated noise projections.
    """

    noise_samples = noise_samples.astype(np.float32)
    basis = basis.astype(np.float32)

    sz = int(np.sqrt(noise_samples.shape[0]))  # Image size
    batch_size = max(1, num_of_exp_noise // 10)  # Define batch size (at least 1)
    num_batches = (num_of_exp_noise + batch_size - 1) // batch_size  # Compute number of batches

    # Compute output size
    sz_pn = tf.nn.conv2d(
        tf.reshape(noise_samples[:, 0], (1, sz, sz, 1)),
        tf.reshape(basis[:, :, 0], (basis.shape[0], basis.shape[1], 1, 1)),
        strides=[1, 1, 1, 1],
        padding='VALID'
    ).shape[1]

    S_z_n = np.zeros((num_of_exp_noise, sz_pn, sz_pn), dtype=np.float32)  # Initialize output
    flipped_basis_stack = tf.convert_to_tensor(
    np.stack([np.flip(np.flip(basis[:, :, j], 0), 1) for j in range(basis.shape[2])]),
    dtype=tf.float32
)

# Reshape to match TensorFlow's expected format: [filter_height, filter_width, in_channels, out_channels]
    flipped_basis_stack = tf.reshape(flipped_basis_stack, [basis.shape[0], basis.shape[1], 1, basis.shape[2]])
    
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, num_of_exp_noise)

        print(f"Processing batch {batch_idx+1}/{num_batches}, size: {end - start}")

        # Extract batch
        noise_batch = noise_samples[:,start:end]
        noise_imgs = np.reshape(noise_batch, (end - start, sz, sz, 1)).astype(np.float32)
        noise_imgs_tf = tf.convert_to_tensor(noise_imgs, dtype=tf.float32)

        # for j in range(basis.shape[2]):
        #     flipped_basis = np.flip(np.flip(basis[:, :, j], 0), 1)  # Flip basis
        #     flipped_basis_tf = tf.convert_to_tensor(
        #         tf.reshape(flipped_basis, (basis.shape[0], basis.shape[1], 1, 1)),
        #         dtype=tf.float32
        #     )

        # # Perform convolution
        # conv_result = tf.nn.conv2d(noise_imgs_tf, flipped_basis_tf, strides=[1, 1, 1, 1], padding='VALID')**2

        conv_result = tf.nn.conv2d(noise_imgs_tf, flipped_basis_stack, strides=[1, 1, 1, 1], padding='VALID') ** 2
         # Sum over all basis functions
        conv_result = tf.reduce_sum(conv_result, axis=-1)  # Correct summation over all basis functions
        # Store results in corresponding index
        S_z_n[start:end,:,:] += tf.squeeze(conv_result).numpy()

        # Clear GPU memory after processing each batch
        del noise_imgs_tf, conv_result
        tf.keras.backend.clear_session()

    S_z_n = np.transpose(S_z_n, (1, 2, 0))
    return S_z_n



def projected_noise_simulation_from_noise_patches_para(noise_samples, basis, num_of_exp_noise):
    """
    Simulates projected noise using given noise samples and basis functions in batches,
    fully optimized for GPU execution.

    Parameters:
    noise_samples : np.ndarray
        Array of noise samples.
    basis : np.ndarray
        Array of basis functions.
    num_of_exp_noise : int
        Number of noise experiments.

    Returns:
    np.ndarray
        Simulated noise projections.
    """

    noise_samples = noise_samples.astype(np.float32)
    basis = basis.astype(np.float32)

    sz = int(np.sqrt(noise_samples.shape[0]))  # Image size

    # ✅ Dynamically adjust batch size based on available GPU memory
    def get_optimal_batch_size(base_size=500):
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

    print(f"Using GPU acceleration. Batch size: {batch_size}")

    # Compute output size
    sz_pn = tf.nn.conv2d(
        tf.reshape(noise_samples[:, 0], (1, sz, sz, 1)),
        tf.reshape(basis[:, :, 0], (basis.shape[0], basis.shape[1], 1, 1)),
        strides=[1, 1, 1, 1],
        padding='VALID'
    ).shape[1]

    S_z_n = np.zeros((num_of_exp_noise, sz_pn, sz_pn), dtype=np.float32)

    # ✅ Precompute flipped basis stack on GPU
    with tf.device('/GPU:0'):
        flipped_basis_stack = tf.convert_to_tensor(
            np.stack([np.flip(np.flip(basis[:, :, j], 0), 1) for j in range(basis.shape[2])]),
            dtype=tf.float32
        )
        flipped_basis_stack = tf.reshape(flipped_basis_stack, [basis.shape[0], basis.shape[1], 1, basis.shape[2]])

    # ✅ TensorFlow compiled function for fast GPU execution
    @tf.function
    def process_batch(noise_imgs_tf, flipped_basis_stack):
        conv_result = tf.nn.conv2d(noise_imgs_tf, flipped_basis_stack, strides=[1, 1, 1, 1], padding='VALID') ** 2
        conv_result = tf.reduce_sum(conv_result, axis=-1)  # Sum over all basis functions
        return conv_result

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, num_of_exp_noise)

        print(f"Processing batch {batch_idx+1}/{num_batches}, size: {end - start}")

        with tf.device('/GPU:0'):
            noise_batch = noise_samples[:, start:end]
            noise_imgs = np.reshape(noise_batch, (end - start, sz, sz, 1)).astype(np.float32)
            noise_imgs_tf = tf.convert_to_tensor(noise_imgs, dtype=tf.float32)

            # Perform convolution on GPU
            conv_result = process_batch(noise_imgs_tf, flipped_basis_stack)

            # Store results in corresponding index
            S_z_n[start:end, :, :] = conv_result.numpy()

            # Clear GPU memory
            del noise_imgs_tf, conv_result
            tf.keras.backend.clear_session()

    return np.transpose(S_z_n, (1, 2, 0))  # Match expected output shape



def projected_noise_simulation_from_noise_patches_para_fast(noise_samples, basis, num_of_exp_noise):
    """
    Simulates projected noise using given noise samples and basis functions in batches,
    distributing computation across multiple GPUs (if available).
    """

    noise_samples = noise_samples.astype(np.float32)
    basis = basis.astype(np.float32)
    sz = int(np.sqrt(noise_samples.shape[0]))  # Image size

    # Determine optimal batch size based on available GPU memory
    def get_optimal_batch_size(base_size=1000):
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            batch_size = max(1, int(free_memory / (sz * sz * 4 * 2)))  # Estimate batch size
            print(f"Adjusted batch size: {batch_size}")
            return batch_size
        return base_size

    batch_size = get_optimal_batch_size()
    num_batches = (num_of_exp_noise + batch_size - 1) // batch_size  # Compute number of batches

    # List available GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        raise RuntimeError("No GPUs found! Consider using a CPU-based approach.")
    print(f"Using {num_gpus} GPUs for parallel processing. Batch size: {batch_size}")

    # Compute output size
    noise_sample_tensor = torch.tensor(noise_samples[:, 0], dtype=torch.float32).view(1, 1, sz, sz).to(device)
    basis_tensor = torch.tensor(basis[:, :, 0], dtype=torch.float32).view(1, 1, basis.shape[0], basis.shape[1]).to(device)
    sz_pn = F.conv2d(noise_sample_tensor, basis_tensor, stride=1, padding=0).shape[-1]

    # Initialize output tensor
    S_z_n = np.zeros((num_of_exp_noise, sz_pn, sz_pn), dtype=np.float32)

    # Flip basis stack
    flipped_basis_stack = np.stack([np.flip(np.flip(basis[:, :, j], 0), 1) for j in range(basis.shape[2])])
    flipped_basis_stack = torch.tensor(flipped_basis_stack, dtype=torch.float32).view(basis.shape[2], 1, basis.shape[0], basis.shape[1]).to(device)

    # Function to process a batch
    def process_batch(noise_imgs_tensor, flipped_basis_stack):
        conv_result = F.conv2d(noise_imgs_tensor, flipped_basis_stack, stride=1, padding=0) ** 2
        conv_result = torch.sum(conv_result, dim=1)  # Sum over all basis functions
        return conv_result

    results = [None] * num_batches  

    # Process batches
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, num_of_exp_noise)

        gpu_idx = batch_idx % num_gpus
        device_name = f"cuda:{gpu_idx}"
        print(f"Processing batch {batch_idx+1}/{num_batches} on {device_name}")

        noise_batch = noise_samples[:, start:end]
        noise_imgs = np.reshape(noise_batch, (end - start, sz, sz, 1)).astype(np.float32)
        noise_imgs_tensor = torch.tensor(noise_imgs, dtype=torch.float32).permute(0, 3, 1, 2).to(device)

        with torch.no_grad():  # Disable gradient calculations for faster computation
            results[batch_idx] = process_batch(noise_imgs_tensor, flipped_basis_stack).cpu().numpy()

        # Clear GPU memory
        del noise_imgs_tensor
        torch.cuda.empty_cache()

    # Collect results
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, num_of_exp_noise)
        S_z_n[start:end, :, :] += results[batch_idx]

    return np.transpose(S_z_n, (1, 2, 0))  # Match expected output shape


def projected_noise_simulation_from_noise_patches_para_fast_cpu(noise_samples, basis, num_of_exp_noise):
    """
    Simulates projected noise using given noise samples and basis functions in batches,
    distributing computation across multiple CPUs.
    """

    # Ensure using CPU
    device = torch.device('cpu')

    # Convert to PyTorch tensors and move to CPU
    noise_samples = torch.tensor(noise_samples, dtype=torch.float32, device=device)
    basis = torch.tensor(basis, dtype=torch.float32, device=device)

    sz = int(np.sqrt(noise_samples.shape[0]))  # Image size

    # Dynamically determine batch size based on available memory
    def get_optimal_batch_size(base_size=1000):
        return base_size  # Default batch size for CPU

    batch_size = get_optimal_batch_size()
    num_batches = (num_of_exp_noise + batch_size - 1) // batch_size  # Compute number of batches

    print(f"Using CPU for processing. Batch size: {batch_size}")

    # Compute output size
    example_image = noise_samples[:, 0].reshape(1, 1, sz, sz)
    example_basis = basis[:, :, 0].reshape(1, 1, basis.shape[0], basis.shape[1])
    sz_pn = F.conv2d(example_image, example_basis, stride=1, padding=0).shape[-1]

    # Initialize output tensor
    S_z_n = torch.zeros((num_of_exp_noise, sz_pn, sz_pn), dtype=torch.float32, device=device)

    # Flip basis stack and reshape for PyTorch convolution
    flipped_basis_stack = torch.stack([torch.flip(basis[:, :, j], [0, 1]) for j in range(basis.shape[2])], dim=0)
    flipped_basis_stack = flipped_basis_stack.unsqueeze(1)  # [num_filters, 1, H, W]

    # Process batches
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, num_of_exp_noise)

        print(f"Processing batch {batch_idx+1}/{num_batches} on CPU")

        # Extract batch and reshape
        noise_batch = noise_samples[:, start:end]
        noise_imgs = noise_batch.reshape(end - start, 1, sz, sz)

        # Perform convolution and sum over basis functions
        conv_result = F.conv2d(noise_imgs, flipped_basis_stack, stride=1, padding=0) ** 2
        conv_result = torch.sum(conv_result, dim=1)  # Sum over all basis functions
        
        # Store results in corresponding index
        S_z_n[start:end, :, :] = conv_result

    return S_z_n.permute(1, 2, 0).cpu().numpy()  # Match expected output shape and move to NumPy

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

import torch
import numpy as np

import torch
import numpy as np


def projected_noise_simulation_from_noise_patches_torch(noise_samples, basis, num_of_exp_noise):
    """
    Simulates projected noise using given noise samples and basis functions using PyTorch.

    Parameters:
    noise_samples : Array of noise samples.
    basis         : Array of basis functions.
    num_of_exp_noise : Number of noise experiments.

    Returns:
    S_z : Simulated noise projections.
    """
    # Convert inputs to float32 for consistent precision
    noise_samples = noise_samples.astype(np.float32)
    basis = basis.astype(np.float32)

    sz = int(np.sqrt(noise_samples.shape[0]))  # Image size
    noise_imgs = noise_samples.reshape(
        (num_of_exp_noise, 1, sz, sz))  # Reshape noise samples to 4D (batch, channels, height, width)

    # Convert to PyTorch tensors
    noise_imgs_torch = torch.tensor(noise_imgs, dtype=torch.float32)

    # Flip basis stack to be consistent with TensorFlow's behavior
    flipped_basis_stack = np.stack([np.flip(np.flip(basis[:, :, j], axis=0), axis=1)
                                    for j in range(basis.shape[2])])
    flipped_basis_stack = torch.tensor(flipped_basis_stack, dtype=torch.float32)
    flipped_basis_stack = flipped_basis_stack.unsqueeze(1)  # (out_channels, in_channels, H, W)

    # Perform convolution using torch.nn.functional.conv2d
    conv_result = torch.nn.functional.conv2d(noise_imgs_torch, flipped_basis_stack, padding=0)
    conv_result = conv_result ** 2  # Square the convolution results (as in TensorFlow)
    conv_result = torch.sum(conv_result, dim=1)  # Sum over the channel dimension

    # Convert back to NumPy and transpose to match TensorFlow output shape
    S_z_n = conv_result.detach().numpy()
    S_z_n = np.transpose(S_z_n, (1, 2, 0))

    return S_z_n


def projected_noise_simulation_from_noise_patches_scipy(noise_samples, basis, num_of_exp_noise):
    """
    Simulates projected noise using given noise samples and basis functions using SciPy's FFT for speed.

    Parameters:
    noise_samples : Array of noise samples (NumPy array).
    basis         : Array of basis functions (NumPy array).
    num_of_exp_noise : Number of noise experiments.

    Returns:
    S_z : Simulated noise projections.
    """
    # Convert inputs to float32 for consistent precision
    noise_samples = noise_samples.astype(np.float32)
    basis = basis.astype(np.float32)

    # Reshape noise samples to 4D (num_of_exp_noise, height, width, channels)
    sz = int(np.sqrt(noise_samples.shape[0]))  # Image size
    noise_imgs = noise_samples.reshape((num_of_exp_noise, sz, sz, 1))

    # Flip the basis functions to be consistent with TensorFlow's behavior
    flipped_basis_stack = np.stack([np.flip(np.flip(basis[:, :, j], axis=0), axis=1)
                                    for j in range(basis.shape[2])])

    # Determine output size using a single convolution
    example_conv = fftconvolve(noise_imgs[0, :, :, 0], flipped_basis_stack[0, :, :], mode='valid')
    sz_pn = example_conv.shape[0]

    # Initialize the output array
    S_z = np.zeros((num_of_exp_noise, sz_pn, sz_pn), dtype=np.float32)

    # Perform convolution for each basis function and noise image using FFT
    for j in range(basis.shape[2]):
        for i in range(num_of_exp_noise):
            conv_result = fftconvolve(noise_imgs[i, :, :, 0], flipped_basis_stack[j, :, :], mode='valid')
            S_z[i, :, :] += conv_result   # Sum of squared convolutions

    # Transpose to match the expected output shape (height, width, num_of_exp_noise)
    S_z = np.transpose(S_z, (1, 2, 0))
    return S_z


import tensorflow as tf

def projected_noise_simulation_from_noise_patches_tf(noise_samples, basis, num_of_exp_noise):
    """
    Simulates projected noise using given noise samples and basis functions.

    Parameters:
    noise_samples : Array of noise samples.
    basis         : Array of basis functions.
    num_of_exp_noise : Number of noise experiments.

    Returns:
    S_z : Simulated noise projections.
    """
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


def coortodist2(ind, sz):
    """
    Calculates the distance matrix and related data from coordinate indices.

    Parameters:
    ind : List or array of indices.
    sz  : Size of the square grid.

    Returns:
    coortodistData : A dictionary containing:
        - mat_of_radii  : Matrix of radii (distances) between index pairs.
        - dist_vec_uniq : Unique distances vector.
        - i_d_vec       : Indices of unique distances.
        - i_d_mat       : Matrix form of unique indices.
    """

    # Create a grid of row and column indices
    coords = np.array(np.unravel_index(ind, (sz, sz))).T

    # Calculate pairwise distances using broadcasting
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    mat_of_radii = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Extract unique distances and their indices
    dist_vec_uniq, i_d_vec, i_d_mat = np.unique(mat_of_radii.ravel(), return_index=True, return_inverse=True)

    coortodistData = {
        'mat_of_radii': mat_of_radii,
        'dist_vec_uniq': dist_vec_uniq,
        'i_d_vec': i_d_vec,
        'i_d_mat': i_d_mat.reshape(mat_of_radii.shape)
    }

    return coortodistData
@jit(nopython=True, parallel=True)
def calculate_pairwise_distances(coords):
    sz = coords.shape[0]
    mat_of_radii = np.zeros((sz, sz), dtype=np.float64)
    for i in prange(sz):
        for j in range(sz):
            diff = coords[i] - coords[j]
            mat_of_radii[i, j] = np.sqrt(np.sum(diff ** 2))
    return mat_of_radii

def coortodist3(ind, sz):
    """
    Calculates the distance matrix and related data from coordinate indices.

    Parameters:
    ind : List or array of indices.
    sz  : Size of the square grid.

    Returns:
    coortodistData : A dictionary containing:
        - mat_of_radii  : Matrix of radii (distances) between index pairs.
        - dist_vec_uniq : Unique distances vector.
        - i_d_vec       : Indices of unique distances.
        - i_d_mat       : Matrix form of unique indices.
    """

    # Create a grid of row and column indices
    coords = np.array(np.unravel_index(ind, (sz, sz))).T

    # Calculate pairwise distances using Numba for parallel computation
    mat_of_radii = calculate_pairwise_distances(coords)

    # Extract unique distances and their indices
    dist_vec_uniq, i_d_vec, i_d_mat = np.unique(mat_of_radii.ravel(), return_index=True, return_inverse=True)

    coortodistData = {
        'mat_of_radii': mat_of_radii,
        'dist_vec_uniq': dist_vec_uniq,
        'i_d_vec': i_d_vec,
        'i_d_mat': i_d_mat.reshape(mat_of_radii.shape)
    }

    return coortodistData
def coortodist(ind, sz):
    """
    Calculates the distance matrix and related data from coordinate indices.

    Parameters:
    ind : List or array of indices.
    sz  : Size of the square grid.

    Returns:
    coortodistData : A dictionary containing:
        - mat_of_radii  : Matrix of radii (distances) between index pairs.
        - dist_vec_uniq : Unique distances vector.
        - i_d_vec       : Indices of unique distances.
        - i_d_mat       : Matrix form of unique indices.
    """

    # Create a grid of row and column indices
    coords = np.array(np.unravel_index(ind, (sz, sz))).T

    # Use SciPy to calculate pairwise Euclidean distances efficiently
    mat_of_radii = squareform(pdist(coords, 'euclidean'))

    # Extract unique distances and their ipeak_algorithm_cont_maskndices
    dist_vec_uniq, i_d_vec, i_d_mat = np.unique(mat_of_radii.ravel(), return_index=True, return_inverse=True)

    coortodistData = {
        'mat_of_radii': mat_of_radii,
        'dist_vec_uniq': dist_vec_uniq,
        'i_d_vec': i_d_vec,
        'i_d_mat': i_d_mat.reshape(mat_of_radii.shape)
    }

    return coortodistData
def spatial_cov_from_radial_cov(radial_cov, r, mat_of_radii, dist_vec_uniq, i_d_mat):
    """
    Converts radial covariance into a spatial covariance matrix.

    Parameters:
    radial_cov    : Radial covariance values.
    r             : Radii corresponding to radial covariance.
    mat_of_radii  : Matrix of radii.
    dist_vec_uniq : Unique distance vector.
    i_d_mat       : Index matrix for distances.

    Returns:
    noise_cov : Spatial covariance matrix.
    """

    # Interpolate radial covariance
    radial_cov_interp_func = interp1d(r, radial_cov, kind='linear', fill_value=radial_cov[-1] * 0.1, bounds_error=False)
    radial_cov_interp = radial_cov_interp_func(dist_vec_uniq)

    noise_cov_vec_uniq = np.zeros_like(dist_vec_uniq)

    # Fill in the interpolated covariance values
    for i in range(len(noise_cov_vec_uniq)):
        if dist_vec_uniq[i] > r[-1]:
            noise_cov_vec_uniq[i:] = 0
            break
        noise_cov_vec_uniq[i] = radial_cov_interp[i]

    # Reshape into the spatial covariance matrix
    noise_cov = noise_cov_vec_uniq[i_d_mat].reshape(mat_of_radii.shape)

    return noise_cov


def rotate_image(image, angle):
    """
    Rotate a 2D numpy array (image) by a specified angle.

    :param image: 2D numpy array representing the image.
    :param angle: Angle in degrees to rotate the image.
    :return: Rotated image as a 2D numpy array.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")
    if not isinstance(angle, (int, float)):
        raise ValueError("Angle must be a number.")

    rotated_image = rotate(image, angle, reshape=False)
    return rotated_image


def bsearch(x, LowerBound, UpperBound):
    if LowerBound > x[-1] or UpperBound < x[0] or UpperBound < LowerBound:
        return None, None

    lower_index_a = 0
    lower_index_b = len(x) - 1
    upper_index_a = 0
    upper_index_b = len(x) - 1

    while (lower_index_a + 1 < lower_index_b) or (upper_index_a + 1 < upper_index_b):
        lw = (lower_index_a + lower_index_b) // 2

        if x[lw] >= LowerBound:
            lower_index_b = lw
        else:
            lower_index_a = lw
            if lw > upper_index_a and lw < upper_index_b:
                upper_index_a = lw

        up = (upper_index_a + upper_index_b + 1) // 2
        if x[up] <= UpperBound:
            upper_index_a = up
        else:
            upper_index_b = up
            if up < lower_index_b and up > lower_index_a:
                lower_index_b = up

    lower_index = lower_index_a if x[lower_index_a] >= LowerBound else lower_index_b
    upper_index = upper_index_b if x[upper_index_b] <= UpperBound else upper_index_a

    if upper_index < lower_index:
        return None, None

    return lower_index, upper_index


def cryo_epsdR(vol, samples_idx, max_d=None, verbose=0):
    p = vol.shape[0]
    if vol.shape[1] != p:
        raise ValueError('vol must be a stack of square images')

    if vol.ndim > 3:
        raise ValueError('vol must be a 3D array')

    K = 1 if vol.ndim == 2 else vol.shape[2]

    if max_d is None:
        max_d = p - 1

    max_d = min(max_d, p - 1)

    I, J = np.meshgrid(np.arange(max_d + 1), np.arange(max_d + 1))
    dists = I ** 2 + J ** 2
    dsquare = np.unique(dists[dists <= max_d ** 2])

    corrs = np.zeros(dsquare.shape[0])
    corrcount = np.zeros(dsquare.shape[0])
    x = np.sqrt(dsquare)

    distmap = -np.ones(dists.shape, dtype=int)
    for i in range(max_d + 1):
        for j in range(max_d + 1):
            d = i ** 2 + j ** 2
            if d <= max_d ** 2:
                lower_idx, upper_idx = bsearch(dsquare, d - 1.0e-13, d + 1.0e-13)
                if lower_idx is None or upper_idx is None or lower_idx != upper_idx:
                    raise ValueError('Something went wrong')
                distmap[i, j] = lower_idx

    validdists = np.where(distmap != -1)

    mask = np.zeros((p, p), dtype=bool)
    mask.flat[samples_idx] = True
    tmp = np.zeros((2 * p + 1, 2 * p + 1))
    tmp[:p, :p] = mask
    ftmp = fft2(tmp)
    c = np.real(ifft2(ftmp * np.conj(ftmp)))
    c = np.round(c[:max_d + 1, :max_d + 1]).astype(int)

    R = np.zeros(corrs.shape)

    for k in range(K):
        proj = vol[:, :, k] if K > 1 else vol

        samples = np.zeros((p, p))
        samples.flat[samples_idx] = proj.flat[samples_idx]

        tmp = np.zeros((2 * p + 1, 2 * p + 1))
        tmp[:p, :p] = samples
        ftmp = fft2(tmp)
        s = np.real(ifft2(ftmp * np.conj(ftmp)))
        s = s[:max_d + 1, :max_d + 1]

        for currdist in np.nditer(validdists):
            dmidx = distmap[tuple(currdist)]
            corrs[dmidx] += s[currdist]
            corrcount[dmidx] += c[currdist]

    idx = corrcount != 0
    R[idx] = corrs[idx] / corrcount[idx]
    cnt = corrcount[idx]

    x = x[idx]
    R = R[idx]

    return R, x, cnt


def estimate_noise_radial_cov_2(noise_img_scaled, box_sz, num_of_patches):
    rDelAlgorithm = int(box_sz / 2)+ 1
    box_sz_var = int(box_sz)

    Y_no_mean = noise_img_scaled - np.mean(noise_img_scaled)
    var_box = np.ones((box_sz_var, box_sz_var))
    Y_var = convolve2d(Y_no_mean ** 2, var_box, mode='same')
    Y_var[:int(box_sz), :] = -np.inf
    Y_var[:, :int(box_sz)] = -np.inf
    Y_var[-int(box_sz):, :] = -np.inf
    Y_var[:, -int(box_sz):] = -np.inf
    Y_var_sz = Y_var.shape[0]
    idxRow = np.arange(Y_var.shape[0])
    idxCol = np.arange(Y_var.shape[1])
    noise_mean = 0
    cnt_p = 0
    p_max = 1
    cnt = 1
    mean_patches = []
    noise_patches = []

    while p_max > 0:
        p_max = np.max(Y_var)
        if p_max < 0:
            break

        I = np.argmax(Y_var)
        i_row, i_col = np.unravel_index(I, Y_var.shape)

        idxRowCandidate = np.zeros(Y_var_sz, dtype=bool)
        idxColCandidate = np.zeros(Y_var_sz, dtype=bool)
        idxRowCandidate[max(i_row - rDelAlgorithm, 0):min(i_row + rDelAlgorithm, Y_var_sz)] = True
        idxColCandidate[max(i_col - rDelAlgorithm, 0):min(i_col + rDelAlgorithm, Y_var_sz)] = True

        tmp = noise_img_scaled[i_row - rDelAlgorithm:i_row + rDelAlgorithm,
              i_col - rDelAlgorithm:i_col + rDelAlgorithm]
        if cnt > num_of_patches:
            patch = noise_img_scaled[i_row - rDelAlgorithm:i_row + rDelAlgorithm,
                    i_col - rDelAlgorithm:i_col + rDelAlgorithm]
            noise_patches.append(patch - np.mean(tmp))
            mean_patches.append(np.mean(tmp))
            cnt_p += 1

        Y_var[np.ix_(idxRow[idxRowCandidate], idxCol[idxColCandidate])] = -np.inf
        cnt += 1

    noise_mean = np.mean(mean_patches)
    max_d = int(np.floor(0.8 * noise_patches[0].shape[0]))
    p = noise_patches[0].shape[0]
    noise_patches_arr = np.array(noise_patches), range(p ** 2)
    noise_patches_arr = noise_patches_arr[0]
    noise_patches_swaped  = np.transpose(noise_patches_arr, (2, 1, 0))
    samples_idx = np.array(range(p ** 2))
    radial_cov, r, cnt = cryo_epsdR(noise_patches_swaped, samples_idx, max_d)

    return radial_cov, r, noise_mean


def rotate_image(image, angle):
    """
    Rotate a 2D numpy array (image) by a specified angle.

    :param image: 2D numpy array representing the image.
    :param angle: Angle in degrees to rotate the image.
    :return: Rotated image as a 2D numpy array.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")
    if not isinstance(angle, (int, float)):
        raise ValueError("Angle must be a number.")

    rotated_image = rotate(image, angle, reshape=False)
    return rotated_image

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
def zero_out_far_values(matrix):
    matrix = matrix.copy()  # Create a writable copy
    n = matrix.shape[0]
    center = n // 2
    max_distance = n / 2

    # Create a grid of indices
    y, x = np.ogrid[:n, :n]

    # Calculate distances from the center
    distances = np.sqrt((x - center) ** 2 + (y - center) ** 2)

    # Zero out elements that are farther than max_distance
    matrix[distances > max_distance] = 0

    return matrix

def test_basis_on_rotated_image(class_averages, steerable_basis_vectors, obj_sz):
    n_img = 0
    angle_rot = 90
    img = Image(class_averages[n_img])
    scaled_img = img.downsample(obj_sz)
    scaled_img = scaled_img.asnumpy()[0]
    scaled_img_rotated = rotate_image(scaled_img, angle_rot)
    norm_rot = np.linalg.norm(np.dot(np.transpose(steerable_basis_vectors), scaled_img_rotated.flatten()))
    norm_reg = np.linalg.norm(np.dot(np.transpose(steerable_basis_vectors), scaled_img.flatten()))
    norm_rot_err = np.abs(norm_rot-norm_reg)/norm_reg
    return norm_rot_err
import numpy as np


def sort_steerable_basis(class_averages_vectors, steerable_basis_vectors,gamma,M):
    """
    Sort steerable basis vectors based on the squared coefficients computed from class averages.

    Parameters:
    class_averages_vectors : ndarray
        2D array where each column represents a flattened class average.
    steerable_basis_vectors : ndarray
        2D array of orthonormal basis vectors with the same size as class_averages_vectors.

    Returns:
    sorted_basis_vectors : ndarray
        Steerable basis vectors sorted based on the squared coefficients.
    """
    # Compute the coefficients for each class average vector
    coefficients = np.dot(steerable_basis_vectors.T, class_averages_vectors)

    # Square the coefficients to emphasize the contribution of each basis vector
    coefficients_sqr = coefficients ** 2

    # Initialize projected norm vector
    projected_norm = np.zeros(class_averages_vectors.shape[1])

    # Calculate norms of the class_averages_vectors
    class_norms = np.linalg.norm(class_averages_vectors, axis=0)

    # List to keep track of the selected vector indices
    selected_indices = []

    ### Step 1: Initial Selection
    # Find the column with the lowest class norm
    min_class_norm_idx = np.argmin(class_norms)

    # Find the index of the highest coefficient in this column
    tmp = coefficients_sqr[:, min_class_norm_idx]
    initial_vector_idx = np.argmax(tmp)

    # Append the initial vector index to the selected list
    selected_indices.append(initial_vector_idx)

    # Update projected norm vector with the contributions of the initial vector
    projected_norm += coefficients_sqr[initial_vector_idx, :]

    # Set the selected coefficients to zero to avoid re-selection
    coefficients_sqr[initial_vector_idx, :] = 0

    ### Step 2: Iterative Selection
    # Loop until the condition is met
    while len(selected_indices) < M:
        #np.min(projected_norm) < gamma * np.min(class_norms)**2
        # Find the column with the lowest projected norm
        min_proj_norm_idx = np.argmin(projected_norm)

        # Find the index of the highest coefficient in this column
        tmp = coefficients_sqr[:, min_proj_norm_idx]
        best_vector_idx = np.argmax(tmp)

        # Append the best vector index to the selected list
        selected_indices.append(best_vector_idx)

        # Update projected norm vector with the contributions of the selected vector
        projected_norm += coefficients_sqr[best_vector_idx, :]

        # Set the selected coefficients to zero to avoid re-selection
        coefficients_sqr[best_vector_idx, :] = 0

    # Create the sorted basis vectors
    sorted_basis_vectors = steerable_basis_vectors[:, selected_indices]

    return sorted_basis_vectors

def sort_steerable_basis_new(basis_full, objects, noise_patches, max_dimension): 
    
    coeff_objects = np.square((basis_full.T)@objects)
    coeff_noise = np.square((basis_full.T)@noise_patches)
    
    # first compute the full projected_snr
    projected_norm_objects = coeff_objects.sum(axis=0)
    projected_norm_noise = coeff_noise.sum(axis=0)  
    projected_snr = projected_norm_objects/projected_norm_noise.mean()
    min_idx_projected_snr =  np.argmin(projected_snr)
    # second update projected_snr at each step
    projected_snr_per_dim = []
    sorted_basis_lst = []
    projected_norm_objects = np.zeros(projected_norm_objects.shape)
    projected_norm_noise = np.zeros(projected_norm_noise.shape)
    for dim in range(max_dimension):
        numerator = coeff_objects[:,min_idx_projected_snr]+projected_norm_objects[min_idx_projected_snr] 
        denomerator = np.zeros(coeff_noise.shape[0])
        for n in range(coeff_noise.shape[1]):
            denomerator += coeff_noise[:,n] + projected_norm_noise[n]
        denomerator = denomerator/coeff_noise.shape[1]
        ratio = numerator/denomerator
        max_idx_candidate = np.argmax(ratio)
        projected_norm_objects += coeff_objects[max_idx_candidate,:]
        projected_norm_noise += coeff_noise[max_idx_candidate,:]
        sorted_basis_lst.append(basis_full[:,max_idx_candidate])
        coeff_objects = np.delete(coeff_objects,max_idx_candidate,axis=0)
        coeff_noise = np.delete(coeff_noise,max_idx_candidate,axis=0)         
        projected_snr = projected_norm_objects/projected_norm_noise.mean()
        min_idx_projected_snr =  np.argmin(projected_snr)
        projected_snr_per_dim.append(projected_snr[min_idx_projected_snr])
    sorted_basis_vectors = np.column_stack(sorted_basis_lst)
    return sorted_basis_vectors,projected_snr_per_dim


def plot_projection_ratio_with_average_and_min(templates, noise_patches, basis_vectors):
    """
    Plot the ratios of average and minimum template projection norms to the mean noise projection norm
    as a function of the number of basis vectors. Return the number of basis functions for which 
    each ratio is maximal.

    Parameters:
    templates : ndarray
        2D array where each column represents a flattened template (shape: [D, N]).
    noise_patches : ndarray
        2D array where each column represents a flattened noise patch (shape: [D, S]).
    basis_vectors : ndarray
        2D array of orthonormal basis vectors (shape: [D, M_max]).

    Returns:
    M_optimal_mean : int
        Number of basis functions for which the average ratio is maximal.
    M_optimal_min : int
        Number of basis functions for which the minimum ratio is maximal.
    """
    D, N = templates.shape
    _, S = noise_patches.shape
    M_max = basis_vectors.shape[1]

    # Initialize arrays to store results
    avg_template_norms = []
    min_template_norms = []
    mean_noise_norms = []
    ratios_mean = []
    ratios_min = []

    # Iterate over the number of basis vectors
    for m in range(1, M_max + 1):
        # Select the first `m` basis vectors
        current_basis = basis_vectors[:, :m]
        
        # Project templates and noise patches onto the current basis
        template_projections = np.dot(current_basis.T, templates)  # Shape: [m, N]
        noise_projections = np.dot(current_basis.T, noise_patches)  # Shape: [m, S]

        # Compute projection norms
        template_norms = np.sum(template_projections**2, axis=0)  # Norm for each template
        noise_norms = np.sum(noise_projections**2, axis=0)  # Norm for each noise patch

        # Compute metrics
        avg_template_norm = np.mean(template_norms)  # Average projection norm for templates
        min_template_norm = np.min(template_norms)  # Minimum projection norm for templates
        mean_noise_norm = np.mean(noise_norms)  # Mean projection norm for noise
        ratio_mean = avg_template_norm / mean_noise_norm
        ratio_min = min_template_norm / mean_noise_norm
        
        # Store results
        avg_template_norms.append(avg_template_norm)
        min_template_norms.append(min_template_norm)
        mean_noise_norms.append(mean_noise_norm)
        ratios_mean.append(ratio_mean)
        ratios_min.append(ratio_min)

    # Find the number of basis functions (M) where each ratio is maximal
    M_optimal_mean = np.argmax(ratios_mean) + 1  # Add 1 because M starts from 1
    M_optimal_min = np.argmax(ratios_min) + 1

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, M_max + 1), ratios_mean, marker='o', label='Avg Template Norm / Mean Noise Norm', color='b')
    plt.plot(range(1, M_max + 1), ratios_min, marker='x', label='Min Template Norm / Mean Noise Norm', color='g')
    plt.axhline(1, color='r', linestyle='--', label='Ratio = 1')
    plt.axvline(M_optimal_mean, color='b', linestyle='--', label=f'M_optimal_mean = {M_optimal_mean}')
    plt.axvline(M_optimal_min, color='g', linestyle='--', label=f'M_optimal_min = {M_optimal_min}')
    plt.xlabel('Number of Basis Vectors')
    plt.ylabel('Ratio')
    plt.title('Projection Norm Ratios as a Function of Basis Vectors')
    plt.legend()
    plt.grid(True)
    plt.show()

    return M_optimal_mean, M_optimal_min


def sort_steerable_basis_with_noise(class_averages_vectors, steerable_basis_vectors, noise_patches, M):
    """
    Sort steerable basis vectors based on the normalized projection norms computed from class averages and noise.

    Parameters:
    class_averages_vectors : ndarray
        2D array where each column represents a flattened class average (shape: [D, N]).
    steerable_basis_vectors : ndarray
        2D array of orthonormal basis vectors (shape: [D, M_max]).
    noise_patches : ndarray
        2D array where each column represents a flattened noise patch (shape: [D, S]).
    M : int
        Number of basis vectors to select.

    Returns:
    sorted_basis_vectors : ndarray
        Selected and sorted steerable basis vectors (shape: [D, M]).
    """
    # Compute initial coefficients for templates and noise patches
    template_coefficients = np.dot(steerable_basis_vectors.T, class_averages_vectors)# Shape: [M_max, N]
    proj = np.dot(template_coefficients, steerable_basis_vectors) 
    temp = np.sum(proj)
    noise_coefficients = np.dot(steerable_basis_vectors.T, noise_patches)  # Shape: [M_max, S]

    # Initialize cumulative norms
    cumulative_template_norms = np.sum(template_coefficients ** 2, axis=0)  # Initial template norms
    cumulative_noise_norms = np.mean(noise_coefficients ** 2, axis=1)  # Initial noise norms (mean over noise patches)

    # Initialize variables
    selected_indices = []
    D, N = class_averages_vectors.shape
    M_max = steerable_basis_vectors.shape[1]

    for step in range(M):
        # Step 1: Find the template with the smallest projection norm
        f_min_idx = np.argmin(cumulative_template_norms)

        # Step 2: Select the basis vector that maximizes the normalized projection score
        remaining_indices = list(set(range(M_max)) - set(selected_indices))
        best_vector_idx = remaining_indices[np.argmax(
            (template_coefficients[remaining_indices, f_min_idx] ** 2) /
            cumulative_noise_norms[remaining_indices]
        )]

        # Append the selected basis vector to the list
        selected_indices.append(best_vector_idx)

        # Step 3: Update norms by removing the contribution of the selected basis vector
        best_vector = steerable_basis_vectors[:, best_vector_idx]
        template_proj = np.dot(best_vector.T, class_averages_vectors)  # Projection of templates
        noise_proj = np.dot(best_vector.T, noise_patches)  # Projection of noise patches

        # Update cumulative norms
        cumulative_template_norms -= template_proj ** 2
        cumulative_noise_norms -= np.mean(noise_proj ** 2, axis=0)  # Update using mean over noise patches

        # Prevent re-selection by nullifying the coefficients of the chosen vector
        template_coefficients[best_vector_idx, :] = 0
        noise_coefficients[best_vector_idx, :] = np.inf

    # Create the sorted basis vectors
    sorted_basis_vectors = steerable_basis_vectors[:, selected_indices]

    return sorted_basis_vectors

def sort_steerable_basis_with_noise_updated(class_averages_vectors, steerable_basis_vectors, noise_patches, M):
    """
    Sort steerable basis vectors based on the normalized projection norms computed from class averages and noise.

    Parameters:
    class_averages_vectors : ndarray
        2D array where each column represents a flattened class average (shape: [D, N]).
    steerable_basis_vectors : ndarray
        2D array of orthonormal basis vectors (shape: [D, M_max]).
    noise_patches : ndarray
        2D array where each column represents a flattened noise patch (shape: [D, S]).
    M : int
        Number of basis vectors to select.

    Returns:
    sorted_basis_vectors : ndarray
        Selected and sorted steerable basis vectors (shape: [D, M]).
    """
    # Compute initial coefficients for noise patches
    noise_coefficients = np.dot(steerable_basis_vectors.T, noise_patches)  # Shape: [M_max, S]

    # Initialize variables
    D, N = class_averages_vectors.shape
    _, S = noise_patches.shape
    M_max = steerable_basis_vectors.shape[1]
    selected_indices = []

    # Compute the initial norms of each template
    template_norms = np.sum(class_averages_vectors ** 2, axis=0)  # Shape: [N]
    cumulative_template_norms = np.zeros_like(template_norms)  # Initialize cumulative template norms
    cumulative_noise_norms = np.zeros(M_max)  # Initialize cumulative noise norms for each basis vector

    for step in range(M):
        # Step 1: Find the template with the smallest norm
        f_min_idx = np.argmin(template_norms)  # Weakest template index

        # Step 3: Select the basis vector that maximizes the cumulative projected SNR
        remaining_indices = list(set(range(M_max)) - set(selected_indices))
        
        # Precompute cumulative contributions for all remaining vectors
        best_vector_idx = remaining_indices[np.argmax([
            (
                # Cumulative Template Norm
                cumulative_template_norms[f_min_idx] +
                (np.dot(steerable_basis_vectors[:, j].T, class_averages_vectors[:, f_min_idx]) ** 2)
            ) /
            (
                # Cumulative Noise Norm
                cumulative_noise_norms[j] +
                np.mean((np.dot(steerable_basis_vectors[:, j].T, noise_patches)) ** 2)
            )
            for j in remaining_indices
        ])]


        # Append the selected basis vector to the list
        selected_indices.append(best_vector_idx)

        # Step 3: Update the template norms and cumulative noise norms
        best_vector = steerable_basis_vectors[:, best_vector_idx]
        template_proj = np.dot(best_vector.T, class_averages_vectors)  # Projection of templates onto the selected vector
        noise_proj = np.dot(best_vector.T, noise_patches)  # Projection of noise patches onto the selected vector

        # Update cumulative norms
        cumulative_template_norms += template_proj ** 2
        cumulative_noise_norms[best_vector_idx] += np.mean(noise_proj ** 2)

        # Update the norms of the templates
        template_norms -= template_proj ** 2

    # Create the sorted basis vectors
    sorted_basis_vectors = steerable_basis_vectors[:, selected_indices]

    return sorted_basis_vectors






def sort_steerable_basis2(class_averages, steerable_basis_vectors, obj_sz):
    coeff_vals = np.zeros((class_averages.shape[0], steerable_basis_vectors.shape[1]))

    for n_img in range(class_averages.shape[0]):
        img = Image(class_averages[n_img])
        scaled_img = img.downsample(obj_sz)
        scaled_img = scaled_img.asnumpy()[0]
        for i in range(steerable_basis_vectors.shape[1]):
            basis_vector = steerable_basis_vectors[:, i]
            coeff = np.dot(basis_vector.T, scaled_img.flatten())
            coeff_vals[n_img, i] = coeff**2
            #norm_err[n_img, i] = np.linalg.norm(scaled_img.flatten() - projection) / np.linalg.norm(scaled_img.flatten())

    max_coeff_vals = np.max(coeff_vals, axis=0)
    sorted_indices = np.argsort(max_coeff_vals)[::-1]
    sorted_steerable_basis_vectors = steerable_basis_vectors[:, sorted_indices]
    return sorted_steerable_basis_vectors

def remove_low_variance_elements(data, percentage=10):
    """
    Remove a specified percentage of elements with the lowest variance after flattening.

    Parameters:
    data (ndarray): The input array of shape (num_images, height, width).
    percentage (float): The percentage of elements to remove, based on variance.

    Returns:
    ndarray: The filtered array with low-variance elements removed.
    """
    # Flatten the images along the spatial dimensions
    flattened_data = data.reshape(data.shape[0], -1)

    # Calculate the variance for each flattened element
    variances = np.var(flattened_data, axis=1)

    # Determine the cutoff for the lowest percentage
    cutoff = np.percentile(variances, percentage)

    # Filter out elements with variance below the cutoff
    high_variance_indices = np.where(variances > cutoff)[0]

    # Return the filtered data
    filtered_data = data[high_variance_indices]
    return filtered_data, high_variance_indices



    



def dominent_steerable_basis(class_averages, sorted_steerable_basis_vectors, obj_sz):
    norm_err = np.zeros((class_averages.shape[0], sorted_steerable_basis_vectors.shape[1]))

    for n_img in range(class_averages.shape[0]):
        img = Image(class_averages[n_img])
        scaled_img = img.downsample(obj_sz)
        scaled_img = scaled_img.asnumpy()[0]
        for i in range(sorted_steerable_basis_vectors.shape[1]):
            basis_vectors = sorted_steerable_basis_vectors[:, :i+1]
            projection = np.dot(basis_vectors, np.dot(basis_vectors.T, scaled_img.flatten()))
            norm_err[n_img, i] = np.linalg.norm(scaled_img.flatten() - projection) / np.linalg.norm(scaled_img.flatten())

    max_norm_err = np.max(norm_err, axis=0)
    return max_norm_err
def test_basis_on_images(class_averages, steerable_basis_vectors, obj_sz):
    norm_err = np.zeros(class_averages.shape[0])
    norm_err_span = np.zeros(class_averages.shape[0])
    for n_img in range(class_averages.shape[0]):
        img = Image(class_averages[n_img])
        scaled_img = img.downsample(obj_sz)
        scaled_img = scaled_img.asnumpy()[0]
        norm_err[n_img] = np.linalg.norm(np.dot(steerable_basis_vectors.T, scaled_img.flatten()))/np.linalg.norm(scaled_img.flatten())
        norm_err_span[n_img] = np.linalg.norm(scaled_img.flatten()-np.dot(steerable_basis_vectors,np.dot(steerable_basis_vectors.T, scaled_img.flatten()))) / np.linalg.norm(
            scaled_img.flatten())
    return norm_err,norm_err_span

def test_basis_on_images_keren(class_averages, steerable_basis_vectors, obj_sz):
    norm_err = np.zeros(class_averages.shape[0])
    norm_err_span = np.zeros(class_averages.shape[0])
    for n_img in range(class_averages.shape[0]):
        norm_err[n_img] = np.linalg.norm(np.dot(steerable_basis_vectors.T, class_averages[n_img].flatten()))/np.linalg.norm(class_averages[n_img].flatten())
        norm_err_span[n_img] = np.linalg.norm(class_averages[n_img].flatten()-np.dot(steerable_basis_vectors,np.dot(steerable_basis_vectors.T, class_averages[n_img].flatten()))) / np.linalg.norm(
            class_averages[n_img].flatten())
    return norm_err,norm_err_span


def plot_class_averages(class_averages):
    # Determine the number of images
    num_images = class_averages.shape[0]

    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    # Loop through each image and display it
    for i in range(num_images):
        ax = axes[i]
        ax.imshow(class_averages[i], cmap='gray')
        ax.axis('off')
        ax.set_title(f'Image {i + 1}')

    # Show the plot
    plt.show()


def compute_the_steerable_images(objects,obj_sz,fb_basis,eigen_vectors_per_ang_lst,mean_noise_per_ang_lst):
    # compute the radial Fourier component to each class average
    # get the coefficients of the class averages in the FB basis
    steerable_euclidian_l = np.zeros((obj_sz, obj_sz, 1+2*(fb_basis.ell_max), objects.shape[0]))
    denoised_images = np.zeros(objects.shape)
    for n_img in range(objects.shape[0]):
        img = Image(objects[n_img].astype(np.float32))
        img_fb_coeff = fb_basis.evaluate_t(img)
        img_fb_coeff_denoise = img_fb_coeff.copy()
        v_0 = img_fb_coeff.copy()
        v_0._data[0,fb_basis.k_max[0]:] = 0
        # denoise the image by removing the noise component
        Q = eigen_vectors_per_ang_lst[0]
        v_0._data[0, :fb_basis.k_max[0]] =v_0._data[0, :fb_basis.k_max[0]] - mean_noise_per_ang_lst[0]
        v_0._data[0, :fb_basis.k_max[0]] = v_0._data[0, :fb_basis.k_max[0]]-Q@(Q.T@v_0._data[0, :fb_basis.k_max[0]])
        img_fb_coeff_denoise._data[0, :fb_basis.k_max[0]] = v_0._data[0, :fb_basis.k_max[0]]
        v_0_img = fb_basis.evaluate(v_0).asnumpy()[0]
        steerable_euclidian_l[:,:,0,n_img] = fb_basis.evaluate(v_0)
        coeff_k_index_start = fb_basis.k_max[0]
        for m in range(1, fb_basis.ell_max+1):
            l_idx = 2*m-1
            k_idx = fb_basis.k_max[m]
            coeff_k_index_end_cos = coeff_k_index_start + k_idx
            coeff_k_index_end_sin = coeff_k_index_end_cos + k_idx
            vcos = img_fb_coeff.copy()
            vcos._data[0,:coeff_k_index_start]=0
            vcos._data[0,coeff_k_index_end_cos:] = 0
            # denoise the image by removing the noise component
            vcos._data[0, coeff_k_index_start:coeff_k_index_end_cos] = vcos._data[0,coeff_k_index_start:coeff_k_index_end_cos] - mean_noise_per_ang_lst[l_idx]
            Q = eigen_vectors_per_ang_lst[l_idx]
            vcos._data[0, coeff_k_index_start:coeff_k_index_end_cos] = vcos._data[0, coeff_k_index_start:coeff_k_index_end_cos] - Q@(Q.T@vcos._data[0, coeff_k_index_start:coeff_k_index_end_cos])
            img_fb_coeff_denoise._data[0, coeff_k_index_start:coeff_k_index_end_cos] = vcos._data[0, coeff_k_index_start:coeff_k_index_end_cos]
            vsin = img_fb_coeff.copy()
            vsin._data[0,:coeff_k_index_end_cos]=0
            vsin._data[0,coeff_k_index_end_sin:] = 0
            # denoise the image by removing the noise component
            vsin._data[0, coeff_k_index_end_cos:coeff_k_index_end_sin] = vsin._data[0, coeff_k_index_end_cos:coeff_k_index_end_sin] - mean_noise_per_ang_lst[l_idx+1]
            Q = eigen_vectors_per_ang_lst[l_idx+1]
            vsin._data[0, coeff_k_index_end_cos:coeff_k_index_end_sin] = vsin._data[0, coeff_k_index_end_cos:coeff_k_index_end_sin] - Q@(Q.T@vsin._data[0, coeff_k_index_end_cos:coeff_k_index_end_sin])
            img_fb_coeff_denoise._data[0, coeff_k_index_end_cos:coeff_k_index_end_sin] = vsin._data[0, coeff_k_index_end_cos:coeff_k_index_end_sin]

            vcos_img = fb_basis.evaluate(vcos).asnumpy()[0]
            vsin_img = fb_basis.evaluate(vsin).asnumpy()[0]
            steerable_euclidian_l[:,:,l_idx,n_img] = vcos_img
            steerable_euclidian_l[:, :, l_idx+1, n_img] = vsin_img
            #steerable_euclidian_l[:,:,l_idx,n_img] = vcos_img + 1j*vsin_img
            #steerable_euclidian_l[:, :, l_idx+1, n_img] = vcos_img - 1j * vsin_img
            coeff_k_index_start = coeff_k_index_end_sin
        denoised_images[n_img,:,:] = fb_basis.evaluate(img_fb_coeff_denoise).asnumpy()[0]
    return steerable_euclidian_l,denoised_images



def compute_the_steerable_images_normalization(class_averages, obj_sz, l_max, norm_err):
    # Initialize the FB basis
    fb_basis = FBBasis2D(size=(obj_sz, obj_sz), ell_max=l_max)

    # Initialize the steerable output array
    steerable_euclidian_l = np.zeros((obj_sz, obj_sz, 1 + 2 * (fb_basis.ell_max), class_averages.shape[0]))

    for n_img in range(class_averages.shape[0]):
        # Preprocess the image: downsample and convert to the required format
        img = Image(class_averages[n_img])
        scaled_img = np.asarray(img.downsample(obj_sz)).astype(np.float32)

        # Compute FB coefficients for the image
        img_fb_coeff = fb_basis.evaluate_t(scaled_img)

        # Normalize the FB coefficients to preserve energy
        coeff_energy = np.sum(np.abs(img_fb_coeff._data)**2)
        img_energy = np.linalg.norm(scaled_img)**2
        if coeff_energy > 0:  # Avoid division by zero
            scaling_factor = np.sqrt(img_energy / coeff_energy)
            img_fb_coeff._data *= scaling_factor

        # Extract the zeroth angular component
        v_0 = img_fb_coeff.copy()
        v_0._data[0, fb_basis.k_max[0]:] = 0
        v_0_img = fb_basis.evaluate(v_0).asnumpy()[0]
        steerable_euclidian_l[:, :, 0, n_img] = v_0_img

        coeff_k_index_start = fb_basis.k_max[0]
        for m in range(1, fb_basis.ell_max + 1):
            l_idx = 2 * m - 1
            k_idx = fb_basis.k_max[m]
            coeff_k_index_end_cos = coeff_k_index_start + k_idx
            coeff_k_index_end_sin = coeff_k_index_end_cos + k_idx

            # Cosine component
            vcos = img_fb_coeff.copy()
            vcos._data[0, :coeff_k_index_start] = 0
            vcos._data[0, coeff_k_index_end_cos:] = 0
            vcos_img = fb_basis.evaluate(vcos).asnumpy()[0]
            steerable_euclidian_l[:, :, l_idx, n_img] = vcos_img

            # Sine component
            vsin = img_fb_coeff.copy()
            vsin._data[0, :coeff_k_index_end_cos] = 0
            vsin._data[0, coeff_k_index_end_sin:] = 0
            vsin_img = fb_basis.evaluate(vsin).asnumpy()[0]
            steerable_euclidian_l[:, :, l_idx + 1, n_img] = vsin_img

            coeff_k_index_start = coeff_k_index_end_sin

    return steerable_euclidian_l

def compute_num_components(class_averages, steerable_euclidian_l, norm_err, obj_sz):
    number_of_components = np.zeros(steerable_euclidian_l.shape[3])
    for n_img in range(steerable_euclidian_l.shape[3]):
        img = Image(class_averages[n_img])
        scaled_img = img.downsample(obj_sz)
        scaled_img = scaled_img.asnumpy()
        scaled_img = scaled_img[0]
        m = 0
        norm_est = 1
        max_m = (steerable_euclidian_l.shape[2] - 1) // 2
        while norm_est > norm_err and m <= max_m:
            if m == 0:
                euc_est = steerable_euclidian_l[:,:,0,n_img]
            else:
                euc_est = euc_est + steerable_euclidian_l[:,:,2 * m - 1,n_img]+steerable_euclidian_l[:,:,2 * m ,n_img]
            norm_est = np.linalg.norm(zero_out_far_values(scaled_img) - zero_out_far_values(euc_est), ord='fro') / np.linalg.norm(zero_out_far_values(scaled_img), ord='fro')
            m = m + 1
        number_of_components[n_img] = m-1
    return number_of_components


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

def compute_the_steerable_basis_with_noise(steerable_euclidian_l,steerable_euclidian_l_noise):
    # Initialize an empty list to accumulate Q matrices
    steerable_basis_vectors_list = []

    for l in range(steerable_euclidian_l.shape[2]):
        l_images_objects = steerable_euclidian_l[:, :, l, :]
        l_images_objects_denoise = np.zeros(l_images_objects.shape)
        l_images_noise = steerable_euclidian_l_noise[:, :, l, :]
        mean_noise_l = np.mean(l_images_noise,axis=2)
        normelized_mean_noise_l = mean_noise_l/np.linalg.norm(mean_noise_l) 
        for n in range(l_images_objects.shape[2]):
            l_images_objects_denoise[:,:,n] = l_images_objects[:,:,n] - np.sum(l_images_objects[:,:,n]*normelized_mean_noise_l)*normelized_mean_noise_l
        Q = qr_factorization_matrices(l_images_objects_denoise)
        steerable_basis_vectors_list.append(Q)  # Add each Q matrix to the list

    # Concatenate all Q matrices horizontally at the end
    steerable_basis_vectors = np.hstack(steerable_basis_vectors_list)

    return steerable_basis_vectors

def compute_the_steerable_basis_1(steerable_euclidian_l):
    # Initialize an empty list to accumulate Q matrices
    steerable_basis_vectors_list = []

    for l in range(steerable_euclidian_l.shape[2]):
        l_images = steerable_euclidian_l[:, :, l, :]
        Q = qr_factorization_matrices(l_images)
        steerable_basis_vectors_list.append(Q)  # Add each Q matrix to the list

    # Concatenate all Q matrices horizontally at the end
    concatenated_basis = np.hstack(steerable_basis_vectors_list)
    steerable_basis_vectors, _ = np.linalg.qr(concatenated_basis)
    
    return steerable_basis_vectors


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
    
def is_patch_contaminated(contamination_mask, x, y, patch_size):
    """
    Check if the patch centered at (x, y) intersects with contamination.
    
    Parameters:
    contamination_mask : Downsampled contamination mask.
    x, y               : Center coordinates of the patch.
    patch_size         : Size of the patch.
    
    Returns:
    contaminated : Boolean, True if patch intersects contamination.
    """
    half_patch = patch_size // 2
    row_start = max(0, y - half_patch)
    row_end = min(contamination_mask.shape[0], y + half_patch)
    col_start = max(0, x - half_patch)
    col_end = min(contamination_mask.shape[1], x + half_patch)
    
    # Check if any pixel in this region is contaminated
    return np.any(contamination_mask[row_start:row_end, col_start:col_end] == 1)

    ### Image and Patch Processing Functions ###

    def adjust_contrast(image, contrast_factor=1.0, brightness_factor=0.0):
        # Adjust contrast and brightness
        adjusted_image = contrast_factor * image + brightness_factor
        adjusted_image = np.clip(adjusted_image, 0, 1)  # Clip values to [0, 1]
        return adjusted_image




def extract_objects_from_coordinates(Y, coordinates, patch_size):
    """
    Extract object patches from the image based on coordinates.

    Parameters:
    Y            : The input image (2D array).
    coordinates  : List of (x, y) coordinates.
    patch_size   : The size of the square patch to extract.

    Returns:
    patches : Array of extracted patches.
    """
    half_patch = patch_size // 2
    patches = []

    for (x, y) in coordinates:
        # Define patch boundaries
        row_start = max(0, y - half_patch)
        row_end = min(Y.shape[0], y + half_patch)
        col_start = max(0, x - half_patch)
        col_end = min(Y.shape[1], x + half_patch)

        # Extract the patch
        patch = Y[row_start:row_end, col_start:col_end]
        if patch.shape == (patch_size, patch_size):
            patches.append(patch)

    return np.array(patches)




def noise_reduction_wiener(image, mysize=5, noise=None):
    """
    Apply Wiener filtering to reduce noise in a grayscale cryo-EM micrograph.

    Parameters:
    - image: 2D numpy array representing the grayscale cryo-EM micrograph.
    - mysize: Size of the filter window (larger values result in more smoothing).
    - noise: Estimate of the noise power; if None, the noise is estimated from the image.

    Returns:
    - denoised_image: 2D numpy array of the denoised image.
    """
    # Apply the Wiener filter
    denoised_image = wiener(image, mysize=mysize, noise=noise)

    return denoised_image


def noise_reduction(image, sigma=1.0):
    """
    Apply Gaussian filtering to reduce noise in a cryo-EM image.

    Parameters:
    - image: 2D numpy array representing the cryo-EM micrograph.
    - sigma: Standard deviation for Gaussian kernel, controlling the amount of smoothing.
             Higher sigma = more smoothing.

    Returns:
    - denoised_image: 2D numpy array of the denoised image.
    """
    # Apply Gaussian filter for noise reduction
    denoised_image = gaussian_filter(image, sigma=sigma)

    return denoised_image


def contrast_stretch(image):
    """
    Apply contrast stretching to the input micrograph image.

    Parameters:
    - image: 2D numpy array representing the micrograph.

    Returns:
    - stretched_image: 2D numpy array of the contrast-stretched image.
    """
    # Get the minimum and maximum pixel values in the image
    min_val = np.min(image)
    max_val = np.max(image)

    # Stretch the image to the full range [0, 1]
    stretched_image = (image - min_val) / (max_val - min_val)

    return stretched_image


def extract_object_patches_and_return(object_img_scaled, box_sz, num_of_patches=10):
    """
    Extract object patches of size box_sz x box_sz from a micrograph, stopping after num_of_patches (default 10).
    """
    rDelAlgorithm = (int(box_sz) + 1)
    box_sz_var = int(box_sz / 3)

    Y_no_mean = object_img_scaled - np.mean(object_img_scaled)
    var_box = np.ones((box_sz_var, box_sz_var))
    Y_var = convolve2d(Y_no_mean ** 2, var_box, mode='same')

    # Exclude boundaries
    Y_var[:int(box_sz), :] = -np.inf
    Y_var[:, :int(box_sz)] = -np.inf
    Y_var[-int(box_sz):, :] = -np.inf
    Y_var[:, -int(box_sz):] = -np.inf

    object_patches = []
    cnt_p = 0
    p_max = 1

    while p_max > 0 and cnt_p < num_of_patches:
        p_max = np.max(Y_var)
        if p_max < 0:
            break
        I = np.argmax(Y_var)
        i_row, i_col = np.unravel_index(I, Y_var.shape)
        idxRowCandidate = np.zeros(Y_var.shape[0], dtype=bool)
        idxColCandidate = np.zeros(Y_var.shape[1], dtype=bool)
        idxRowCandidate[max(i_row - rDelAlgorithm, 0):min(i_row + rDelAlgorithm, Y_var.shape[0])] = True
        idxColCandidate[max(i_col - rDelAlgorithm, 0):min(i_col + rDelAlgorithm, Y_var.shape[1])] = True
        # patch = object_img_scaled[i_row:i_row + box_sz, i_col:i_col + box_sz]
        # patch = object_img_scaled[i_row:i_row + box_sz, i_col:i_col + box_sz]
        patch = object_img_scaled[i_row - box_sz // 2:i_row + box_sz // 2, i_col - box_sz // 2:i_col + box_sz // 2]
        # Check if patch shape is valid and ensure it doesn't overlap with contamination
        if patch.shape == (box_sz, box_sz) and not is_patch_contaminated(contamination_mask, i_col, i_row, box_sz):
            noise_patches.append(patch)
            cnt_p += 1
        Y_var[np.ix_(idxRowCandidate, idxColCandidate)] = -np.inf

    return np.array(object_patches)


def extract_noise_patches_and_return_min_variance(noise_img_scaled,contamination_mask, box_sz, num_of_patches=30):
    """
    Extract noise patches of size box_sz x box_sz from a micrograph, starting from the patches with the lowest variance.
    """
    # rDelAlgorithm = (int(box_sz) + 1)
    rDelAlgorithm = int(box_sz / 2)
    box_sz_var = int(box_sz / 4)
    Y_no_mean = noise_img_scaled - np.mean(noise_img_scaled)
    var_box = np.ones((box_sz_var, box_sz_var))
    Y_var = convolve2d(Y_no_mean ** 2, var_box, mode='same')

    # Exclude boundaries
    Y_var[:int(box_sz_var), :] = np.inf
    Y_var[:, :int(box_sz_var)] = np.inf
    Y_var[-int(box_sz_var):, :] = np.inf
    Y_var[:, -int(box_sz_var):] = np.inf

    noise_patches = []
    cnt_p = 0
    p_min = 0

    while p_min < np.inf and cnt_p < num_of_patches:
        p_min = np.min(Y_var)  # Get the minimum variance now
        if p_min == np.inf:
            break
        I = np.argmin(Y_var)  # Index of the minimum variance value
        i_row, i_col = np.unravel_index(I, Y_var.shape)
        idxRowCandidate = np.zeros(Y_var.shape[0], dtype=bool)
        idxColCandidate = np.zeros(Y_var.shape[1], dtype=bool)
        idxRowCandidate[max(i_row - rDelAlgorithm, 0):min(i_row + rDelAlgorithm, Y_var.shape[0])] = True
        idxColCandidate[max(i_col - rDelAlgorithm, 0):min(i_col + rDelAlgorithm, Y_var.shape[1])] = True
        # patch = noise_img_scaled[i_row:i_row + box_sz, i_col:i_col + box_sz]
        if np.mod(box_sz, 2) == 1:
            patch = noise_img_scaled[i_row - 1 - box_sz // 2:i_row + box_sz // 2,
                    i_col - 1 - box_sz // 2:i_col + box_sz // 2]
        else:
            patch = noise_img_scaled[i_row - box_sz // 2:i_row + box_sz // 2, i_col - box_sz // 2:i_col + box_sz // 2]
        if patch.shape == (box_sz, box_sz):
            noise_patches.append(patch)
            cnt_p += 1
        # Set the selected area to np.inf so it won't be selected again
        Y_var[np.ix_(idxRowCandidate, idxColCandidate)] = np.inf

    return np.array(noise_patches)


# from scipy.signal import convolve2d
# import numpy as np

# import numpy as np
# from scipy.signal import convolve2d
# from joblib import Parallel, delayed





def compute_patch_variance(padded_img, i, j, box_sz):
    """
    Compute variance of a patch centered at (i, j).

    Parameters:
    padded_img : np.ndarray : Padded micrograph image.
    i, j       : int        : Center coordinates of the patch in the padded image.
    box_sz     : int        : Size of the patch.

    Returns:
    patch_variance : float : Variance of the patch.
    """
    half_box_sz = box_sz // 2
    patch = padded_img[i - half_box_sz:i + half_box_sz, j - half_box_sz:j + half_box_sz]
    patch_mean = np.mean(patch)
    patch_variance = np.mean((patch - patch_mean) ** 2)
    return patch_variance



def extract_noise_patches_and_coor_return_min_variance(noise_img_scaled, contamination_mask, box_sz, num_of_patches=30):
    """
    Extract noise patches of size box_sz x box_sz from a micrograph, starting from the patches with the lowest variance.

    Returns:
    noise_patches    : ndarray : Array of extracted noise patches.
    patch_coordinates : list   : List of (row, col) center coordinates for the patches.
    """

    # Define half_box_sz depending on whether box_sz is even or odd
    half_box_sz = box_sz // 2 if box_sz % 2 == 0 else (box_sz // 2) + 1

    # Create an averaging kernel (box filter)
    kernel = np.ones((box_sz, box_sz), dtype=np.float32) / (box_sz * box_sz)
    kernel_torch = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, box_sz, box_sz)

    # Convert image to PyTorch tensor
    noise_img_scaled_torch = torch.tensor(noise_img_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Perform convolution for mean
    mean_img_torch = F.conv2d(noise_img_scaled_torch, kernel_torch, stride=1, padding='same')

    # Perform convolution for mean of squared image (for variance calculation)
    mean_squared_img_torch = F.conv2d(noise_img_scaled_torch ** 2, kernel_torch, stride=1, padding='same')

    # Convert results back to NumPy
    mean_img_np = mean_img_torch.squeeze().numpy()  # Remove extra dimensions
    mean_squared_img_np = mean_squared_img_torch.squeeze().numpy()
    
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
            patch = noise_img_scaled[
                max(0, i_row - half_box_sz): min(i_row + half_box_sz, noise_img_scaled.shape[0]),
                max(0, i_col - half_box_sz): min(i_col + half_box_sz, noise_img_scaled.shape[1])
            ]
        else:
            patch = noise_img_scaled[
                max(0, i_row - half_box_sz): min(i_row + half_box_sz - 1, noise_img_scaled.shape[0]),
                max(0, i_col - half_box_sz): min(i_col + half_box_sz - 1, noise_img_scaled.shape[1])
            ]

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


import numpy as np
from scipy.ndimage import convolve


def extract_noise_patches_and_coor_return_min_variance_numpy(noise_img_scaled, contamination_mask, box_sz,
                                                             num_of_patches=30):
    """
    Extract noise patches of size box_sz x box_sz from a micrograph, starting from the patches with the lowest variance.

    Returns:
    noise_patches    : ndarray : Array of extracted noise patches.
    patch_coordinates : list   : List of (row, col) center coordinates for the patches.
    """

    # Define half_box_sz depending on whether box_sz is even or odd
    half_box_sz = box_sz // 2 if box_sz % 2 == 0 else (box_sz // 2) + 1

    # Create an averaging kernel (box filter)
    kernel = np.ones((box_sz, box_sz), dtype=np.float32) / (box_sz * box_sz)

    # Perform convolution for mean using SciPy's convolve (memory efficient)
    mean_img = convolve(noise_img_scaled, kernel, mode='constant', cval=0.0)

    # Perform convolution for mean of squared image (for variance calculation)
    mean_squared_img = convolve(noise_img_scaled ** 2, kernel, mode='constant', cval=0.0)

    # Variance formula: E[x^2] - (E[x])^2
    Y_var = mean_squared_img - mean_img ** 2

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
            patch = noise_img_scaled[
                    max(0, i_row - half_box_sz): min(i_row + half_box_sz, noise_img_scaled.shape[0]),
                    max(0, i_col - half_box_sz): min(i_col + half_box_sz, noise_img_scaled.shape[1])
                    ]
        else:
            patch = noise_img_scaled[
                    max(0, i_row - half_box_sz): min(i_row + half_box_sz - 1, noise_img_scaled.shape[0]),
                    max(0, i_col - half_box_sz): min(i_col + half_box_sz - 1, noise_img_scaled.shape[1])
                    ]

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
def extract_noise_patches_and_coor_return_min_variance_torch(noise_img_scaled, contamination_mask, box_sz,
                                                             num_of_patches=30):
    """
    Extract noise patches of size box_sz x box_sz from a micrograph, starting from the patches with the lowest variance.

    Returns:
    noise_patches    : ndarray : Array of extracted noise patches.
    patch_coordinates : list   : List of (row, col) center coordinates for the patches.
    """

    # Use GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define half_box_sz depending on whether box_sz is even or odd
    half_box_sz = box_sz // 2 if box_sz % 2 == 0 else (box_sz // 2) + 1

    # Create an averaging kernel (box filter) using float32 to reduce memory usage
    kernel = np.ones((box_sz, box_sz), dtype=np.float32) / (box_sz * box_sz)
    kernel_torch = torch.tensor(kernel, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    # Convert image to PyTorch tensor and move to device
    noise_img_scaled_torch = torch.tensor(noise_img_scaled, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(
        0)

    # Perform convolution for mean and mean of squared image
    mean_img_torch = F.conv2d(noise_img_scaled_torch, kernel_torch, stride=1, padding='same')
    mean_squared_img_torch = F.conv2d(noise_img_scaled_torch ** 2, kernel_torch, stride=1, padding='same')

    # Clear memory for intermediate tensors after use
    del kernel_torch
    del noise_img_scaled_torch
    torch.cuda.empty_cache()
    gc.collect()

    # Calculate variance using in-place operations
    Y_var_torch = mean_squared_img_torch - mean_img_torch.pow(2)

    # Clear intermediate tensors to free memory
    del mean_img_torch
    del mean_squared_img_torch
    torch.cuda.empty_cache()
    gc.collect()

    # Move to CPU and convert to NumPy
    Y_var = Y_var_torch.squeeze().cpu().numpy()

    # Clear the remaining GPU tensors
    del Y_var_torch
    torch.cuda.empty_cache()
    gc.collect()

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

        # Index of the minimum variance value
        i_row, i_col = np.unravel_index(np.argmin(Y_var), Y_var.shape)

        # Extract patch, ensuring it remains box_sz x box_sz
        row_start = max(0, i_row - half_box_sz)
        row_end = min(i_row + half_box_sz, noise_img_scaled.shape[0])
        col_start = max(0, i_col - half_box_sz)
        col_end = min(i_col + half_box_sz, noise_img_scaled.shape[1])

        patch = noise_img_scaled[row_start:row_end, col_start:col_end]

        # Skip invalid or out-of-bound patches
        if patch.shape != (box_sz, box_sz):
            Y_var[i_row, i_col] = np.inf  # Mask the invalid location
            continue

        # Append patch and coordinates
        noise_patches.append(patch)
        patch_coordinates.append((i_row, i_col))
        cnt_p += 1

        # Mask the region to avoid overlap
        row_start = max(0, i_row - box_sz)
        row_end = min(Y_var.shape[0], i_row + box_sz + 1)
        col_start = max(0, i_col - box_sz)
        col_end = min(i_col + box_sz, Y_var.shape[1])
        Y_var[row_start:row_end, col_start:col_end] = np.inf  # Set the area to infinity

    # Convert noise patches to NumPy array
    noise_patches = np.array(noise_patches)

    # Clear cache one last time before returning results
    torch.cuda.empty_cache()
    gc.collect()

    return noise_patches, patch_coordinates

def extract_noise_patches_and_coor_return_min_variance_tf(noise_img_scaled, contamination_mask, box_sz ,num_of_patches=30):

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

    x =1
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
            patch = noise_img_scaled[
                max(0, i_row - half_box_sz): min(i_row + half_box_sz, noise_img_scaled.shape[0]),
                max(0, i_col - half_box_sz): min(i_col + half_box_sz, noise_img_scaled.shape[1])
            ]
        else:
            patch = noise_img_scaled[
                max(0, i_row - half_box_sz): min(i_row + half_box_sz - 1, noise_img_scaled.shape[0]),
                max(0, i_col - half_box_sz): min(i_col + half_box_sz - 1, noise_img_scaled.shape[1])
            ]

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

def extract_smaller_patches_with_coords(larger_patches, large_patch_coords, small_patch_size, num_smaller_patches):
    """
    Extract smaller patches with minimal variance and return their coordinates.

    Parameters:
    larger_patches      : ndarray : Array of larger noise patches [N, H, W].
    large_patch_coords  : list    : List of (row, col) coordinates of larger patch centers.
    small_patch_size    : int     : Desired size of smaller patches.
    num_smaller_patches : int     : Number of smaller patches to extract.

    Returns:
    smaller_patches : list : Extracted smaller patches.
    small_coords    : list : Coordinates of the smaller patches relative to the original micrograph.
    """
    half_small_sz = small_patch_size // 2
    smaller_patches = []
    small_coords = []

    # Compute variance for larger patches and sort
    larger_patch_variances = [np.var(patch) for patch in larger_patches]
    sorted_indices = np.argsort(larger_patch_variances)
    sorted_larger_patches = [larger_patches[i] for i in sorted_indices]
    sorted_large_coords = [large_patch_coords[i] for i in sorted_indices]

    # Debugging: Check the sorting of patches
    print(f"Number of larger patches: {len(larger_patches)}")
    print(f"Number of larger patch coordinates: {len(large_patch_coords)}")
    print(f"Computed variances: {larger_patch_variances}")
    print(f"Sorted variances: {[larger_patch_variances[i] for i in sorted_indices]}")

    # Extract smaller patches
    for patch_idx, (patch, (center_row, center_col)) in enumerate(zip(sorted_larger_patches, sorted_large_coords)):
        # Debugging: Check the current larger patch
        print(f"\nProcessing larger patch {patch_idx + 1}/{len(sorted_larger_patches)}")
        print(f"Center: ({center_row}, {center_col}), Patch shape: {patch.shape}")

        # Handle invalid patches
        if not np.isfinite(patch).all():
            print(f"Skipping larger patch {patch_idx + 1} due to invalid values.")
            continue

        # Create a variance map for the current larger patch
        var_map = np.zeros_like(patch)
        padded_patch = np.pad(patch, half_small_sz, mode='constant', constant_values=np.inf)

        for i in range(half_small_sz, patch.shape[0] + half_small_sz):
            for j in range(half_small_sz, patch.shape[1] + half_small_sz):
                small_patch = padded_patch[i - half_small_sz:i + half_small_sz,
                                           j - half_small_sz:j + half_small_sz]
                # Skip invalid patches
                if not np.isfinite(small_patch).all():
                    var_map[i - half_small_sz, j - half_small_sz] = np.inf
                    continue

                small_patch_mean = np.mean(small_patch)
                small_patch_variance = np.mean((small_patch - small_patch_mean) ** 2)
                var_map[i - half_small_sz, j - half_small_sz] = small_patch_variance

        # Debugging: Check variance map statistics
        print(f"Variance map min: {np.nanmin(var_map)}, max: {np.nanmax(var_map)}")

        # Extract the smallest variance smaller patches
        indices = np.argsort(var_map.flatten())[:num_smaller_patches]
        for idx in indices:
            i, j = np.unravel_index(idx, var_map.shape)
            small_patch = patch[i - half_small_sz:i + half_small_sz,
                                j - half_small_sz:j + half_small_sz]
            if small_patch.shape == (small_patch_size, small_patch_size):
                # Adjust coordinates to the original micrograph
                orig_row = center_row - patch.shape[0] // 2 + i
                orig_col = center_col - patch.shape[1] // 2 + j
                smaller_patches.append(small_patch)
                small_coords.append((orig_row, orig_col))

                # Debugging: Check extracted patch
                print(f"Extracted smaller patch at ({orig_row}, {orig_col}) with variance: {np.var(small_patch)}")

                if len(smaller_patches) >= num_smaller_patches:
                    print("Reached desired number of smaller patches.")
                    break

        # Stop once the desired number of smaller patches is reached
        if len(smaller_patches) >= num_smaller_patches:
            break

    # Debugging: Final results
    print(f"\nTotal smaller patches extracted: {len(smaller_patches)}")
    return np.array(smaller_patches), small_coords



def extract_high_variance_patches_with_skip(noise_img_scaled, box_sz, obj_sz_down_scaled, delta, num_of_patches=30):
    """
    Extract noise patches of size box_sz x box_sz from an image by skipping the top N patches with the highest variance,
    and then selecting the next num_of_patches patches with the highest variance.

    Parameters:
    - noise_img_scaled (2D NumPy array): Input noise image.
    - box_sz (int): Size of each square patch (box_sz x box_sz).
    - obj_sz_down_scaled (int): Downscaled object size, used to calculate deletion region.
    - delta (int): Additional parameter to adjust the size of the deletion region.
    - num_of_patches (int): Number of patches to extract after skipping (default: 30).

    Returns:
    - noise_patches (3D NumPy array): Extracted patches with shape (num_of_patches, box_sz, box_sz).
    """

    # Step 1: Calculate the side length of the deletion region
    sideLengthAlgorithmL = 2 * obj_sz_down_scaled + delta
    rDelAlgorithmL = sideLengthAlgorithmL // 2  # Radius for deletion

    print(f"Side Length Algorithm L: {sideLengthAlgorithmL}")
    print(f"Deletion Radius (rDelAlgorithmL): {rDelAlgorithmL}")

    # Step 2: Calculate the number of patches to skip
    N_skip = (noise_img_scaled.shape[0] // (sideLengthAlgorithmL)) * (
                noise_img_scaled.shape[1] // (sideLengthAlgorithmL))
    print(f"Number of patches to skip (N_skip): {N_skip}")

    # Step 3: Center the image by removing the mean
    Y_no_mean = noise_img_scaled - np.mean(noise_img_scaled)

    # Step 4: Compute the local variance map using convolution
    var_box = np.ones((box_sz, box_sz))
    Y_var = convolve2d(Y_no_mean ** 2, var_box, mode='same')

    # Step 5: Exclude boundary regions by setting their variance to -inf to prevent selection
    Y_var[:rDelAlgorithmL, :] = -np.inf
    Y_var[:, :rDelAlgorithmL] = -np.inf
    Y_var[-rDelAlgorithmL:, :] = -np.inf
    Y_var[:, -rDelAlgorithmL:] = -np.inf

    # Step 6: Initialize lists to store patches and counters
    noise_patches = []
    cnt_p = 0  # Counter for extracted patches
    cnt_skip = 0  # Counter for skipped patches

    # Step 7: Start the first phase - Skipping the top N_skip patches with highest variance
    print("Starting the skipping phase...")
    while cnt_skip < N_skip:
        p_max = np.max(Y_var)
        if p_max == -np.inf:
            print("No more valid patches available during skipping phase.")
            break

        # Find the index of the patch with the highest variance
        I = np.argmax(Y_var)
        i_row, i_col = np.unravel_index(I, Y_var.shape)

        # Define the boundaries of the patch
        half_box = box_sz // 2
        if box_sz % 2 == 1:
            # For odd box sizes, ensure the patch is centered correctly
            row_start = i_row - half_box
            row_end = i_row + half_box + 1
            col_start = i_col - half_box
            col_end = i_col + half_box + 1
        else:
            # For even box sizes, adjust indices accordingly
            row_start = i_row - half_box
            row_end = i_row + half_box
            col_start = i_col - half_box
            col_end = i_col + half_box

        # Extract the patch
        patch = noise_img_scaled[row_start:row_end, col_start:col_end]

        # Check if the patch has the correct size
        if patch.shape == (box_sz, box_sz):
            # Append the patch to the noise_patches list (though we're skipping them)
            # If you don't want to store skipped patches, you can omit this step
            # noise_patches.append(patch)  # Uncomment if you want to keep skipped patches
            cnt_skip += 1
        else:
            print(f"Skipped invalid patch at ({i_row}, {i_col}) with shape {patch.shape}")

        # Define the region to zero out around the selected patch
        del_row_start = max(i_row - rDelAlgorithmL, 0)
        del_row_end = min(i_row + rDelAlgorithmL, Y_var.shape[0])
        del_col_start = max(i_col - rDelAlgorithmL, 0)
        del_col_end = min(i_col + rDelAlgorithmL, Y_var.shape[1])

        # Zero out the defined region to skip this patch
        Y_var[del_row_start:del_row_end, del_col_start:del_col_end] = -np.inf

    # Step 8: Start the second phase - Extracting the next num_of_patches patches with highest variance
    print("Starting the extraction phase...")
    while cnt_p < num_of_patches:
        p_max = np.max(Y_var)
        if p_max == -np.inf:
            print("No more valid patches available during extraction phase.")
            break

        # Find the index of the patch with the highest variance
        I = np.argmax(Y_var)
        i_row, i_col = np.unravel_index(I, Y_var.shape)

        # Define the boundaries of the patch
        half_box = box_sz // 2
        if box_sz % 2 == 1:
            # For odd box sizes, ensure the patch is centered correctly
            row_start = i_row - half_box
            row_end = i_row + half_box + 1
            col_start = i_col - half_box
            col_end = i_col + half_box + 1
        else:
            # For even box sizes, adjust indices accordingly
            row_start = i_row - half_box
            row_end = i_row + half_box
            col_start = i_col - half_box
            col_end = i_col + half_box

        # Extract the patch
        patch = noise_img_scaled[row_start:row_end, col_start:col_end]

        # Check if the patch has the correct size
        if patch.shape == (box_sz, box_sz):
            # Append the patch to the noise_patches list
            noise_patches.append(patch)
            cnt_p += 1
            print(f"Extracted patch {cnt_p} at ({i_row}, {i_col}) with variance {p_max:.2f}")
        else:
            print(f"Skipped invalid patch at ({i_row}, {i_col}) with shape {patch.shape}")

        # Define the region to zero out around the selected patch
        del_row_start = max(i_row - rDelAlgorithmL, 0)
        del_row_end = min(i_row + rDelAlgorithmL, Y_var.shape[0])
        del_col_start = max(i_col - rDelAlgorithmL, 0)
        del_col_end = min(i_col + rDelAlgorithmL, Y_var.shape[1])

        # Zero out the defined region to prevent overlapping
        Y_var[del_row_start:del_row_end, del_col_start:del_col_end] = 0

    return np.array(noise_patches)#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 12:50:30 2024

@author: kerenmor
"""

## This is the main code used as a proof of cnמcept
import os

from scipy.spatial.distance import squareform, pdist

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import convolve
import os.path
import os.path
from aspire.image import *
from aspire.image import Image as AspireImage
# from scipy.signal import correlate2d
from aspire.basis import FBBasis2D
import os
# from scipy.ndimage import rotate
# from scipy.interpolate import interp1d
from numpy.fft import fft2, ifft2
# from scipy.linalg import eigh
# from scipy.signal import convolve
# from scipy.signal import convolve2d
# import scipy.linalg
# from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from numba import jit, prange
# import tensorflow as tf
import pickle
from matplotlib.patches import Circle
# from scipy.spatial.distance import pdist, squareform
# from scipy.signal import convolve2d

# from scipy.signal import wiener
# from scipy.linalg import toeplitz
# import cupy as cp
# from cupyx.scipy.signal import correlate2d  # GPU-accelerated correlation
# from cupyx.scipy.linalg import toeplitz  # GPU-accelerated Toeplitz
import numpy as np
from scipy.signal import convolve2d
import torch
import torch.nn.functional as F
import gc
# Set device to MPS if available
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
import os
import mrcfile
import re
import numpy as np
from sklearn.decomposition import TruncatedSVD

# import cupy as cp
import numpy as np
# from cupyx.scipy.ndimage import convolve
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle



def sort_steerable_basis_fixed(basis_full, objects, noise_patches, max_dimension):
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

    # Calculate coefficients (square of projections)
    coeff_objects = np.square((basis_full.T) @ objects)
    coeff_noise = np.square((basis_full.T) @ noise_patches)

    # Full projected SNR
    projected_norm_objects = coeff_objects.sum(axis=0)
    projected_norm_noise = coeff_noise.sum(axis=0)
    plt.plot(np.sort(projected_norm_objects)[::-1])
    plt.plot(np.sort(projected_norm_noise)[::-1])
    plt.show()
    projected_snr = projected_norm_objects / projected_norm_noise.mean()
    min_idx_projected_snr = np.argmin(projected_snr)
    # projected_norm_objects = np.zeros((117))
    # for n in range(117):
    #     projected_norm_objects[n] = np.linalg.norm((basis_full.T) @ objects[:, n])**2
    # plt.plot(np.sort(projected_norm_objects)[::-1])
    # plt.show()
    # Initialize lists and variables
    projected_snr_per_dim = []
    sorted_basis_lst = []
    sorted_basis_idx = []
    projected_norm_objects = np.zeros(projected_norm_objects.shape)
    projected_norm_noise = np.zeros(projected_norm_noise.shape)
    basis_idx = []

    # Initialize a mask to track used indices
    active_mask = np.ones(basis_full.shape[1], dtype=bool)

    # Loop to select max_dimension basis vectors
    for dim in range(max_dimension):

        numerator = coeff_objects[:, min_idx_projected_snr] + projected_norm_objects[min_idx_projected_snr]
        denominator = np.zeros(coeff_noise.shape[0])

        # Update the denominator for all noise patches
        for n in range(coeff_noise.shape[1]):
            denominator += coeff_noise[:, n] + projected_norm_noise[n]
        denominator = denominator / coeff_noise.shape[1]

        ratio = numerator / denominator
        # ratio = numerator
        # Only consider active indices for max selection
        ratio[~active_mask] = -np.inf
        max_idx_candidate = np.argmax(ratio)
        basis_idx.append(max_idx_candidate)
        # Update norms
        projected_norm_objects += coeff_objects[max_idx_candidate, :]
        projected_norm_noise += coeff_noise[max_idx_candidate, :]

        # Append the sorted basis vector
        sorted_basis_lst.append(basis_full[:, max_idx_candidate])
        sorted_basis_idx.append(max_idx_candidate)

        # Mark this index as used
        active_mask[max_idx_candidate] = False

        # Sanity check
        # sorted_basis_vectors = np.column_stack(sorted_basis_lst)
        # projected_norm_objects_check = np.square((sorted_basis_vectors.T) @ objects).sum(axis=0)
        # projected_norm_noise_check = np.square((sorted_basis_vectors.T) @ noise_patches).sum(axis=0)
        # # projected_norm_objects_check = np.square((basis_full[:, ~active_mask].T) @ objects).sum(axis=0)
        # # projected_norm_noise_check = np.square((basis_full[:, ~active_mask].T) @ noise_patches).sum(axis=0)
        # diff_num = (projected_norm_objects_check - projected_norm_objects) / projected_norm_objects
        # diff_den = (projected_norm_noise_check - projected_norm_noise) / projected_norm_noise
        # print(f"Sanity Check objects (Iteration {dim + 1}):\n", diff_num)
        # print(f"Sanity Check noise (Iteration {dim + 1}):\n", diff_den)

        #Recompute projected SNR
        projected_snr = projected_norm_objects / projected_norm_noise.mean()
        min_idx_projected_snr = np.argmin(projected_snr)
        # max_idx_projected_snr = np.argmax(projected_snr)
        projected_snr_per_dim.append(projected_snr[min_idx_projected_snr])


    # Stack sorted basis vectors
    sorted_basis_vectors = np.column_stack(sorted_basis_lst)

    return sorted_basis_vectors, projected_snr_per_dim,projected_norm_objects,projected_norm_noise, basis_idx
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


def plot_patches_with_coordinates(Y, noise_patches_coor, patch_size):
    """
    Plots a micrograph with coordinates marked and boxes drawn around them to represent patches.

    Parameters:
    Y (ndarray): The micrograph image to be plotted.
    noise_patches_coor (ndarray or list): Array or list of tuples (x, y) coordinates for the center of each patch.
    patch_size (int): The size of the patch (side length of the square box).
    """
    # Convert noise_patches_coor to a NumPy array if it’s a list of tuples
    noise_patches_coor = np.array(noise_patches_coor)

    # Plot the micrograph image
    plt.figure(figsize=(8, 8))
    plt.imshow(Y, cmap='gray')  # You can use other colormaps if needed

    # Plot the coordinates on top of the image
    plt.scatter(noise_patches_coor[:, 0], noise_patches_coor[:, 1], c='red', s=10, label='Coordinates')

    # Add rectangles around each coordinate (as the center of the patch)
    for (x, y) in noise_patches_coor:
        # Create a rectangle with bottom-left corner adjusted to center around (x, y)
        rect = Rectangle((x - patch_size // 2, y - patch_size // 2), patch_size, patch_size,
                         linewidth=1, edgecolor='blue', facecolor='none', linestyle='--')
        plt.gca().add_patch(rect)

    # Optional: Add labels and title
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Micrograph with Noise Patch Coordinates and Boxes')
    plt.legend()

    # Show the plot
    plt.show()

def projected_noise_simulation_from_noise_patches_para_cup(noise_samples, basis, num_of_exp_noise):
    """
    Simulates projected noise using given noise samples and basis functions in batches,
    distributing computation across multiple GPUs using CuPy.
    """

    # 🔹 **Keep `noise_samples` on CPU, move only batches**
    x = 1
    noise_samples_cpu = noise_samples.astype(np.float32)

    # List available GPUs
    num_gpus = cp.cuda.runtime.getDeviceCount()
    if num_gpus == 0:
        raise RuntimeError("No GPUs found! Consider using a CPU-based approach.")

    print(f"Using {num_gpus} GPUs for parallel processing.")

    # 🔹 **Move `basis` to GPU**
    basis_gpu = cp.asarray(basis, dtype=cp.float32)

    # 🔹 **Compute output size**
    sz = int(np.sqrt(noise_samples.shape[0]))  # Image size
    test_conv = convolve(cp.reshape(cp.asarray(noise_samples_cpu[:, 0]), (sz, sz)),
                         cp.reshape(basis_gpu[:, :, 0], (basis.shape[0], basis.shape[1])), mode='constant')
    sz_pn = test_conv.size  # **Flattened size of convolved output (should be 16641)**

    print(f"Output shape: ({num_of_exp_noise}, {sz_pn})")

    # 🔹 **Use disk storage to avoid memory crash**
    S_z_n = np.memmap('S_z_n.dat', dtype=np.float32, mode='w+', shape=(num_of_exp_noise, sz_pn))

    # 🔹 **Reduce memory use by processing small batches**
    max_batch_size_per_gpu = max(1, num_of_exp_noise // (10 * num_gpus))
    print(f"Using batch size: {max_batch_size_per_gpu}")

    # 🔹 **Preallocate memory for flipped basis (on GPU)**
    flipped_basis_stack = cp.stack([cp.flip(cp.flip(basis_gpu[:, :, j], 0), 1) for j in range(basis.shape[2])])

    # 🔹 **Function to process a batch on a given GPU**
    def process_batch(start, end, device_id):
        """Processes a batch on a given GPU, ensuring basis is moved to the correct GPU."""
        with cp.cuda.Device(device_id):
            # 🔹 **Ensure `flipped_basis_stack` is on the correct GPU**
            flipped_basis_stack_gpu = flipped_basis_stack.copy()

            # Move batch to GPU
            noise_batch_gpu = cp.asarray(noise_samples_cpu[:, start:end])  # Move only batch to GPU
            noise_imgs_gpu = cp.reshape(noise_batch_gpu, (end - start, sz, sz))

            # Perform convolution for each image
            conv_results = cp.array([
                convolve(noise_imgs_gpu[i], flipped_basis_stack_gpu[j], mode='constant')**2
                for i in range(end - start) for j in range(basis.shape[2])
            ])

            # Sum over all basis functions and flatten
            conv_results = cp.sum(conv_results.reshape(end - start, basis.shape[2], sz_pn), axis=1)

            # 🔹 **Move results back to CPU and free GPU memory**
            conv_results_cpu = cp.asnumpy(conv_results)
            del noise_batch_gpu, noise_imgs_gpu, conv_results, flipped_basis_stack_gpu
            cp.get_default_memory_pool().free_all_blocks()

            return conv_results_cpu

    # 🔹 **Process in smaller batches**
    for batch_idx in range((num_of_exp_noise + max_batch_size_per_gpu - 1) // max_batch_size_per_gpu):
        start = batch_idx * max_batch_size_per_gpu
        end = min((batch_idx + 1) * max_batch_size_per_gpu, num_of_exp_noise)

        # Process only one batch at a time to avoid memory overload
        batch_result = process_batch(start, end, batch_idx % num_gpus)

        # Store result in file-backed array instead of RAM
        S_z_n[start:end] = batch_result

    # 🔹 **Flush memory-mapped file and close it**
    S_z_n.flush()

    return S_z_n


def compute_top_eigen_svd(patches, num_components=15):
    """
    Compute the top eigenvectors and eigenvalues using Truncated SVD,
    ensuring the data is centered before applying SVD.

    Parameters:
    - patches: (15, 129, 129) array where each patch is 129x129
    - num_components: Number of top eigenvectors to extract (default: 15)

    Returns:
    - eigenvalues: (15,) array of top 15 eigenvalues
    - eigenvectors: (129*129, 15) array of top 15 eigenvectors
    """
    # Reshape patches from (15, 129, 129) to (15, 16641) for SVD
    patches_reshaped = patches.reshape(patches.shape[0], -1)  # (15, 16641)

    # Center the data: Subtract the mean image across patches
    mean_patch = np.mean(patches_reshaped, axis=0, keepdims=True)  # (1, 16641)
    patches_centered = patches_reshaped - mean_patch

    # Apply Truncated SVD to extract the top principal components
    svd = TruncatedSVD(n_components=num_components)
    transformed_patches = svd.fit_transform(patches_centered)  # Ensures singular values are computed

    # The top eigenvalues are approximated as the square of the singular values
    top_eigenvalues = svd.singular_values_**2 / (patches.shape[0] - 1)

    # The top eigenvectors are the right singular vectors
    top_eigenvectors = svd.components_.T  # (16641, 15)

    return top_eigenvalues, top_eigenvectors

def generate_gaussian_samples(eigenvalues, eigenvectors, num_samples=50):
    """
    Generate multiple samples from a Gaussian process estimated from noise patches.

    Parameters:
    - eigenvalues: (15,) array of top 15 eigenvalues
    - eigenvectors: (N, 15) array of top 15 eigenvectors
    - num_samples: Number of noise samples to generate

    Returns:
    - noise_samples: (N, num_samples) array of generated Gaussian samples
    """
    # Compute the square root of eigenvalues (15x15 diagonal matrix)
    Lambda_sqrt = np.diag(np.sqrt(eigenvalues))

    # Generate standard normal random vectors (15 x num_samples)
    Z = np.random.randn(len(eigenvalues), num_samples)

    # Compute correlated noise in reduced dimension
    Y = Lambda_sqrt @ Z

    # Transform back to the original space
    noise_samples = eigenvectors @ Y

    return noise_samples

def extract_patches(micrograph, coord_path, box_size=None, file_type="star"):
    """
    Extract patches from the micrograph image based on coordinates in .star or .box files.

    Parameters:
    micrograph : ndarray : The micrograph image as a NumPy array.
    coord_path : str : Path to the .star or .box file.
    box_size : int : Size of the square patch to extract (required for .star files, ignored for .box files).
    file_type : str : Type of file, either 'star' or 'box'.

    Returns:
    patches : list : A list of extracted patches as NumPy arrays.
    """
    if file_type not in ["star", "box"]:
        raise ValueError("Invalid file_type. Must be 'star' or 'box'.")

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

    def load_box(file_path):
        boxes = np.loadtxt(file_path, delimiter='\t', usecols=(0, 1, 2, 3))
        return boxes  # Array of [x, y, w, h]

    # Load coordinates
    if file_type == "star":
        if box_size is None:
            raise ValueError("box_size must be specified for .star files.")
        coordinates = load_star(coord_path)
    else:  # file_type == "box"
        coordinates = load_box(coord_path)

    # Extract patches
    patches = []
    for i, coord in enumerate(coordinates):
        if file_type == "star":
            # For .star files, coord is (x, y)
            x, y = int(round(coord[0])), int(round(coord[1]))
            half_box = box_size // 2
            x_start = max(0, x - half_box)
            x_end = min(micrograph.shape[1], x + half_box)
            y_start = max(0, y - half_box)
            y_end = min(micrograph.shape[0], y + half_box)
            patch = micrograph[y_start:y_end, x_start:x_end]
        else:  # file_type == "box"
            # For .box files, coord is (x, y, w, h)
            x, y, w, h = map(int, coord)
            x_start = max(0, x)
            x_end = min(micrograph.shape[1], x + w)
            y_start = max(0, y)
            y_end = min(micrograph.shape[0], y + h)
            patch = micrograph[y_start:y_end, x_start:x_end]

        # If the patch is not the exact desired size, pad with zeros (only for star patches)
        if file_type == "star" and patch.shape != (box_size, box_size):
            patch_padded = np.zeros((box_size, box_size), dtype=patch.dtype)
            patch_padded[:patch.shape[0], :patch.shape[1]] = patch
            patch = patch_padded

        patches.append(patch)

    print(f"Extracted {len(patches)} patches from {coord_path}.")
    return patches


def read_coordinates(coord_file):
    """
    Read coordinates from .star, .csv, or .box files.

    Parameters:
        coord_file : str : Path to the coordinate file.

    Returns:
        coordinates : ndarray : Array of (x, y) coordinates.
        file_type   : str     : 'star', 'csv', or 'box' based on the file type.
    """
    coordinates = []

    # Detect file type
    if coord_file.endswith('.star'):
        file_type = 'star'
    elif coord_file.endswith('.csv'):
        file_type = 'csv'
    elif coord_file.endswith('.box'):
        file_type = 'box'
    else:
        raise ValueError("Unsupported file type. Only .star, .csv, and .box files are supported.")

    print(f"Reading file: {coord_file} (type: {file_type})")

    with open(coord_file, 'r') as f:
        inside_loop = False  # Flag for .star files

        for line_number, line in enumerate(f, 1):
            print(f"Raw line {line_number}: {line!r}")  # Debug: Print raw line
            line = line.strip()

            # Parsing for .star files
            if file_type == 'star':
                if not line or line.startswith(('data_', 'loop_', '_')):
                    if line.startswith('loop_'):
                        inside_loop = True
                        print(f"'loop_' detected on line {line_number}. Starting to process data rows.")
                    else:
                        print(f"Skipping header/metadata line {line_number}: {line}")
                    continue

                if inside_loop:
                    tokens = re.split(r'\s+', line)  # Split on any whitespace
                    print(f"Tokens from line {line_number}: {tokens}")  # Debug: Show tokens
                    if len(tokens) >= 2:
                        try:
                            x = int(float(tokens[0]))  # Column #1
                            y = int(float(tokens[1]))  # Column #2
                            coordinates.append((x, y))
                            print(f"Valid coordinate: ({x}, {y})")
                        except ValueError as e:
                            print(f"Error parsing line {line_number}: {line.strip()}, Error: {e}")
                            continue

            # Parsing for .csv files
            elif file_type == 'csv':
                tokens = line.split(',')
                print(f"Tokens from line {line_number}: {tokens}")  # Debug: Show tokens
                if len(tokens) >= 2:
                    try:
                        x = int(float(tokens[0]))  # First column
                        y = int(float(tokens[1]))  # Second column
                        coordinates.append((x, y))
                        print(f"Valid coordinate: ({x}, {y})")
                    except ValueError as e:
                        print(f"Error parsing line {line_number}: {line.strip()}, Error: {e}")
                        continue

            # Parsing for .box files
            elif file_type == 'box':
                tokens = line.split()
                print(f"Tokens from line {line_number}: {tokens}")  # Debug: Show tokens
                if len(tokens) >= 4:
                    try:
                        # Box file: top-left (x, y) and dimensions (w, h)
                        x = int(tokens[0])
                        y = int(tokens[1])
                        w = int(tokens[2])
                        h = int(tokens[3])
                        # Convert top-left (x, y) to center coordinates
                        x_center = x + w // 2
                        y_center = y + h // 2
                        coordinates.append((x_center, y_center))
                        print(f"Valid coordinate (box): ({x_center}, {y_center})")
                    except ValueError as e:
                        print(f"Error parsing line {line_number}: {line.strip()}, Error: {e}")
                        continue

    # Check for empty coordinates
    if not coordinates:
        print(f"No valid coordinates found in {coord_file}.")
    else:
        print(f"Total coordinates parsed: {len(coordinates)}")

    return np.array(coordinates), file_type







def check_orthonormal_columns(matrix, tol=1e-6):
    """
    Check if the columns of a given matrix are orthonormal.

    :param matrix: 2D numpy array representing the matrix.
    :param tol: Tolerance for checking orthonormality.
    :return: True if columns are orthonormal, False otherwise.
    """
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")

    # Compute the dot product of the matrix with its transpose
    dot_product = np.dot(matrix.T, matrix)

    # Check if the dot product is close to the identity matrix
    identity_matrix = np.eye(matrix.shape[1])
    orthonormal = np.allclose(dot_product, identity_matrix, atol=tol)

    if not orthonormal:
        print("Dot product matrix (should be identity):\n", dot_product)

    return orthonormal


def save_coortodist_data(coortodistData, filename):
    with open(filename, 'wb') as file:
        pickle.dump(coortodistData, file)

# Function to load coortodistData
def load_coortodist_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
#def ensure_positive_definite(matrix):
 #   """Attempt Cholesky decomposition and adjust matrix to be positive definite if necessary."""
  #  try:
   #     _ = scipy.linalg.cholesky(matrix, lower=True)
    #    return matrix
    #except np.linalg.LinAlgError:
        # Adding a small value to the diagonal
     #   return make_positive_definite(matrix)

#def ensure_positive_definite(matrix, epsilon=1e-6, max_attempts=10):
#     """
#     Ensure the input matrix is positive definite by attempting to add a small value to the diagonal.

#     Parameters:
#     matrix : ndarray
#         Input matrix to check.
#     epsilon : float, optional
#         Initial value to add to the diagonal if the matrix is not positive definite.
#     max_attempts : int, optional
#         Maximum number of attempts to adjust the matrix.

#     Returns:
#     matrix : ndarray
#         Adjusted positive definite matrix.

#     Raises:
#     LinAlgError:
#         If the matrix cannot be made positive definite after max_attempts.
#     """
#     for attempt in range(max_attempts):
#         try:
#             _ = scipy.linalg.cholesky(matrix, lower=True)
#             return matrix
#         except np.linalg.LinAlgError:
#             diag_increment = epsilon * (10 ** attempt)  # Increase epsilon exponentially
#             matrix += np.eye(matrix.shape[0]) * diag_increment
#             print(f"Matrix adjusted with diagonal increment: {diag_increment:.2e}")

#     raise np.linalg.LinAlgError("Matrix could not be made positive definite after multiple attempts.")


def ensure_positive_definite_sqrt(cov_matrix):
    """
    Ensures the covariance matrix is positive semi-definite and returns its square root.

    If more than 10% of eigenvalues are negative, the function raises an error.

    Parameters:
        cov_matrix (cp.ndarray): Input covariance matrix (H*W, H*W).

    Returns:
        cov_matrix_sqrt (cp.ndarray): Square root of the adjusted covariance matrix.
    """
    # Free GPU memory
    cp.cuda.Device(6).use()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()

    cov_matrix = cp.asarray(cov_matrix, dtype=cp.float32)

    try:
        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = cp.linalg.eigh(cov_matrix)
    except AttributeError:
        print("⚠️ CuPy failed on GPU. Switching to CPU...")
        cov_matrix_cpu = cp.asnumpy(cov_matrix)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix_cpu)
        eigvals, eigvecs = cp.asarray(eigvals), cp.asarray(eigvecs)

    # Count negative eigenvalues
    num_negative = cp.sum(eigvals < 0)
    total_eigenvalues = eigvals.size

    # If more than 10% of eigenvalues are negative, raise an error
    if num_negative > 0.1 * total_eigenvalues:
        raise ValueError(f"❌ Too many negative eigenvalues! ({num_negative}/{total_eigenvalues})")

    # Find the smallest positive eigenvalue
    min_positive = cp.min(eigvals[eigvals > 0]) if cp.any(eigvals > 0) else 0

    # Replace negative eigenvalues with the smallest positive eigenvalue
    eigvals = cp.where(eigvals < 0, min_positive, eigvals)

    # Compute the square root of the eigenvalues
    eigvals_sqrt = cp.sqrt(eigvals)

    # Compute the square root of the covariance matrix
    cov_matrix_sqrt = eigvecs @ cp.diag(eigvals_sqrt) @ eigvecs.T

    return cov_matrix_sqrt


# def ensure_positive_definite_sqrt(cov_matrix):
#     """
#     Ensures the covariance matrix is positive semi-definite and returns its square root.

#     If more than 10% of eigenvalues are negative, the function raises an error.

#     Parameters:
#         cov_matrix (tf.Tensor or np.ndarray): Input covariance matrix (H*W, H*W).

#     Returns:
#         cov_matrix_sqrt (np.ndarray): Square root of the adjusted covariance matrix.
#     """
#     # Convert input to TensorFlow tensor if it's a NumPy array
#     cov_matrix = tf.convert_to_tensor(cov_matrix, dtype=tf.float32)

#     # Compute eigenvalues and eigenvectors
#     eigvals, eigvecs = tf.linalg.eigh(cov_matrix)

#     # Count negative eigenvalues
#     num_negative = tf.reduce_sum(tf.cast(eigvals < 0, tf.int32))
#     total_eigenvalues = tf.size(eigvals)

#     # If more than 10% of eigenvalues are negative, raise an error
#     if tf.greater(num_negative, tf.cast(0.1 * tf.cast(total_eigenvalues, tf.float32), tf.int32)):
#         raise ValueError(f"❌ Too many negative eigenvalues! ({num_negative.numpy()}/{total_eigenvalues.numpy()})")

#     # Find the smallest positive eigenvalue
#     min_positive = tf.reduce_min(tf.boolean_mask(eigvals, eigvals > 0))

#     # Replace negative eigenvalues with the smallest positive eigenvalue
#     eigvals = tf.where(eigvals < 0, min_positive, eigvals)

#     # Compute the square root of the eigenvalues
#     eigvals_sqrt = tf.sqrt(eigvals)

#     # Compute the square root of the covariance matrix
#     cov_matrix_sqrt = eigvecs @ tf.linalg.diag(eigvals_sqrt) @ tf.transpose(eigvecs)

#     # Convert the result back to NumPy array
#     return cov_matrix_sqrt.numpy()

def make_positive_definite(matrix, epsilon=1e-6):
    """Ensure the matrix is positive definite by adding a small value to the diagonal."""
    return matrix + epsilon * np.eye(matrix.shape[0])
def convolve_and_sum1(img, basis, num_of_basis_functions):
    img_sz = img.shape[0]
    S = np.zeros((img_sz, img_sz, num_of_basis_functions))
    for m in range(num_of_basis_functions):
        # Use np.flip only once outside the loop
        kernel = np.flip(np.flip(basis[:, :, m], axis=0), axis=1)
        S[:, :, m] = convolve2d(img, kernel, mode='same') ** 2
    return np.sum(S, axis=2)
def convolve_and_sum(img, basis, num_of_basis_functions):
    img_sz = img.shape[0]
    S = torch.zeros((img_sz, img_sz, num_of_basis_functions), device=device)

    # Precompute flipped kernels and make a copy to avoid negative strides
    flipped_kernels = np.flip(np.flip(basis, axis=0), axis=1).copy()

    # Move data to device
    img_tensor = torch.tensor(img, device=device, dtype=torch.float32)
    flipped_kernels_tensor = torch.tensor(flipped_kernels, device=device, dtype=torch.float32)

    for m in range(num_of_basis_functions):
        kernel = flipped_kernels_tensor[:, :, m]
        convolved = torch.nn.functional.conv2d(img_tensor.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding='same')
        S[:, :, m] = convolved.squeeze() ** 2

    return torch.sum(S, axis=2).cpu().numpy()
def save_fig(directory, filename, fig):
    """
    Saves a Matplotlib figure to the specified directory.

    Parameters:
    directory : Path to the directory where the figure will be saved.
    filename  : Name of the file.
    fig       : Matplotlib figure object.
    """
    filepath = os.path.join(directory, filename)
    fig.savefig(filepath, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
def coords_output2(Y_peaks_loc, addr_coords, microName, mgScale, mgBigSz, K, patchSzPickBox=300):
    """
    Writes particle coordinates to .star and .box files.

    Parameters:
    Y_peaks_loc : ndarray
        Array of peak locations.
    addr_coords : str
        Directory address where files will be saved.
    microName : str
        Name of the micrograph.
    mgScale : float
        Scaling factor for micrograph.
    mgBigSz : tuple
        Size of the micrograph (height, width).
    K : int
        Number of peaks to process.
    patchSzPickBox : int, optional
        Size of the patch for picking box. Default is 300.
    """

    # Ensure the directory exists
    os.makedirs(addr_coords, exist_ok=True)

    # Open files for writing
    particlesCordinateStar_path = os.path.join(addr_coords, f'{microName}.star')
    particlesCordinateBox_path = os.path.join(addr_coords, f'{microName}.box')

    with open(particlesCordinateStar_path, 'w') as particlesCordinateStar, \
            open(particlesCordinateBox_path, 'w') as particlesCordinateBox:

        # Format Relion star file
        particlesCordinateStar.write('data_\n\n')
        particlesCordinateStar.write('loop_\n')
        particlesCordinateStar.write('_rlnCoordinateX #1\n')
        particlesCordinateStar.write('_rlnCoordinateY #2\n')

        for i in range(K):
            i_colPatch = Y_peaks_loc[i, 1]
            i_rowPatch = Y_peaks_loc[i, 0]
            x_star = (1 / mgScale) * i_colPatch
            y_star = (mgBigSz[0] + 1) - (1 / mgScale) * i_rowPatch
            x_box = x_star - np.floor(patchSzPickBox / 2)
            y_box = y_star - np.floor(patchSzPickBox / 2)

            particlesCordinateStar.write(f'{x_star:.0f}\t{y_star:.0f}\n')
            particlesCordinateBox.write(f'{x_box:.0f}\t{y_box:.0f}\t{patchSzPickBox:.0f}\t{patchSzPickBox:.0f}\n')
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



def coords_output_csv(Y_peaks_loc, addr_coords, microName, mgScale, mgBigSz, K, patchSzPickBox, ground_truth_csv=None):
    """
    Writes particle coordinates to .csv (and optionally .star/.box) files for given peaks.

    Parameters:
    Y_peaks_loc : Array of peak locations.
    addr_coords : Directory path for output files.
    microName   : Name of the micrograph.
    mgScale     : Scaling factor for micrograph.
    mgBigSz     : Size of the micrograph (tuple or list).
    K           : Number of peaks to output.
    patchSzPickBox : Size of the box around each particle.
    ground_truth_csv : Path to the ground truth .csv file (optional).
    """
    # Prepare file paths
    csv_file_path = os.path.join(addr_coords, f"{microName}.csv")
    star_file_path = os.path.join(addr_coords, f"{microName}.star")
    box_file_path = os.path.join(addr_coords, f"{microName}.box")

    # Initialize ground truth data if provided
    ground_truth = None
    if ground_truth_csv and os.path.exists(ground_truth_csv):
        ground_truth = pd.read_csv(ground_truth_csv)

    # Open files for writing
    with open(star_file_path, 'w') as particlesCordinateStar, open(box_file_path, 'w') as particlesCordinateBox:
        # Write headers for the .star file
        particlesCordinateStar.write("data_\n\nloop_\n")
        particlesCordinateStar.write("_rlnCoordinateX #1\n")
        particlesCordinateStar.write("_rlnCoordinateY #2\n\n")

        # Prepare .csv output
        output_data = []

        for i in range(K):
            i_colPatch = Y_peaks_loc[i, 1]
            i_rowPatch = Y_peaks_loc[i, 0]

            # Calculate coordinates for .star and .csv
            x_coord = (1 / mgScale) * i_colPatch
            y_coord = (mgBigSz[0] + 1) - (1 / mgScale) * i_rowPatch
            output_data.append([x_coord, y_coord])

            # Write to .star file
            particlesCordinateStar.write(f"{x_coord:.0f}\t{y_coord:.0f}\n")

            # Calculate coordinates for .box file
            x_box = x_coord - patchSzPickBox // 2
            y_box = y_coord - patchSzPickBox // 2
            particlesCordinateBox.write(f"{x_box:.0f}\t{y_box:.0f}\t{patchSzPickBox:.0f}\t{patchSzPickBox:.0f}\n")

        # Write to .csv
        output_df = pd.DataFrame(output_data, columns=["x", "y"])
        output_df.to_csv(csv_file_path, index=False)

    if ground_truth is not None:
        # Compare with ground truth
        comparison = pd.merge(output_df, ground_truth, on=["x", "y"], how="inner")
        print(f"Matched particles: {len(comparison)} / {K}")
        comparison.to_csv(os.path.join(addr_coords, f"{microName}_matched.csv"), index=False)



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

def extract_non_overlapping_patches(image, patch_size, step):
    """
    Extract patches of size (patch_size x patch_size) from the input image.
    The next patch's top-left corner is step pixels apart.
    Returns an array of patches and the patch top-left corners (i, j).
    """
    patches = []
    centers = []
    img_height, img_width = image.shape
    patch_height, patch_width = patch_size

    # Compute how many patches we can fit along each dimension
    n_patches_y = (img_height - patch_height) // step + 1
    n_patches_x = (img_width - patch_width) // step + 1

    # Loop over the image using the calculated number of patches
    for i in range(n_patches_y):
        for j in range(n_patches_x):
            start_y = i * step
            start_x = j * step
            patch = image[start_y:start_y + patch_height, start_x:start_x + patch_width]
            patches.append(patch)
            centers.append((start_y, start_x))  # Store the top-left corner of each patch

    return np.array(patches), centers



# def compute_projection_norms(patches, basis_vectors):
#     """
#     Computes the norm of the projection of each patch onto the given 30 basis vectors.
#     Patches are of shape (64, 64), and basis_vectors is of shape (64, 64, 30).
#     """
#     n_patches = patches.shape[0]
#     n_basis = basis_vectors.shape[2]  # Number of basis vectors (30)

#     projection_norms = np.zeros(n_patches)
#     for i in range(n_patches):
#         patch_flattened = patches[i].flatten()

#         # For each basis vector, compute the projection
#         inner_products = np.zeros(n_basis)
#         for j in range(n_basis):
#             basis_flattened = basis_vectors[:, :, j].flatten()
#             inner_products[j] = np.dot(patch_flattened, basis_flattened)

#         # Compute the norm of the projection onto the basis set
#         sum_of_squares = np.sum(inner_products ** 2)
#         projection_norms[i] = np.sqrt(sum_of_squares)

#     return projection_norms



def peak_algorithm_cont_mask(img, basis, sideLengthAlgorithm, contamination_mask=None, obj_sz_down_scaled=None, contamination_threshold=0.5, debug=False):
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
    # Ensure using CPU (change to 'cuda' if GPU is available)
    device = torch.device('cpu')

    # Convert image and basis to PyTorch tensors and move to device (CPU)
    img_tensor = torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    basis = torch.tensor(basis, dtype=torch.float32, device=device)
    num_of_basis_functions = basis.shape[2]
    rDelAlgorithm = round(sideLengthAlgorithm // 2)

    peaks = []
    peaks_loc = []
    obj_sz = basis.shape[0]

    # Perform convolution and sum over basis functions
    S = torch.zeros_like(img_tensor, dtype=torch.float32, device=device)

    flipped_basis_list = [torch.flip(basis[:, :, j], dims=[0, 1]) for j in range(num_of_basis_functions)]
    for flipped_basis in flipped_basis_list:
        flipped_basis_shape = flipped_basis.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        conv_result = F.conv2d(img_tensor, flipped_basis_shape, stride=1, padding='same') ** 2
        S += conv_result

    S = S.squeeze().cpu().numpy()
    scoringMat = S.copy()
    scoringMat[:obj_sz // 2, :] = 0
    scoringMat[:, :obj_sz // 2] = 0
    scoringMat[-obj_sz // 2:, :] = 0
    scoringMat[:, -obj_sz // 2:] = 0

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
        rows, cols = np.ogrid[:scoringMat.shape[0], :scoringMat.shape[1]]
        mask = (rows - i_row) ** 2 + (cols - i_col) ** 2 <= rDelAlgorithm ** 2
        if debug:
            mask_circ = ((rows - i_row) ** 2 + (cols - i_col) ** 2 >= rDelAlgorithm ** 2 - 50) & \
                        ((rows - i_row) ** 2 + (cols - i_col) ** 2 <= rDelAlgorithm ** 2)
            img_copy = img.copy()
            img_copy[mask_circ] = 3000

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

    num_of_basis_functions = basis.shape[2]
    rDelAlgorithm = round(sideLengthAlgorithm // 2)
    obj_sz = basis.shape[0]
    basis = basis.astype(np.float32)
    img = img.astype(np.float32)

    # Pre-allocate the scoring map
    S = np.zeros(img.shape, dtype=np.float32)

    # Perform convolution using fftconvolve from SciPy
    for j in range(num_of_basis_functions):
        # Flip the basis function horizontally and vertically
        flipped_basis = np.flip(np.flip(basis[:, :, j], axis=0), axis=1)

        # Perform convolution and sum of squared convolutions
        conv_result = fftconvolve(img, flipped_basis, mode='same')
        S += conv_result ** 2

    scoringMat = S.copy()

    # Zero out the edges to avoid boundary artifacts
    scoringMat[:obj_sz // 2, :] = 0
    scoringMat[:, :obj_sz // 2] = 0
    scoringMat[-obj_sz // 2:, :] = 0
    scoringMat[:, -obj_sz // 2:] = 0

    peaks = []
    peaks_loc = []

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
        rows, cols = np.ogrid[:scoringMat.shape[0], :scoringMat.shape[1]]
        mask = (rows - i_row) ** 2 + (cols - i_col) ** 2 <= rDelAlgorithm ** 2
        if debug:
            mask_circ = ((rows - i_row) ** 2 + (cols - i_col) ** 2 >= rDelAlgorithm ** 2 - 50) & \
                        ((rows - i_row) ** 2 + (cols - i_col) ** 2 <= rDelAlgorithm ** 2)
            img_copy = img.copy()
            img_copy[mask_circ] = 3000

        scoringMat[mask] = 0
        peaks.append(pMax)
        peaks_loc.append([i_row, i_col])

    return np.array(peaks), np.array(peaks_loc), S

def peak_algorithm_cont_mask_torch_swap(img, basis, sideLengthAlgorithm, contamination_mask=None, obj_sz_down_scaled=None,
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
    # Use GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert image and basis to PyTorch tensors
    img_tensor = torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    basis_tensor = torch.tensor(basis, dtype=torch.float32, device=device)

    num_of_basis_functions = basis_tensor.shape[2]
    rDelAlgorithm = round(sideLengthAlgorithm // 2)
    obj_sz = basis_tensor.shape[0]

    # Pre-allocate the scoring map and move it to the device
    S = torch.zeros_like(img_tensor, dtype=torch.float32, device=device)

    # Perform convolution using in-place operations to save memory
    for j in range(num_of_basis_functions):
        flipped_basis = torch.flip(basis_tensor[:, :, j], dims=[0, 1])
        flipped_basis = flipped_basis.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        # Use in-place operation for convolution result
        conv_result = F.conv2d(img_tensor, flipped_basis, stride=1, padding='same')
        conv_result.pow_(2)  # In-place square operation
        S.add_(conv_result)  # In-place addition

    # Move the scoring map back to CPU and convert to NumPy
    S = S.squeeze().cpu().numpy()
    scoringMat = S.copy()

    # Clear the cache to free up GPU memory
    torch.cuda.empty_cache()

    # Zero out the edges to avoid boundary artifacts
    scoringMat[:obj_sz // 2, :] = 0
    scoringMat[:, :obj_sz // 2] = 0
    scoringMat[-obj_sz // 2:, :] = 0
    scoringMat[:, -obj_sz // 2:] = 0

    peaks = []
    peaks_loc = []

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
        rows, cols = np.ogrid[:scoringMat.shape[0], :scoringMat.shape[1]]
        mask = (rows - i_row) ** 2 + (cols - i_col) ** 2 <= rDelAlgorithm ** 2
        if debug:
            mask_circ = ((rows - i_row) ** 2 + (cols - i_col) ** 2 >= rDelAlgorithm ** 2 - 50) & \
                        ((rows - i_row) ** 2 + (cols - i_col) ** 2 <= rDelAlgorithm ** 2)
            img_copy = img.copy()
            img_copy[mask_circ] = 3000

        scoringMat[mask] = 0
        peaks.append(pMax)
        peaks_loc.append([i_row, i_col])

    # Clear cache one last time before returning results
    torch.cuda.empty_cache()

    return np.array(peaks), np.array(peaks_loc), S


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

def peak_algorithm_cont_mask_tf_old(img, basis, sideLengthAlgorithm, contamination_mask=None, obj_sz_down_scaled=None, contamination_threshold=0.5, debug=False):
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
        rows, cols = np.ogrid[:scoringMat.shape[0], :scoringMat.shape[1]]
        mask = (rows - i_row) ** 2 + (cols - i_col) ** 2 <= rDelAlgorithm ** 2
        if debug:
            mask_circ = ((rows - i_row) ** 2 + (cols - i_col) ** 2 >= rDelAlgorithm ** 2 - 50) & \
                        ((rows - i_row) ** 2 + (cols - i_col) ** 2 <= rDelAlgorithm ** 2)
            img_copy = img_no_mean.copy()
            img_copy[mask_circ] = 3000

        scoringMat[mask] = 0
        peaks.append(pMax)
        peaks_loc.append([i_row, i_col])

    return np.array(peaks), np.array(peaks_loc), S

def peak_algorithm_boxes(img, basis, sideLengthAlgorithm):
    """
    Identify peaks in an image using convolution with basis functions.

    Parameters:
    img                : Input image.
    basis              : Array of basis functions.
    sideLengthAlgorithm: Length parameter for peak extraction.

    Returns:
    peaks     : List of peak values.
    peaks_loc : List of peak locations (coordinates).
    S         : Scoring map from convolution.
    """

    num_of_basis_functions = basis.shape[2]

    img_sz_max = max(img.shape[0],img.shape[1])
    rDelAlgorithm = round(sideLengthAlgorithm // 2)
    #peaks_size = (img_sz_max // rDelAlgorithm) ** 2
    # Convert the result explicitly to an integer before using as array size
    #peaks_size = round(peaks_size)
    # Initialize lists to store peaks and their locations
    peaks = []
    peaks_loc = []
    obj_sz = basis.shape[0]
    basis_vectors = np.reshape(basis, (basis.shape[0]**2, basis.shape[2]))
    # S = convolve_and_sum(img, basis, num_of_basis_functions)
    img_shape = tf.reshape(img, [1, img.shape[0], img.shape[1], 1])
    img_shape = tf.cast(img_shape, tf.float32)
    flipped_basis = np.flip(np.flip(basis[:, :, -1], 0), 1)
    flipped_basis_shape = tf.reshape(flipped_basis,[flipped_basis.shape[0], flipped_basis.shape[1], 1, 1 ])
    flipped_basis_shape = tf.cast(flipped_basis_shape, tf.float32)
    S= tf.nn.conv2d(img_shape, flipped_basis_shape,strides=[1,1,1,1], padding='SAME') **2
    for j in range(basis.shape[2]-1):
        flipped_basis = np.flip(np.flip(basis[:, :, j], 0), 1)
        flipped_basis_shape = tf.reshape(flipped_basis,[flipped_basis.shape[0], flipped_basis.shape[1], 1, 1 ])
        flipped_basis_shape = tf.cast(flipped_basis_shape, tf.float32)
        S += tf.nn.conv2d(img_shape, flipped_basis_shape,strides=[1,1,1,1], padding='SAME') ** 2
    S = tf.squeeze(S).numpy()
    # Loop over each possible center point
    # B = basis.shape[0]
    # N = img_sz
    # half_B = int(B // 2)  # Half the patch size to find boundaries around the center
    # S_n = np.zeros((img_sz, img_sz))
    # for i in range(half_B, N - half_B):
    #     for j in range(half_B, N - half_B):
    #         # Extract the patch centered at (i, j)
    #         patch = img[i - half_B:i + half_B, j - half_B:j + half_B]
    #         patch_flattened = patch.flatten()
    #         inner_products = np.dot(basis_vectors.T,patch_flattened)
    #         sum_of_squares = np.sum(inner_products ** 2)
    #         projection_norm = np.sqrt(sum_of_squares)
    #         S_n[i,j] = projection_norm
    # Populate matrix S using the projection norms
    # for idx, (i, j) in enumerate(centers):
    #     S[i, j] = projection_norms[idx]
    scoringMat = S.copy()
    scoringMat[:obj_sz // 2, :] = 0
    scoringMat[:, :obj_sz // 2] = 0
    scoringMat[-obj_sz // 2:, :] = 0
    scoringMat[:, -obj_sz // 2:] = 0


    idxRow = np.arange(scoringMat.shape[0])
    idxCol = np.arange(scoringMat.shape[1])
    cnt = 0
    pMax = 1

    while pMax > 0:
        # Find the maximum value and its location in the scoring matrix
        pMax = np.max(scoringMat)
        if pMax <= 0:
            break
        I = np.argmax(scoringMat)
        i_row, i_col = np.unravel_index(I, scoringMat.shape)


        # Define the bounding box limits for row and column indices
        row_start = max(i_row - rDelAlgorithm, 0)
        row_end = min(i_row + rDelAlgorithm, scoringMat.shape[0])
        col_start = max(i_col - rDelAlgorithm, 0)
        col_end = min(i_col + rDelAlgorithm, scoringMat.shape[1])

        # Zero out the region corresponding to the bounding box
        scoringMat[row_start:row_end+1, col_start:col_end+1] = 0

        # Debug print: count zeroed-out pixels
        num_zeroed_out = np.sum(scoringMat == 0)
        print(f"Number of zeroed-out pixels: {num_zeroed_out}")

        cnt += 1

        # Append peak value and location
        peaks.append(pMax)
        peaks_loc.append([i_row, i_col])

    # Convert lists to NumPy arrays
    peaks = np.array(peaks)
    peaks_loc = np.array(peaks_loc)

    return peaks, peaks_loc, S


def peak_algorithm(img, basis, sideLengthAlgorithm):
    """
    Identify peaks in an image using convolution with basis functions.

    Parameters:
    img                : Input image.
    basis              : Array of basis functions.
    sideLengthAlgorithm: Length parameter for peak extraction.

    Returns:
    peaks     : List of peak values.
    peaks_loc : List of peak locations (coordinates).
    S         : Scoring map from convolution.
    """

    num_of_basis_functions = basis.shape[2]

    img_sz_max = max(img.shape[0], img.shape[1])
    rDelAlgorithm = round(sideLengthAlgorithm // 2)

    peaks = []
    peaks_loc = []
    obj_sz = basis.shape[0]
    basis_vectors = np.reshape(basis, (basis.shape[0] ** 2, basis.shape[2]))

    # Perform convolution and sum over basis functions
    img_shape = tf.reshape(img, [1, img.shape[0], img.shape[1], 1])
    img_shape = tf.cast(img_shape, tf.float32)
    flipped_basis = np.flip(np.flip(basis[:, :, -1], 0), 1)
    flipped_basis_shape = tf.reshape(flipped_basis, [flipped_basis.shape[0], flipped_basis.shape[1], 1, 1])
    flipped_basis_shape = tf.cast(flipped_basis_shape, tf.float32)
    S = tf.nn.conv2d(img_shape, flipped_basis_shape, strides=[1, 1, 1, 1], padding='SAME') ** 2
    for j in range(basis.shape[2] - 1):
        flipped_basis = np.flip(np.flip(basis[:, :, j], 0), 1)
        flipped_basis_shape = tf.reshape(flipped_basis, [flipped_basis.shape[0], flipped_basis.shape[1], 1, 1])
        flipped_basis_shape = tf.cast(flipped_basis_shape, tf.float32)
        S += tf.nn.conv2d(img_shape, flipped_basis_shape, strides=[1, 1, 1, 1], padding='SAME') ** 2
    S = tf.squeeze(S).numpy()

    # Copy the scoring matrix and apply boundary conditions
    scoringMat = S.copy()
    img_copy = img.copy()
    scoringMat[:obj_sz // 2, :] = 0
    scoringMat[:, :obj_sz // 2] = 0
    scoringMat[-obj_sz // 2:, :] = 0
    scoringMat[:, -obj_sz // 2:] = 0

    cnt = 0
    pMax = 1

    while pMax > 0:
        # Find the maximum value and its location in the scoring matrix
        pMax = np.max(scoringMat)
        if pMax <= 0:
            break
        I = np.argmax(scoringMat)
        i_row, i_col = np.unravel_index(I, scoringMat.shape)

        # Create a circular mask to delete a "ball" around the peak
        rows, cols = np.ogrid[:scoringMat.shape[0], :scoringMat.shape[1]]
        mask = (rows - i_row) ** 2 + (cols - i_col) ** 2 <= rDelAlgorithm ** 2
        mask_circ = ((rows - i_row) ** 2 + (cols - i_col) ** 2 >= rDelAlgorithm ** 2 - 50) & \
                    ((rows - i_row) ** 2 + (cols - i_col) ** 2 <= rDelAlgorithm ** 2)
        # Zero out the circular region (ball) in the scoring matrix
        scoringMat[mask] =0
        img_copy[mask_circ] = 3000

        cnt += 1

        # Append peak value and location
        peaks.append(pMax)
        peaks_loc.append([i_row, i_col])

    # Convert lists to NumPy arrays
    peaks = np.array(peaks)
    peaks_loc = np.array(peaks_loc)

    return peaks, peaks_loc, S




def names_sub_folder(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]


def projected_noise_simulation_from_noise_patches(noise_samples, basis, num_of_exp_noise):
    """
    Simulates projected noise using given noise samples and basis functions in batches.

    Parameters:
    noise_samples : Array of noise samples.
    basis         : Array of basis functions.
    num_of_exp_noise : Number of noise experiments.

    Returns:
    S_z : Simulated noise projections.
    """

    noise_samples = noise_samples.astype(np.float32)
    basis = basis.astype(np.float32)

    sz = int(np.sqrt(noise_samples.shape[0]))  # Image size
    batch_size = max(1, num_of_exp_noise // 10)  # Define batch size (at least 1)
    num_batches = (num_of_exp_noise + batch_size - 1) // batch_size  # Compute number of batches

    # Compute output size
    sz_pn = tf.nn.conv2d(
        tf.reshape(noise_samples[:, 0], (1, sz, sz, 1)),
        tf.reshape(basis[:, :, 0], (basis.shape[0], basis.shape[1], 1, 1)),
        strides=[1, 1, 1, 1],
        padding='VALID'
    ).shape[1]

    S_z_n = np.zeros((num_of_exp_noise, sz_pn, sz_pn), dtype=np.float32)  # Initialize output
    flipped_basis_stack = tf.convert_to_tensor(
    np.stack([np.flip(np.flip(basis[:, :, j], 0), 1) for j in range(basis.shape[2])]),
    dtype=tf.float32
)

# Reshape to match TensorFlow's expected format: [filter_height, filter_width, in_channels, out_channels]
    flipped_basis_stack = tf.reshape(flipped_basis_stack, [basis.shape[0], basis.shape[1], 1, basis.shape[2]])

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, num_of_exp_noise)

        print(f"Processing batch {batch_idx+1}/{num_batches}, size: {end - start}")

        # Extract batch
        noise_batch = noise_samples[:,start:end]
        noise_imgs = np.reshape(noise_batch, (end - start, sz, sz, 1)).astype(np.float32)
        noise_imgs_tf = tf.convert_to_tensor(noise_imgs, dtype=tf.float32)

        # for j in range(basis.shape[2]):
        #     flipped_basis = np.flip(np.flip(basis[:, :, j], 0), 1)  # Flip basis
        #     flipped_basis_tf = tf.convert_to_tensor(
        #         tf.reshape(flipped_basis, (basis.shape[0], basis.shape[1], 1, 1)),
        #         dtype=tf.float32
        #     )

        # # Perform convolution
        # conv_result = tf.nn.conv2d(noise_imgs_tf, flipped_basis_tf, strides=[1, 1, 1, 1], padding='VALID')**2

        conv_result = tf.nn.conv2d(noise_imgs_tf, flipped_basis_stack, strides=[1, 1, 1, 1], padding='VALID') ** 2
         # Sum over all basis functions
        conv_result = tf.reduce_sum(conv_result, axis=-1)  # Correct summation over all basis functions
        # Store results in corresponding index
        S_z_n[start:end,:,:] += tf.squeeze(conv_result).numpy()

        # Clear GPU memory after processing each batch
        del noise_imgs_tf, conv_result
        tf.keras.backend.clear_session()

    S_z_n = np.transpose(S_z_n, (1, 2, 0))
    return S_z_n



def projected_noise_simulation_from_noise_patches_para(noise_samples, basis, num_of_exp_noise):
    """
    Simulates projected noise using given noise samples and basis functions in batches,
    fully optimized for GPU execution.

    Parameters:
    noise_samples : np.ndarray
        Array of noise samples.
    basis : np.ndarray
        Array of basis functions.
    num_of_exp_noise : int
        Number of noise experiments.

    Returns:
    np.ndarray
        Simulated noise projections.
    """

    noise_samples = noise_samples.astype(np.float32)
    basis = basis.astype(np.float32)

    sz = int(np.sqrt(noise_samples.shape[0]))  # Image size

    # ✅ Dynamically adjust batch size based on available GPU memory
    def get_optimal_batch_size(base_size=500):
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

    print(f"Using GPU acceleration. Batch size: {batch_size}")

    # Compute output size
    sz_pn = tf.nn.conv2d(
        tf.reshape(noise_samples[:, 0], (1, sz, sz, 1)),
        tf.reshape(basis[:, :, 0], (basis.shape[0], basis.shape[1], 1, 1)),
        strides=[1, 1, 1, 1],
        padding='VALID'
    ).shape[1]

    S_z_n = np.zeros((num_of_exp_noise, sz_pn, sz_pn), dtype=np.float32)

    # ✅ Precompute flipped basis stack on GPU
    with tf.device('/GPU:0'):
        flipped_basis_stack = tf.convert_to_tensor(
            np.stack([np.flip(np.flip(basis[:, :, j], 0), 1) for j in range(basis.shape[2])]),
            dtype=tf.float32
        )
        flipped_basis_stack = tf.reshape(flipped_basis_stack, [basis.shape[0], basis.shape[1], 1, basis.shape[2]])

    # ✅ TensorFlow compiled function for fast GPU execution
    @tf.function
    def process_batch(noise_imgs_tf, flipped_basis_stack):
        conv_result = tf.nn.conv2d(noise_imgs_tf, flipped_basis_stack, strides=[1, 1, 1, 1], padding='VALID') ** 2
        conv_result = tf.reduce_sum(conv_result, axis=-1)  # Sum over all basis functions
        return conv_result

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, num_of_exp_noise)

        print(f"Processing batch {batch_idx+1}/{num_batches}, size: {end - start}")

        with tf.device('/GPU:0'):
            noise_batch = noise_samples[:, start:end]
            noise_imgs = np.reshape(noise_batch, (end - start, sz, sz, 1)).astype(np.float32)
            noise_imgs_tf = tf.convert_to_tensor(noise_imgs, dtype=tf.float32)

            # Perform convolution on GPU
            conv_result = process_batch(noise_imgs_tf, flipped_basis_stack)

            # Store results in corresponding index
            S_z_n[start:end, :, :] = conv_result.numpy()

            # Clear GPU memory
            del noise_imgs_tf, conv_result
            tf.keras.backend.clear_session()

    return np.transpose(S_z_n, (1, 2, 0))  # Match expected output shape



def projected_noise_simulation_from_noise_patches_para_fast(noise_samples, basis, num_of_exp_noise):
    """
    Simulates projected noise using given noise samples and basis functions in batches,
    distributing computation across multiple GPUs (if available).
    """

    noise_samples = noise_samples.astype(np.float32)
    basis = basis.astype(np.float32)
    sz = int(np.sqrt(noise_samples.shape[0]))  # Image size

    # Determine optimal batch size based on available GPU memory
    def get_optimal_batch_size(base_size=1000):
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            batch_size = max(1, int(free_memory / (sz * sz * 4 * 2)))  # Estimate batch size
            print(f"Adjusted batch size: {batch_size}")
            return batch_size
        return base_size

    batch_size = get_optimal_batch_size()
    num_batches = (num_of_exp_noise + batch_size - 1) // batch_size  # Compute number of batches

    # List available GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        raise RuntimeError("No GPUs found! Consider using a CPU-based approach.")
    print(f"Using {num_gpus} GPUs for parallel processing. Batch size: {batch_size}")

    # Compute output size
    noise_sample_tensor = torch.tensor(noise_samples[:, 0], dtype=torch.float32).view(1, 1, sz, sz).to(device)
    basis_tensor = torch.tensor(basis[:, :, 0], dtype=torch.float32).view(1, 1, basis.shape[0], basis.shape[1]).to(device)
    sz_pn = F.conv2d(noise_sample_tensor, basis_tensor, stride=1, padding=0).shape[-1]

    # Initialize output tensor
    S_z_n = np.zeros((num_of_exp_noise, sz_pn, sz_pn), dtype=np.float32)

    # Flip basis stack
    flipped_basis_stack = np.stack([np.flip(np.flip(basis[:, :, j], 0), 1) for j in range(basis.shape[2])])
    flipped_basis_stack = torch.tensor(flipped_basis_stack, dtype=torch.float32).view(basis.shape[2], 1, basis.shape[0], basis.shape[1]).to(device)

    # Function to process a batch
    def process_batch(noise_imgs_tensor, flipped_basis_stack):
        conv_result = F.conv2d(noise_imgs_tensor, flipped_basis_stack, stride=1, padding=0) ** 2
        conv_result = torch.sum(conv_result, dim=1)  # Sum over all basis functions
        return conv_result

    results = [None] * num_batches

    # Process batches
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, num_of_exp_noise)

        gpu_idx = batch_idx % num_gpus
        device_name = f"cuda:{gpu_idx}"
        print(f"Processing batch {batch_idx+1}/{num_batches} on {device_name}")

        noise_batch = noise_samples[:, start:end]
        noise_imgs = np.reshape(noise_batch, (end - start, sz, sz, 1)).astype(np.float32)
        noise_imgs_tensor = torch.tensor(noise_imgs, dtype=torch.float32).permute(0, 3, 1, 2).to(device)

        with torch.no_grad():  # Disable gradient calculations for faster computation
            results[batch_idx] = process_batch(noise_imgs_tensor, flipped_basis_stack).cpu().numpy()

        # Clear GPU memory
        del noise_imgs_tensor
        torch.cuda.empty_cache()

    # Collect results
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, num_of_exp_noise)
        S_z_n[start:end, :, :] += results[batch_idx]

    return np.transpose(S_z_n, (1, 2, 0))  # Match expected output shape


def projected_noise_simulation_from_noise_patches_para_fast_cpu(noise_samples, basis, num_of_exp_noise):
    """
    Simulates projected noise using given noise samples and basis functions in batches,
    distributing computation across multiple CPUs.
    """

    # Ensure using CPU
    device = torch.device('cpu')

    # Convert to PyTorch tensors and move to CPU
    noise_samples = torch.tensor(noise_samples, dtype=torch.float32, device=device)
    basis = torch.tensor(basis, dtype=torch.float32, device=device)

    sz = int(np.sqrt(noise_samples.shape[0]))  # Image size

    # Dynamically determine batch size based on available memory
    def get_optimal_batch_size(base_size=1000):
        return base_size  # Default batch size for CPU

    batch_size = get_optimal_batch_size()
    num_batches = (num_of_exp_noise + batch_size - 1) // batch_size  # Compute number of batches

    print(f"Using CPU for processing. Batch size: {batch_size}")

    # Compute output size
    example_image = noise_samples[:, 0].reshape(1, 1, sz, sz)
    example_basis = basis[:, :, 0].reshape(1, 1, basis.shape[0], basis.shape[1])
    sz_pn = F.conv2d(example_image, example_basis, stride=1, padding=0).shape[-1]

    # Initialize output tensor
    S_z_n = torch.zeros((num_of_exp_noise, sz_pn, sz_pn), dtype=torch.float32, device=device)

    # Flip basis stack and reshape for PyTorch convolution
    flipped_basis_stack = torch.stack([torch.flip(basis[:, :, j], [0, 1]) for j in range(basis.shape[2])], dim=0)
    flipped_basis_stack = flipped_basis_stack.unsqueeze(1)  # [num_filters, 1, H, W]

    # Process batches
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, num_of_exp_noise)

        print(f"Processing batch {batch_idx+1}/{num_batches} on CPU")

        # Extract batch and reshape
        noise_batch = noise_samples[:, start:end]
        noise_imgs = noise_batch.reshape(end - start, 1, sz, sz)

        # Perform convolution and sum over basis functions
        conv_result = F.conv2d(noise_imgs, flipped_basis_stack, stride=1, padding=0) ** 2
        conv_result = torch.sum(conv_result, dim=1)  # Sum over all basis functions

        # Store results in corresponding index
        S_z_n[start:end, :, :] = conv_result

    return S_z_n.permute(1, 2, 0).cpu().numpy()  # Match expected output shape and move to NumPy

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

import torch
import numpy as np

import torch
import numpy as np


def projected_noise_simulation_from_noise_patches_torch(noise_samples, basis, num_of_exp_noise):
    """
    Simulates projected noise using given noise samples and basis functions using PyTorch.

    Parameters:
    noise_samples : Array of noise samples.
    basis         : Array of basis functions.
    num_of_exp_noise : Number of noise experiments.

    Returns:
    S_z : Simulated noise projections.
    """
    # Convert inputs to float32 for consistent precision
    noise_samples = noise_samples.astype(np.float32)
    basis = basis.astype(np.float32)

    sz = int(np.sqrt(noise_samples.shape[0]))  # Image size
    noise_imgs = noise_samples.reshape(
        (num_of_exp_noise, 1, sz, sz))  # Reshape noise samples to 4D (batch, channels, height, width)

    # Convert to PyTorch tensors
    noise_imgs_torch = torch.tensor(noise_imgs, dtype=torch.float32)

    # Flip basis stack to be consistent with TensorFlow's behavior
    flipped_basis_stack = np.stack([np.flip(np.flip(basis[:, :, j], axis=0), axis=1)
                                    for j in range(basis.shape[2])])
    flipped_basis_stack = torch.tensor(flipped_basis_stack, dtype=torch.float32)
    flipped_basis_stack = flipped_basis_stack.unsqueeze(1)  # (out_channels, in_channels, H, W)

    # Perform convolution using torch.nn.functional.conv2d
    conv_result = torch.nn.functional.conv2d(noise_imgs_torch, flipped_basis_stack, padding=0)
    conv_result = conv_result ** 2  # Square the convolution results (as in TensorFlow)
    conv_result = torch.sum(conv_result, dim=1)  # Sum over the channel dimension

    # Convert back to NumPy and transpose to match TensorFlow output shape
    S_z_n = conv_result.detach().numpy()
    S_z_n = np.transpose(S_z_n, (1, 2, 0))

    return S_z_n


def projected_noise_simulation_from_noise_patches_scipy(noise_samples, basis, num_of_exp_noise):
    """
    Simulates projected noise using given noise samples and basis functions using SciPy's FFT for speed.

    Parameters:
    noise_samples : Array of noise samples (NumPy array).
    basis         : Array of basis functions (NumPy array).
    num_of_exp_noise : Number of noise experiments.

    Returns:
    S_z : Simulated noise projections.
    """
    # Convert inputs to float32 for consistent precision
    noise_samples = noise_samples.astype(np.float32)
    basis = basis.astype(np.float32)

    # Reshape noise samples to 4D (num_of_exp_noise, height, width, channels)
    sz = int(np.sqrt(noise_samples.shape[0]))  # Image size
    noise_imgs = noise_samples.reshape((num_of_exp_noise, sz, sz, 1))

    # Flip the basis functions to be consistent with TensorFlow's behavior
    flipped_basis_stack = np.stack([np.flip(np.flip(basis[:, :, j], axis=0), axis=1)
                                    for j in range(basis.shape[2])])

    # Determine output size using a single convolution
    example_conv = fftconvolve(noise_imgs[0, :, :, 0], flipped_basis_stack[0, :, :], mode='valid')
    sz_pn = example_conv.shape[0]

    # Initialize the output array
    S_z = np.zeros((num_of_exp_noise, sz_pn, sz_pn), dtype=np.float32)

    # Perform convolution for each basis function and noise image using FFT
    for j in range(basis.shape[2]):
        for i in range(num_of_exp_noise):
            conv_result = fftconvolve(noise_imgs[i, :, :, 0], flipped_basis_stack[j, :, :], mode='valid')
            S_z[i, :, :] += conv_result   # Sum of squared convolutions

    # Transpose to match the expected output shape (height, width, num_of_exp_noise)
    S_z = np.transpose(S_z, (1, 2, 0))
    return S_z


import tensorflow as tf

def projected_noise_simulation_from_noise_patches_tf(noise_samples, basis, num_of_exp_noise):
    """
    Simulates projected noise using given noise samples and basis functions.

    Parameters:
    noise_samples : Array of noise samples.
    basis         : Array of basis functions.
    num_of_exp_noise : Number of noise experiments.

    Returns:
    S_z : Simulated noise projections.
    """
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


def coortodist2(ind, sz):
    """
    Calculates the distance matrix and related data from coordinate indices.

    Parameters:
    ind : List or array of indices.
    sz  : Size of the square grid.

    Returns:
    coortodistData : A dictionary containing:
        - mat_of_radii  : Matrix of radii (distances) between index pairs.
        - dist_vec_uniq : Unique distances vector.
        - i_d_vec       : Indices of unique distances.
        - i_d_mat       : Matrix form of unique indices.
    """

    # Create a grid of row and column indices
    coords = np.array(np.unravel_index(ind, (sz, sz))).T

    # Calculate pairwise distances using broadcasting
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    mat_of_radii = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Extract unique distances and their indices
    dist_vec_uniq, i_d_vec, i_d_mat = np.unique(mat_of_radii.ravel(), return_index=True, return_inverse=True)

    coortodistData = {
        'mat_of_radii': mat_of_radii,
        'dist_vec_uniq': dist_vec_uniq,
        'i_d_vec': i_d_vec,
        'i_d_mat': i_d_mat.reshape(mat_of_radii.shape)
    }

    return coortodistData
@jit(nopython=True, parallel=True)
def calculate_pairwise_distances(coords):
    sz = coords.shape[0]
    mat_of_radii = np.zeros((sz, sz), dtype=np.float64)
    for i in prange(sz):
        for j in range(sz):
            diff = coords[i] - coords[j]
            mat_of_radii[i, j] = np.sqrt(np.sum(diff ** 2))
    return mat_of_radii

def coortodist3(ind, sz):
    """
    Calculates the distance matrix and related data from coordinate indices.

    Parameters:
    ind : List or array of indices.
    sz  : Size of the square grid.

    Returns:
    coortodistData : A dictionary containing:
        - mat_of_radii  : Matrix of radii (distances) between index pairs.
        - dist_vec_uniq : Unique distances vector.
        - i_d_vec       : Indices of unique distances.
        - i_d_mat       : Matrix form of unique indices.
    """

    # Create a grid of row and column indices
    coords = np.array(np.unravel_index(ind, (sz, sz))).T

    # Calculate pairwise distances using Numba for parallel computation
    mat_of_radii = calculate_pairwise_distances(coords)

    # Extract unique distances and their indices
    dist_vec_uniq, i_d_vec, i_d_mat = np.unique(mat_of_radii.ravel(), return_index=True, return_inverse=True)

    coortodistData = {
        'mat_of_radii': mat_of_radii,
        'dist_vec_uniq': dist_vec_uniq,
        'i_d_vec': i_d_vec,
        'i_d_mat': i_d_mat.reshape(mat_of_radii.shape)
    }

    return coortodistData
def coortodist(ind, sz):
    """
    Calculates the distance matrix and related data from coordinate indices.

    Parameters:
    ind : List or array of indices.
    sz  : Size of the square grid.

    Returns:
    coortodistData : A dictionary containing:
        - mat_of_radii  : Matrix of radii (distances) between index pairs.
        - dist_vec_uniq : Unique distances vector.
        - i_d_vec       : Indices of unique distances.
        - i_d_mat       : Matrix form of unique indices.
    """

    # Create a grid of row and column indices
    coords = np.array(np.unravel_index(ind, (sz, sz))).T

    # Use SciPy to calculate pairwise Euclidean distances efficiently
    mat_of_radii = squareform(pdist(coords, 'euclidean'))

    # Extract unique distances and their ipeak_algorithm_cont_maskndices
    dist_vec_uniq, i_d_vec, i_d_mat = np.unique(mat_of_radii.ravel(), return_index=True, return_inverse=True)

    coortodistData = {
        'mat_of_radii': mat_of_radii,
        'dist_vec_uniq': dist_vec_uniq,
        'i_d_vec': i_d_vec,
        'i_d_mat': i_d_mat.reshape(mat_of_radii.shape)
    }

    return coortodistData
def spatial_cov_from_radial_cov(radial_cov, r, mat_of_radii, dist_vec_uniq, i_d_mat):
    """
    Converts radial covariance into a spatial covariance matrix.

    Parameters:
    radial_cov    : Radial covariance values.
    r             : Radii corresponding to radial covariance.
    mat_of_radii  : Matrix of radii.
    dist_vec_uniq : Unique distance vector.
    i_d_mat       : Index matrix for distances.

    Returns:
    noise_cov : Spatial covariance matrix.
    """

    # Interpolate radial covariance
    radial_cov_interp_func = interp1d(r, radial_cov, kind='linear', fill_value=radial_cov[-1] * 0.1, bounds_error=False)
    radial_cov_interp = radial_cov_interp_func(dist_vec_uniq)

    noise_cov_vec_uniq = np.zeros_like(dist_vec_uniq)

    # Fill in the interpolated covariance values
    for i in range(len(noise_cov_vec_uniq)):
        if dist_vec_uniq[i] > r[-1]:
            noise_cov_vec_uniq[i:] = 0
            break
        noise_cov_vec_uniq[i] = radial_cov_interp[i]

    # Reshape into the spatial covariance matrix
    noise_cov = noise_cov_vec_uniq[i_d_mat].reshape(mat_of_radii.shape)

    return noise_cov


def rotate_image(image, angle):
    """
    Rotate a 2D numpy array (image) by a specified angle.

    :param image: 2D numpy array representing the image.
    :param angle: Angle in degrees to rotate the image.
    :return: Rotated image as a 2D numpy array.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")
    if not isinstance(angle, (int, float)):
        raise ValueError("Angle must be a number.")

    rotated_image = rotate(image, angle, reshape=False)
    return rotated_image


def bsearch(x, LowerBound, UpperBound):
    if LowerBound > x[-1] or UpperBound < x[0] or UpperBound < LowerBound:
        return None, None

    lower_index_a = 0
    lower_index_b = len(x) - 1
    upper_index_a = 0
    upper_index_b = len(x) - 1

    while (lower_index_a + 1 < lower_index_b) or (upper_index_a + 1 < upper_index_b):
        lw = (lower_index_a + lower_index_b) // 2

        if x[lw] >= LowerBound:
            lower_index_b = lw
        else:
            lower_index_a = lw
            if lw > upper_index_a and lw < upper_index_b:
                upper_index_a = lw

        up = (upper_index_a + upper_index_b + 1) // 2
        if x[up] <= UpperBound:
            upper_index_a = up
        else:
            upper_index_b = up
            if up < lower_index_b and up > lower_index_a:
                lower_index_b = up

    lower_index = lower_index_a if x[lower_index_a] >= LowerBound else lower_index_b
    upper_index = upper_index_b if x[upper_index_b] <= UpperBound else upper_index_a

    if upper_index < lower_index:
        return None, None

    return lower_index, upper_index


def cryo_epsdR(vol, samples_idx, max_d=None, verbose=0):
    p = vol.shape[0]
    if vol.shape[1] != p:
        raise ValueError('vol must be a stack of square images')

    if vol.ndim > 3:
        raise ValueError('vol must be a 3D array')

    K = 1 if vol.ndim == 2 else vol.shape[2]

    if max_d is None:
        max_d = p - 1

    max_d = min(max_d, p - 1)

    I, J = np.meshgrid(np.arange(max_d + 1), np.arange(max_d + 1))
    dists = I ** 2 + J ** 2
    dsquare = np.unique(dists[dists <= max_d ** 2])

    corrs = np.zeros(dsquare.shape[0])
    corrcount = np.zeros(dsquare.shape[0])
    x = np.sqrt(dsquare)

    distmap = -np.ones(dists.shape, dtype=int)
    for i in range(max_d + 1):
        for j in range(max_d + 1):
            d = i ** 2 + j ** 2
            if d <= max_d ** 2:
                lower_idx, upper_idx = bsearch(dsquare, d - 1.0e-13, d + 1.0e-13)
                if lower_idx is None or upper_idx is None or lower_idx != upper_idx:
                    raise ValueError('Something went wrong')
                distmap[i, j] = lower_idx

    validdists = np.where(distmap != -1)

    mask = np.zeros((p, p), dtype=bool)
    mask.flat[samples_idx] = True
    tmp = np.zeros((2 * p + 1, 2 * p + 1))
    tmp[:p, :p] = mask
    ftmp = fft2(tmp)
    c = np.real(ifft2(ftmp * np.conj(ftmp)))
    c = np.round(c[:max_d + 1, :max_d + 1]).astype(int)

    R = np.zeros(corrs.shape)

    for k in range(K):
        proj = vol[:, :, k] if K > 1 else vol

        samples = np.zeros((p, p))
        samples.flat[samples_idx] = proj.flat[samples_idx]

        tmp = np.zeros((2 * p + 1, 2 * p + 1))
        tmp[:p, :p] = samples
        ftmp = fft2(tmp)
        s = np.real(ifft2(ftmp * np.conj(ftmp)))
        s = s[:max_d + 1, :max_d + 1]

        for currdist in np.nditer(validdists):
            dmidx = distmap[tuple(currdist)]
            corrs[dmidx] += s[currdist]
            corrcount[dmidx] += c[currdist]

    idx = corrcount != 0
    R[idx] = corrs[idx] / corrcount[idx]
    cnt = corrcount[idx]

    x = x[idx]
    R = R[idx]

    return R, x, cnt


def estimate_noise_radial_cov_2(noise_img_scaled, box_sz, num_of_patches):
    rDelAlgorithm = int(box_sz / 2)+ 1
    box_sz_var = int(box_sz)

    Y_no_mean = noise_img_scaled - np.mean(noise_img_scaled)
    var_box = np.ones((box_sz_var, box_sz_var))
    Y_var = convolve2d(Y_no_mean ** 2, var_box, mode='same')
    Y_var[:int(box_sz), :] = -np.inf
    Y_var[:, :int(box_sz)] = -np.inf
    Y_var[-int(box_sz):, :] = -np.inf
    Y_var[:, -int(box_sz):] = -np.inf
    Y_var_sz = Y_var.shape[0]
    idxRow = np.arange(Y_var.shape[0])
    idxCol = np.arange(Y_var.shape[1])
    noise_mean = 0
    cnt_p = 0
    p_max = 1
    cnt = 1
    mean_patches = []
    noise_patches = []

    while p_max > 0:
        p_max = np.max(Y_var)
        if p_max < 0:
            break

        I = np.argmax(Y_var)
        i_row, i_col = np.unravel_index(I, Y_var.shape)

        idxRowCandidate = np.zeros(Y_var_sz, dtype=bool)
        idxColCandidate = np.zeros(Y_var_sz, dtype=bool)
        idxRowCandidate[max(i_row - rDelAlgorithm, 0):min(i_row + rDelAlgorithm, Y_var_sz)] = True
        idxColCandidate[max(i_col - rDelAlgorithm, 0):min(i_col + rDelAlgorithm, Y_var_sz)] = True

        tmp = noise_img_scaled[i_row - rDelAlgorithm:i_row + rDelAlgorithm,
              i_col - rDelAlgorithm:i_col + rDelAlgorithm]
        if cnt > num_of_patches:
            patch = noise_img_scaled[i_row - rDelAlgorithm:i_row + rDelAlgorithm,
                    i_col - rDelAlgorithm:i_col + rDelAlgorithm]
            noise_patches.append(patch - np.mean(tmp))
            mean_patches.append(np.mean(tmp))
            cnt_p += 1

        Y_var[np.ix_(idxRow[idxRowCandidate], idxCol[idxColCandidate])] = -np.inf
        cnt += 1

    noise_mean = np.mean(mean_patches)
    max_d = int(np.floor(0.8 * noise_patches[0].shape[0]))
    p = noise_patches[0].shape[0]
    noise_patches_arr = np.array(noise_patches), range(p ** 2)
    noise_patches_arr = noise_patches_arr[0]
    noise_patches_swaped  = np.transpose(noise_patches_arr, (2, 1, 0))
    samples_idx = np.array(range(p ** 2))
    radial_cov, r, cnt = cryo_epsdR(noise_patches_swaped, samples_idx, max_d)

    return radial_cov, r, noise_mean


def rotate_image(image, angle):
    """
    Rotate a 2D numpy array (image) by a specified angle.

    :param image: 2D numpy array representing the image.
    :param angle: Angle in degrees to rotate the image.
    :return: Rotated image as a 2D numpy array.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")
    if not isinstance(angle, (int, float)):
        raise ValueError("Angle must be a number.")

    rotated_image = rotate(image, angle, reshape=False)
    return rotated_image

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
def zero_out_far_values(matrix):
    matrix = matrix.copy()  # Create a writable copy
    n = matrix.shape[0]
    center = n // 2
    max_distance = n / 2

    # Create a grid of indices
    y, x = np.ogrid[:n, :n]

    # Calculate distances from the center
    distances = np.sqrt((x - center) ** 2 + (y - center) ** 2)

    # Zero out elements that are farther than max_distance
    matrix[distances > max_distance] = 0

    return matrix

def test_basis_on_rotated_image(class_averages, steerable_basis_vectors, obj_sz):
    n_img = 0
    angle_rot = 90
    img = Image(class_averages[n_img])
    scaled_img = img.downsample(obj_sz)
    scaled_img = scaled_img.asnumpy()[0]
    scaled_img_rotated = rotate_image(scaled_img, angle_rot)
    norm_rot = np.linalg.norm(np.dot(np.transpose(steerable_basis_vectors), scaled_img_rotated.flatten()))
    norm_reg = np.linalg.norm(np.dot(np.transpose(steerable_basis_vectors), scaled_img.flatten()))
    norm_rot_err = np.abs(norm_rot-norm_reg)/norm_reg
    return norm_rot_err
import numpy as np


def sort_steerable_basis(class_averages_vectors, steerable_basis_vectors,gamma,M):
    """
    Sort steerable basis vectors based on the squared coefficients computed from class averages.

    Parameters:
    class_averages_vectors : ndarray
        2D array where each column represents a flattened class average.
    steerable_basis_vectors : ndarray
        2D array of orthonormal basis vectors with the same size as class_averages_vectors.

    Returns:
    sorted_basis_vectors : ndarray
        Steerable basis vectors sorted based on the squared coefficients.
    """
    # Compute the coefficients for each class average vector
    coefficients = np.dot(steerable_basis_vectors.T, class_averages_vectors)

    # Square the coefficients to emphasize the contribution of each basis vector
    coefficients_sqr = coefficients ** 2

    # Initialize projected norm vector
    projected_norm = np.zeros(class_averages_vectors.shape[1])

    # Calculate norms of the class_averages_vectors
    class_norms = np.linalg.norm(class_averages_vectors, axis=0)

    # List to keep track of the selected vector indices
    selected_indices = []

    ### Step 1: Initial Selection
    # Find the column with the lowest class norm
    min_class_norm_idx = np.argmin(class_norms)

    # Find the index of the highest coefficient in this column
    tmp = coefficients_sqr[:, min_class_norm_idx]
    initial_vector_idx = np.argmax(tmp)

    # Append the initial vector index to the selected list
    selected_indices.append(initial_vector_idx)

    # Update projected norm vector with the contributions of the initial vector
    projected_norm += coefficients_sqr[initial_vector_idx, :]

    # Set the selected coefficients to zero to avoid re-selection
    coefficients_sqr[initial_vector_idx, :] = 0

    ### Step 2: Iterative Selection
    # Loop until the condition is met
    while len(selected_indices) < M:
        #np.min(projected_norm) < gamma * np.min(class_norms)**2
        # Find the column with the lowest projected norm
        min_proj_norm_idx = np.argmin(projected_norm)

        # Find the index of the highest coefficient in this column
        tmp = coefficients_sqr[:, min_proj_norm_idx]
        best_vector_idx = np.argmax(tmp)

        # Append the best vector index to the selected list
        selected_indices.append(best_vector_idx)

        # Update projected norm vector with the contributions of the selected vector
        projected_norm += coefficients_sqr[best_vector_idx, :]

        # Set the selected coefficients to zero to avoid re-selection
        coefficients_sqr[best_vector_idx, :] = 0

    # Create the sorted basis vectors
    sorted_basis_vectors = steerable_basis_vectors[:, selected_indices]

    return sorted_basis_vectors

def sort_steerable_basis_new(basis_full, objects, noise_patches, max_dimension):

    coeff_objects = np.square((basis_full.T)@objects)
    coeff_noise = np.square((basis_full.T)@noise_patches)

    # first compute the full projected_snr
    projected_norm_objects = coeff_objects.sum(axis=0)
    projected_norm_noise = coeff_noise.sum(axis=0)
    projected_snr = projected_norm_objects/projected_norm_noise.mean()
    min_idx_projected_snr =  np.argmin(projected_snr)
    # second update projected_snr at each step
    projected_snr_per_dim = []
    sorted_basis_lst = []
    projected_norm_objects = np.zeros(projected_norm_objects.shape)
    projected_norm_noise = np.zeros(projected_norm_noise.shape)
    for dim in range(max_dimension):
        numerator = coeff_objects[:,min_idx_projected_snr]+projected_norm_objects[min_idx_projected_snr]
        denomerator = np.zeros(coeff_noise.shape[0])
        for n in range(coeff_noise.shape[1]):
            denomerator += coeff_noise[:,n] + projected_norm_noise[n]
        denomerator = denomerator/coeff_noise.shape[1]
        ratio = numerator/denomerator
        max_idx_candidate = np.argmax(ratio)
        projected_norm_objects += coeff_objects[max_idx_candidate,:]
        projected_norm_noise += coeff_noise[max_idx_candidate,:]
        sorted_basis_lst.append(basis_full[:,max_idx_candidate])
        coeff_objects = np.delete(coeff_objects,max_idx_candidate,axis=0)
        coeff_noise = np.delete(coeff_noise,max_idx_candidate,axis=0)
        projected_snr = projected_norm_objects/projected_norm_noise.mean()
        min_idx_projected_snr =  np.argmin(projected_snr)
        projected_snr_per_dim.append(projected_snr[min_idx_projected_snr])
    sorted_basis_vectors = np.column_stack(sorted_basis_lst)
    return sorted_basis_vectors,projected_snr_per_dim


def plot_projection_ratio_with_average_and_min(templates, noise_patches, basis_vectors):
    """
    Plot the ratios of average and minimum template projection norms to the mean noise projection norm
    as a function of the number of basis vectors. Return the number of basis functions for which
    each ratio is maximal.

    Parameters:
    templates : ndarray
        2D array where each column represents a flattened template (shape: [D, N]).
    noise_patches : ndarray
        2D array where each column represents a flattened noise patch (shape: [D, S]).
    basis_vectors : ndarray
        2D array of orthonormal basis vectors (shape: [D, M_max]).

    Returns:
    M_optimal_mean : int
        Number of basis functions for which the average ratio is maximal.
    M_optimal_min : int
        Number of basis functions for which the minimum ratio is maximal.
    """
    D, N = templates.shape
    _, S = noise_patches.shape
    M_max = basis_vectors.shape[1]

    # Initialize arrays to store results
    avg_template_norms = []
    min_template_norms = []
    mean_noise_norms = []
    ratios_mean = []
    ratios_min = []

    # Iterate over the number of basis vectors
    for m in range(1, M_max + 1):
        # Select the first `m` basis vectors
        current_basis = basis_vectors[:, :m]

        # Project templates and noise patches onto the current basis
        template_projections = np.dot(current_basis.T, templates)  # Shape: [m, N]
        noise_projections = np.dot(current_basis.T, noise_patches)  # Shape: [m, S]

        # Compute projection norms
        template_norms = np.sum(template_projections**2, axis=0)  # Norm for each template
        noise_norms = np.sum(noise_projections**2, axis=0)  # Norm for each noise patch

        # Compute metrics
        avg_template_norm = np.mean(template_norms)  # Average projection norm for templates
        min_template_norm = np.min(template_norms)  # Minimum projection norm for templates
        mean_noise_norm = np.mean(noise_norms)  # Mean projection norm for noise
        ratio_mean = avg_template_norm / mean_noise_norm
        ratio_min = min_template_norm / mean_noise_norm

        # Store results
        avg_template_norms.append(avg_template_norm)
        min_template_norms.append(min_template_norm)
        mean_noise_norms.append(mean_noise_norm)
        ratios_mean.append(ratio_mean)
        ratios_min.append(ratio_min)

    # Find the number of basis functions (M) where each ratio is maximal
    M_optimal_mean = np.argmax(ratios_mean) + 1  # Add 1 because M starts from 1
    M_optimal_min = np.argmax(ratios_min) + 1

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, M_max + 1), ratios_mean, marker='o', label='Avg Template Norm / Mean Noise Norm', color='b')
    plt.plot(range(1, M_max + 1), ratios_min, marker='x', label='Min Template Norm / Mean Noise Norm', color='g')
    plt.axhline(1, color='r', linestyle='--', label='Ratio = 1')
    plt.axvline(M_optimal_mean, color='b', linestyle='--', label=f'M_optimal_mean = {M_optimal_mean}')
    plt.axvline(M_optimal_min, color='g', linestyle='--', label=f'M_optimal_min = {M_optimal_min}')
    plt.xlabel('Number of Basis Vectors')
    plt.ylabel('Ratio')
    plt.title('Projection Norm Ratios as a Function of Basis Vectors')
    plt.legend()
    plt.grid(True)
    plt.show()

    return M_optimal_mean, M_optimal_min


def sort_steerable_basis_with_noise(class_averages_vectors, steerable_basis_vectors, noise_patches, M):
    """
    Sort steerable basis vectors based on the normalized projection norms computed from class averages and noise.

    Parameters:
    class_averages_vectors : ndarray
        2D array where each column represents a flattened class average (shape: [D, N]).
    steerable_basis_vectors : ndarray
        2D array of orthonormal basis vectors (shape: [D, M_max]).
    noise_patches : ndarray
        2D array where each column represents a flattened noise patch (shape: [D, S]).
    M : int
        Number of basis vectors to select.

    Returns:
    sorted_basis_vectors : ndarray
        Selected and sorted steerable basis vectors (shape: [D, M]).
    """
    # Compute initial coefficients for templates and noise patches
    template_coefficients = np.dot(steerable_basis_vectors.T, class_averages_vectors)# Shape: [M_max, N]
    proj = np.dot(template_coefficients, steerable_basis_vectors)
    temp = np.sum(proj)
    noise_coefficients = np.dot(steerable_basis_vectors.T, noise_patches)  # Shape: [M_max, S]

    # Initialize cumulative norms
    cumulative_template_norms = np.sum(template_coefficients ** 2, axis=0)  # Initial template norms
    cumulative_noise_norms = np.mean(noise_coefficients ** 2, axis=1)  # Initial noise norms (mean over noise patches)

    # Initialize variables
    selected_indices = []
    D, N = class_averages_vectors.shape
    M_max = steerable_basis_vectors.shape[1]

    for step in range(M):
        # Step 1: Find the template with the smallest projection norm
        f_min_idx = np.argmin(cumulative_template_norms)

        # Step 2: Select the basis vector that maximizes the normalized projection score
        remaining_indices = list(set(range(M_max)) - set(selected_indices))
        best_vector_idx = remaining_indices[np.argmax(
            (template_coefficients[remaining_indices, f_min_idx] ** 2) /
            cumulative_noise_norms[remaining_indices]
        )]

        # Append the selected basis vector to the list
        selected_indices.append(best_vector_idx)

        # Step 3: Update norms by removing the contribution of the selected basis vector
        best_vector = steerable_basis_vectors[:, best_vector_idx]
        template_proj = np.dot(best_vector.T, class_averages_vectors)  # Projection of templates
        noise_proj = np.dot(best_vector.T, noise_patches)  # Projection of noise patches

        # Update cumulative norms
        cumulative_template_norms -= template_proj ** 2
        cumulative_noise_norms -= np.mean(noise_proj ** 2, axis=0)  # Update using mean over noise patches

        # Prevent re-selection by nullifying the coefficients of the chosen vector
        template_coefficients[best_vector_idx, :] = 0
        noise_coefficients[best_vector_idx, :] = np.inf

    # Create the sorted basis vectors
    sorted_basis_vectors = steerable_basis_vectors[:, selected_indices]

    return sorted_basis_vectors

def sort_steerable_basis_with_noise_updated(class_averages_vectors, steerable_basis_vectors, noise_patches, M):
    """
    Sort steerable basis vectors based on the normalized projection norms computed from class averages and noise.

    Parameters:
    class_averages_vectors : ndarray
        2D array where each column represents a flattened class average (shape: [D, N]).
    steerable_basis_vectors : ndarray
        2D array of orthonormal basis vectors (shape: [D, M_max]).
    noise_patches : ndarray
        2D array where each column represents a flattened noise patch (shape: [D, S]).
    M : int
        Number of basis vectors to select.

    Returns:
    sorted_basis_vectors : ndarray
        Selected and sorted steerable basis vectors (shape: [D, M]).
    """
    # Compute initial coefficients for noise patches
    noise_coefficients = np.dot(steerable_basis_vectors.T, noise_patches)  # Shape: [M_max, S]

    # Initialize variables
    D, N = class_averages_vectors.shape
    _, S = noise_patches.shape
    M_max = steerable_basis_vectors.shape[1]
    selected_indices = []

    # Compute the initial norms of each template
    template_norms = np.sum(class_averages_vectors ** 2, axis=0)  # Shape: [N]
    cumulative_template_norms = np.zeros_like(template_norms)  # Initialize cumulative template norms
    cumulative_noise_norms = np.zeros(M_max)  # Initialize cumulative noise norms for each basis vector

    for step in range(M):
        # Step 1: Find the template with the smallest norm
        f_min_idx = np.argmin(template_norms)  # Weakest template index

        # Step 3: Select the basis vector that maximizes the cumulative projected SNR
        remaining_indices = list(set(range(M_max)) - set(selected_indices))

        # Precompute cumulative contributions for all remaining vectors
        best_vector_idx = remaining_indices[np.argmax([
            (
                # Cumulative Template Norm
                cumulative_template_norms[f_min_idx] +
                (np.dot(steerable_basis_vectors[:, j].T, class_averages_vectors[:, f_min_idx]) ** 2)
            ) /
            (
                # Cumulative Noise Norm
                cumulative_noise_norms[j] +
                np.mean((np.dot(steerable_basis_vectors[:, j].T, noise_patches)) ** 2)
            )
            for j in remaining_indices
        ])]


        # Append the selected basis vector to the list
        selected_indices.append(best_vector_idx)

        # Step 3: Update the template norms and cumulative noise norms
        best_vector = steerable_basis_vectors[:, best_vector_idx]
        template_proj = np.dot(best_vector.T, class_averages_vectors)  # Projection of templates onto the selected vector
        noise_proj = np.dot(best_vector.T, noise_patches)  # Projection of noise patches onto the selected vector

        # Update cumulative norms
        cumulative_template_norms += template_proj ** 2
        cumulative_noise_norms[best_vector_idx] += np.mean(noise_proj ** 2)

        # Update the norms of the templates
        template_norms -= template_proj ** 2

    # Create the sorted basis vectors
    sorted_basis_vectors = steerable_basis_vectors[:, selected_indices]

    return sorted_basis_vectors






def sort_steerable_basis2(class_averages, steerable_basis_vectors, obj_sz):
    coeff_vals = np.zeros((class_averages.shape[0], steerable_basis_vectors.shape[1]))

    for n_img in range(class_averages.shape[0]):
        img = Image(class_averages[n_img])
        scaled_img = img.downsample(obj_sz)
        scaled_img = scaled_img.asnumpy()[0]
        for i in range(steerable_basis_vectors.shape[1]):
            basis_vector = steerable_basis_vectors[:, i]
            coeff = np.dot(basis_vector.T, scaled_img.flatten())
            coeff_vals[n_img, i] = coeff**2
            #norm_err[n_img, i] = np.linalg.norm(scaled_img.flatten() - projection) / np.linalg.norm(scaled_img.flatten())

    max_coeff_vals = np.max(coeff_vals, axis=0)
    sorted_indices = np.argsort(max_coeff_vals)[::-1]
    sorted_steerable_basis_vectors = steerable_basis_vectors[:, sorted_indices]
    return sorted_steerable_basis_vectors

def remove_low_variance_elements(data, percentage=10):
    """
    Remove a specified percentage of elements with the lowest variance after flattening.

    Parameters:
    data (ndarray): The input array of shape (num_images, height, width).
    percentage (float): The percentage of elements to remove, based on variance.

    Returns:
    ndarray: The filtered array with low-variance elements removed.
    """
    # Flatten the images along the spatial dimensions
    flattened_data = data.reshape(data.shape[0], -1)

    # Calculate the variance for each flattened element
    variances = np.var(flattened_data, axis=1)

    # Determine the cutoff for the lowest percentage
    cutoff = np.percentile(variances, percentage)

    # Filter out elements with variance below the cutoff
    high_variance_indices = np.where(variances > cutoff)[0]

    # Return the filtered data
    filtered_data = data[high_variance_indices]
    return filtered_data, high_variance_indices







def dominent_steerable_basis(class_averages, sorted_steerable_basis_vectors, obj_sz):
    norm_err = np.zeros((class_averages.shape[0], sorted_steerable_basis_vectors.shape[1]))

    for n_img in range(class_averages.shape[0]):
        img = Image(class_averages[n_img])
        scaled_img = img.downsample(obj_sz)
        scaled_img = scaled_img.asnumpy()[0]
        for i in range(sorted_steerable_basis_vectors.shape[1]):
            basis_vectors = sorted_steerable_basis_vectors[:, :i+1]
            projection = np.dot(basis_vectors, np.dot(basis_vectors.T, scaled_img.flatten()))
            norm_err[n_img, i] = np.linalg.norm(scaled_img.flatten() - projection) / np.linalg.norm(scaled_img.flatten())

    max_norm_err = np.max(norm_err, axis=0)
    return max_norm_err
def test_basis_on_images(class_averages, steerable_basis_vectors, obj_sz):
    norm_err = np.zeros(class_averages.shape[0])
    norm_err_span = np.zeros(class_averages.shape[0])
    for n_img in range(class_averages.shape[0]):
        img = Image(class_averages[n_img])
        scaled_img = img.downsample(obj_sz)
        scaled_img = scaled_img.asnumpy()[0]
        norm_err[n_img] = np.linalg.norm(np.dot(steerable_basis_vectors.T, scaled_img.flatten()))/np.linalg.norm(scaled_img.flatten())
        norm_err_span[n_img] = np.linalg.norm(scaled_img.flatten()-np.dot(steerable_basis_vectors,np.dot(steerable_basis_vectors.T, scaled_img.flatten()))) / np.linalg.norm(
            scaled_img.flatten())
    return norm_err,norm_err_span

def test_basis_on_images_keren(class_averages, steerable_basis_vectors, obj_sz):
    norm_err = np.zeros(class_averages.shape[0])
    norm_err_span = np.zeros(class_averages.shape[0])
    for n_img in range(class_averages.shape[0]):
        norm_err[n_img] = np.linalg.norm(np.dot(steerable_basis_vectors.T, class_averages[n_img].flatten()))/np.linalg.norm(class_averages[n_img].flatten())
        norm_err_span[n_img] = np.linalg.norm(class_averages[n_img].flatten()-np.dot(steerable_basis_vectors,np.dot(steerable_basis_vectors.T, class_averages[n_img].flatten()))) / np.linalg.norm(
            class_averages[n_img].flatten())
    return norm_err,norm_err_span


def plot_class_averages(class_averages):
    # Determine the number of images
    num_images = class_averages.shape[0]

    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    # Loop through each image and display it
    for i in range(num_images):
        ax = axes[i]
        ax.imshow(class_averages[i], cmap='gray')
        ax.axis('off')
        ax.set_title(f'Image {i + 1}')

    # Show the plot
    plt.show()


# def compute_the_steerable_images(class_averages,obj_sz,fb_basis_objects,norm_err):
#     # compute the radial Fourier component to each class average
#     fb_basis = fb_basis_objects
#     # get the coefficients of the class averages in the FB basis
#     steerable_euclidian_l = np.zeros((obj_sz, obj_sz, 1+2*(fb_basis.ell_max), class_averages.shape[0]))
#     for n_img in range(class_averages.shape[0]):
#         img = Image(class_averages[n_img].astype(np.float32))
#         img_fb_coeff = fb_basis.evaluate_t(img)
#         v_0 = img_fb_coeff.copy()
#         v_0._data[0,fb_basis.k_max[0]:] = 0
#         v_0_img = fb_basis.evaluate(v_0).asnumpy()[0]
#         steerable_euclidian_l[:,:,0,n_img] = fb_basis.evaluate(v_0)
#         coeff_k_index_start = fb_basis.k_max[0]
#         for m in range(1, fb_basis.ell_max+1):
#             l_idx = 2*m-1
#             k_idx = fb_basis.k_max[m]
#             coeff_k_index_end_cos = coeff_k_index_start + k_idx
#             coeff_k_index_end_sin = coeff_k_index_end_cos + k_idx
#             vcos = img_fb_coeff.copy()
#             vcos._data[0,:coeff_k_index_start]=0
#             vcos._data[0,coeff_k_index_end_cos:] = 0
#             vsin = img_fb_coeff.copy()
#             vsin._data[0,:coeff_k_index_end_cos]=0
#             vsin._data[0,coeff_k_index_end_sin:] = 0
#             vcos_img = fb_basis.evaluate(vcos).asnumpy()[0]
#             vsin_img = fb_basis.evaluate(vsin).asnumpy()[0]
#             steerable_euclidian_l[:,:,l_idx,n_img] = vcos_img
#             steerable_euclidian_l[:, :, l_idx+1, n_img] = vsin_img
#             #steerable_euclidian_l[:,:,l_idx,n_img] = vcos_img + 1j*vsin_img
#             #steerable_euclidian_l[:, :, l_idx+1, n_img] = vcos_img - 1j * vsin_img
#             coeff_k_index_start = coeff_k_index_end_sin
#     return steerable_euclidian_l



def compute_the_steerable_images_normalization(class_averages, obj_sz, l_max, norm_err):
    # Initialize the FB basis
    fb_basis = FBBasis2D(size=(obj_sz, obj_sz), ell_max=l_max)

    # Initialize the steerable output array
    steerable_euclidian_l = np.zeros((obj_sz, obj_sz, 1 + 2 * (fb_basis.ell_max), class_averages.shape[0]))

    for n_img in range(class_averages.shape[0]):
        # Preprocess the image: downsample and convert to the required format
        img = Image(class_averages[n_img])
        scaled_img = np.asarray(img.downsample(obj_sz)).astype(np.float32)

        # Compute FB coefficients for the image
        img_fb_coeff = fb_basis.evaluate_t(scaled_img)

        # Normalize the FB coefficients to preserve energy
        coeff_energy = np.sum(np.abs(img_fb_coeff._data)**2)
        img_energy = np.linalg.norm(scaled_img)**2
        if coeff_energy > 0:  # Avoid division by zero
            scaling_factor = np.sqrt(img_energy / coeff_energy)
            img_fb_coeff._data *= scaling_factor

        # Extract the zeroth angular component
        v_0 = img_fb_coeff.copy()
        v_0._data[0, fb_basis.k_max[0]:] = 0
        v_0_img = fb_basis.evaluate(v_0).asnumpy()[0]
        steerable_euclidian_l[:, :, 0, n_img] = v_0_img

        coeff_k_index_start = fb_basis.k_max[0]
        for m in range(1, fb_basis.ell_max + 1):
            l_idx = 2 * m - 1
            k_idx = fb_basis.k_max[m]
            coeff_k_index_end_cos = coeff_k_index_start + k_idx
            coeff_k_index_end_sin = coeff_k_index_end_cos + k_idx

            # Cosine component
            vcos = img_fb_coeff.copy()
            vcos._data[0, :coeff_k_index_start] = 0
            vcos._data[0, coeff_k_index_end_cos:] = 0
            vcos_img = fb_basis.evaluate(vcos).asnumpy()[0]
            steerable_euclidian_l[:, :, l_idx, n_img] = vcos_img

            # Sine component
            vsin = img_fb_coeff.copy()
            vsin._data[0, :coeff_k_index_end_cos] = 0
            vsin._data[0, coeff_k_index_end_sin:] = 0
            vsin_img = fb_basis.evaluate(vsin).asnumpy()[0]
            steerable_euclidian_l[:, :, l_idx + 1, n_img] = vsin_img

            coeff_k_index_start = coeff_k_index_end_sin

    return steerable_euclidian_l

def compute_num_components(class_averages, steerable_euclidian_l, norm_err, obj_sz):
    number_of_components = np.zeros(steerable_euclidian_l.shape[3])
    for n_img in range(steerable_euclidian_l.shape[3]):
        img = Image(class_averages[n_img])
        scaled_img = img.downsample(obj_sz)
        scaled_img = scaled_img.asnumpy()
        scaled_img = scaled_img[0]
        m = 0
        norm_est = 1
        max_m = (steerable_euclidian_l.shape[2] - 1) // 2
        while norm_est > norm_err and m <= max_m:
            if m == 0:
                euc_est = steerable_euclidian_l[:,:,0,n_img]
            else:
                euc_est = euc_est + steerable_euclidian_l[:,:,2 * m - 1,n_img]+steerable_euclidian_l[:,:,2 * m ,n_img]
            norm_est = np.linalg.norm(zero_out_far_values(scaled_img) - zero_out_far_values(euc_est), ord='fro') / np.linalg.norm(zero_out_far_values(scaled_img), ord='fro')
            m = m + 1
        number_of_components[n_img] = m-1
    return number_of_components


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

def compute_the_steerable_basis_with_noise(steerable_euclidian_l,steerable_euclidian_l_noise):
    # Initialize an empty list to accumulate Q matrices
    steerable_basis_vectors_list = []

    for l in range(steerable_euclidian_l.shape[2]):
        l_images_objects = steerable_euclidian_l[:, :, l, :]
        l_images_objects_denoise = np.zeros(l_images_objects.shape)
        l_images_noise = steerable_euclidian_l_noise[:, :, l, :]
        mean_noise_l = np.mean(l_images_noise,axis=2)
        normelized_mean_noise_l = mean_noise_l/np.linalg.norm(mean_noise_l)
        for n in range(l_images_objects.shape[2]):
            l_images_objects_denoise[:,:,n] = l_images_objects[:,:,n] - np.sum(l_images_objects[:,:,n]*normelized_mean_noise_l)*normelized_mean_noise_l
        Q = qr_factorization_matrices(l_images_objects_denoise)
        steerable_basis_vectors_list.append(Q)  # Add each Q matrix to the list

    # Concatenate all Q matrices horizontally at the end
    steerable_basis_vectors = np.hstack(steerable_basis_vectors_list)

    return steerable_basis_vectors

def compute_the_steerable_basis_1(steerable_euclidian_l):
    # Initialize an empty list to accumulate Q matrices
    steerable_basis_vectors_list = []

    for l in range(steerable_euclidian_l.shape[2]):
        l_images = steerable_euclidian_l[:, :, l, :]
        Q = qr_factorization_matrices(l_images)
        steerable_basis_vectors_list.append(Q)  # Add each Q matrix to the list

    # Concatenate all Q matrices horizontally at the end
    concatenated_basis = np.hstack(steerable_basis_vectors_list)
    steerable_basis_vectors, _ = np.linalg.qr(concatenated_basis)

    return steerable_basis_vectors


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

def is_patch_contaminated(contamination_mask, x, y, patch_size):
    """
    Check if the patch centered at (x, y) intersects with contamination.

    Parameters:
    contamination_mask : Downsampled contamination mask.
    x, y               : Center coordinates of the patch.
    patch_size         : Size of the patch.

    Returns:
    contaminated : Boolean, True if patch intersects contamination.
    """
    half_patch = patch_size // 2
    row_start = max(0, y - half_patch)
    row_end = min(contamination_mask.shape[0], y + half_patch)
    col_start = max(0, x - half_patch)
    col_end = min(contamination_mask.shape[1], x + half_patch)

    # Check if any pixel in this region is contaminated
    return np.any(contamination_mask[row_start:row_end, col_start:col_end] == 1)

    ### Image and Patch Processing Functions ###

    def adjust_contrast(image, contrast_factor=1.0, brightness_factor=0.0):
        # Adjust contrast and brightness
        adjusted_image = contrast_factor * image + brightness_factor
        adjusted_image = np.clip(adjusted_image, 0, 1)  # Clip values to [0, 1]
        return adjusted_image




def extract_objects_from_coordinates(Y, coordinates, patch_size):
    """
    Extract object patches from the image based on coordinates.

    Parameters:
    Y            : The input image (2D array).
    coordinates  : List of (x, y) coordinates.
    patch_size   : The size of the square patch to extract.

    Returns:
    patches : Array of extracted patches.
    """
    half_patch = patch_size // 2
    patches = []

    for (x, y) in coordinates:
        # Define patch boundaries
        row_start = max(0, y - half_patch)
        row_end = min(Y.shape[0], y + half_patch)
        col_start = max(0, x - half_patch)
        col_end = min(Y.shape[1], x + half_patch)

        # Extract the patch
        patch = Y[row_start:row_end, col_start:col_end]
        if patch.shape == (patch_size, patch_size):
            patches.append(patch)

    return np.array(patches)




def noise_reduction_wiener(image, mysize=5, noise=None):
    """
    Apply Wiener filtering to reduce noise in a grayscale cryo-EM micrograph.

    Parameters:
    - image: 2D numpy array representing the grayscale cryo-EM micrograph.
    - mysize: Size of the filter window (larger values result in more smoothing).
    - noise: Estimate of the noise power; if None, the noise is estimated from the image.

    Returns:
    - denoised_image: 2D numpy array of the denoised image.
    """
    # Apply the Wiener filter
    denoised_image = wiener(image, mysize=mysize, noise=noise)

    return denoised_image


def noise_reduction(image, sigma=1.0):
    """
    Apply Gaussian filtering to reduce noise in a cryo-EM image.

    Parameters:
    - image: 2D numpy array representing the cryo-EM micrograph.
    - sigma: Standard deviation for Gaussian kernel, controlling the amount of smoothing.
             Higher sigma = more smoothing.

    Returns:
    - denoised_image: 2D numpy array of the denoised image.
    """
    # Apply Gaussian filter for noise reduction
    denoised_image = gaussian_filter(image, sigma=sigma)

    return denoised_image


def contrast_stretch(image):
    """
    Apply contrast stretching to the input micrograph image.

    Parameters:
    - image: 2D numpy array representing the micrograph.

    Returns:
    - stretched_image: 2D numpy array of the contrast-stretched image.
    """
    # Get the minimum and maximum pixel values in the image
    min_val = np.min(image)
    max_val = np.max(image)

    # Stretch the image to the full range [0, 1]
    stretched_image = (image - min_val) / (max_val - min_val)

    return stretched_image


def extract_object_patches_and_return(object_img_scaled, box_sz, num_of_patches=10):
    """
    Extract object patches of size box_sz x box_sz from a micrograph, stopping after num_of_patches (default 10).
    """
    rDelAlgorithm = (int(box_sz) + 1)
    box_sz_var = int(box_sz / 3)

    Y_no_mean = object_img_scaled - np.mean(object_img_scaled)
    var_box = np.ones((box_sz_var, box_sz_var))
    Y_var = convolve2d(Y_no_mean ** 2, var_box, mode='same')

    # Exclude boundaries
    Y_var[:int(box_sz), :] = -np.inf
    Y_var[:, :int(box_sz)] = -np.inf
    Y_var[-int(box_sz):, :] = -np.inf
    Y_var[:, -int(box_sz):] = -np.inf

    object_patches = []
    cnt_p = 0
    p_max = 1

    while p_max > 0 and cnt_p < num_of_patches:
        p_max = np.max(Y_var)
        if p_max < 0:
            break
        I = np.argmax(Y_var)
        i_row, i_col = np.unravel_index(I, Y_var.shape)
        idxRowCandidate = np.zeros(Y_var.shape[0], dtype=bool)
        idxColCandidate = np.zeros(Y_var.shape[1], dtype=bool)
        idxRowCandidate[max(i_row - rDelAlgorithm, 0):min(i_row + rDelAlgorithm, Y_var.shape[0])] = True
        idxColCandidate[max(i_col - rDelAlgorithm, 0):min(i_col + rDelAlgorithm, Y_var.shape[1])] = True
        # patch = object_img_scaled[i_row:i_row + box_sz, i_col:i_col + box_sz]
        # patch = object_img_scaled[i_row:i_row + box_sz, i_col:i_col + box_sz]
        patch = object_img_scaled[i_row - box_sz // 2:i_row + box_sz // 2, i_col - box_sz // 2:i_col + box_sz // 2]
        # Check if patch shape is valid and ensure it doesn't overlap with contamination
        if patch.shape == (box_sz, box_sz) and not is_patch_contaminated(contamination_mask, i_col, i_row, box_sz):
            noise_patches.append(patch)
            cnt_p += 1
        Y_var[np.ix_(idxRowCandidate, idxColCandidate)] = -np.inf

    return np.array(object_patches)


def extract_noise_patches_and_return_min_variance(noise_img_scaled,contamination_mask, box_sz, num_of_patches=30):
    """
    Extract noise patches of size box_sz x box_sz from a micrograph, starting from the patches with the lowest variance.
    """
    # rDelAlgorithm = (int(box_sz) + 1)
    rDelAlgorithm = int(box_sz / 2)
    box_sz_var = int(box_sz / 4)
    Y_no_mean = noise_img_scaled - np.mean(noise_img_scaled)
    var_box = np.ones((box_sz_var, box_sz_var))
    Y_var = convolve2d(Y_no_mean ** 2, var_box, mode='same')

    # Exclude boundaries
    Y_var[:int(box_sz_var), :] = np.inf
    Y_var[:, :int(box_sz_var)] = np.inf
    Y_var[-int(box_sz_var):, :] = np.inf
    Y_var[:, -int(box_sz_var):] = np.inf

    noise_patches = []
    cnt_p = 0
    p_min = 0

    while p_min < np.inf and cnt_p < num_of_patches:
        p_min = np.min(Y_var)  # Get the minimum variance now
        if p_min == np.inf:
            break
        I = np.argmin(Y_var)  # Index of the minimum variance value
        i_row, i_col = np.unravel_index(I, Y_var.shape)
        idxRowCandidate = np.zeros(Y_var.shape[0], dtype=bool)
        idxColCandidate = np.zeros(Y_var.shape[1], dtype=bool)
        idxRowCandidate[max(i_row - rDelAlgorithm, 0):min(i_row + rDelAlgorithm, Y_var.shape[0])] = True
        idxColCandidate[max(i_col - rDelAlgorithm, 0):min(i_col + rDelAlgorithm, Y_var.shape[1])] = True
        # patch = noise_img_scaled[i_row:i_row + box_sz, i_col:i_col + box_sz]
        if np.mod(box_sz, 2) == 1:
            patch = noise_img_scaled[i_row - 1 - box_sz // 2:i_row + box_sz // 2,
                    i_col - 1 - box_sz // 2:i_col + box_sz // 2]
        else:
            patch = noise_img_scaled[i_row - box_sz // 2:i_row + box_sz // 2, i_col - box_sz // 2:i_col + box_sz // 2]
        if patch.shape == (box_sz, box_sz):
            noise_patches.append(patch)
            cnt_p += 1
        # Set the selected area to np.inf so it won't be selected again
        Y_var[np.ix_(idxRowCandidate, idxColCandidate)] = np.inf

    return np.array(noise_patches)


# from scipy.signal import convolve2d
# import numpy as np

# import numpy as np
# from scipy.signal import convolve2d
# from joblib import Parallel, delayed





def compute_patch_variance(padded_img, i, j, box_sz):
    """
    Compute variance of a patch centered at (i, j).

    Parameters:
    padded_img : np.ndarray : Padded micrograph image.
    i, j       : int        : Center coordinates of the patch in the padded image.
    box_sz     : int        : Size of the patch.

    Returns:
    patch_variance : float : Variance of the patch.
    """
    half_box_sz = box_sz // 2
    patch = padded_img[i - half_box_sz:i + half_box_sz, j - half_box_sz:j + half_box_sz]
    patch_mean = np.mean(patch)
    patch_variance = np.mean((patch - patch_mean) ** 2)
    return patch_variance



def extract_noise_patches_and_coor_return_min_variance(noise_img_scaled, contamination_mask, box_sz, num_of_patches=30):
    """
    Extract noise patches of size box_sz x box_sz from a micrograph, starting from the patches with the lowest variance.

    Returns:
    noise_patches    : ndarray : Array of extracted noise patches.
    patch_coordinates : list   : List of (row, col) center coordinates for the patches.
    """

    # Define half_box_sz depending on whether box_sz is even or odd
    half_box_sz = box_sz // 2 if box_sz % 2 == 0 else (box_sz // 2) + 1

    # Create an averaging kernel (box filter)
    kernel = np.ones((box_sz, box_sz), dtype=np.float32) / (box_sz * box_sz)
    kernel_torch = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, box_sz, box_sz)

    # Convert image to PyTorch tensor
    noise_img_scaled_torch = torch.tensor(noise_img_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Perform convolution for mean
    mean_img_torch = F.conv2d(noise_img_scaled_torch, kernel_torch, stride=1, padding='same')

    # Perform convolution for mean of squared image (for variance calculation)
    mean_squared_img_torch = F.conv2d(noise_img_scaled_torch ** 2, kernel_torch, stride=1, padding='same')

    # Convert results back to NumPy
    mean_img_np = mean_img_torch.squeeze().numpy()  # Remove extra dimensions
    mean_squared_img_np = mean_squared_img_torch.squeeze().numpy()

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
            patch = noise_img_scaled[
                max(0, i_row - half_box_sz): min(i_row + half_box_sz, noise_img_scaled.shape[0]),
                max(0, i_col - half_box_sz): min(i_col + half_box_sz, noise_img_scaled.shape[1])
            ]
        else:
            patch = noise_img_scaled[
                max(0, i_row - half_box_sz): min(i_row + half_box_sz - 1, noise_img_scaled.shape[0]),
                max(0, i_col - half_box_sz): min(i_col + half_box_sz - 1, noise_img_scaled.shape[1])
            ]

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


import numpy as np
from scipy.ndimage import convolve


def extract_noise_patches_and_coor_return_min_variance_numpy(noise_img_scaled, contamination_mask, box_sz,
                                                             num_of_patches=30):
    """
    Extract noise patches of size box_sz x box_sz from a micrograph, starting from the patches with the lowest variance.

    Returns:
    noise_patches    : ndarray : Array of extracted noise patches.
    patch_coordinates : list   : List of (row, col) center coordinates for the patches.
    """

    # Define half_box_sz depending on whether box_sz is even or odd
    half_box_sz = box_sz // 2 if box_sz % 2 == 0 else (box_sz // 2) + 1

    # Create an averaging kernel (box filter)
    kernel = np.ones((box_sz, box_sz), dtype=np.float32) / (box_sz * box_sz)

    # Perform convolution for mean using SciPy's convolve (memory efficient)
    mean_img = convolve(noise_img_scaled, kernel, mode='constant', cval=0.0)

    # Perform convolution for mean of squared image (for variance calculation)
    mean_squared_img = convolve(noise_img_scaled ** 2, kernel, mode='constant', cval=0.0)

    # Variance formula: E[x^2] - (E[x])^2
    Y_var = mean_squared_img - mean_img ** 2

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
            patch = noise_img_scaled[
                    max(0, i_row - half_box_sz): min(i_row + half_box_sz, noise_img_scaled.shape[0]),
                    max(0, i_col - half_box_sz): min(i_col + half_box_sz, noise_img_scaled.shape[1])
                    ]
        else:
            patch = noise_img_scaled[
                    max(0, i_row - half_box_sz): min(i_row + half_box_sz - 1, noise_img_scaled.shape[0]),
                    max(0, i_col - half_box_sz): min(i_col + half_box_sz - 1, noise_img_scaled.shape[1])
                    ]

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
def extract_noise_patches_and_coor_return_min_variance_torch(noise_img_scaled, contamination_mask, box_sz,
                                                             num_of_patches=30):
    """
    Extract noise patches of size box_sz x box_sz from a micrograph, starting from the patches with the lowest variance.

    Returns:
    noise_patches    : ndarray : Array of extracted noise patches.
    patch_coordinates : list   : List of (row, col) center coordinates for the patches.
    """

    # Use GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define half_box_sz depending on whether box_sz is even or odd
    half_box_sz = box_sz // 2 if box_sz % 2 == 0 else (box_sz // 2) + 1

    # Create an averaging kernel (box filter) using float32 to reduce memory usage
    kernel = np.ones((box_sz, box_sz), dtype=np.float32) / (box_sz * box_sz)
    kernel_torch = torch.tensor(kernel, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    # Convert image to PyTorch tensor and move to device
    noise_img_scaled_torch = torch.tensor(noise_img_scaled, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(
        0)

    # Perform convolution for mean and mean of squared image
    mean_img_torch = F.conv2d(noise_img_scaled_torch, kernel_torch, stride=1, padding='same')
    mean_squared_img_torch = F.conv2d(noise_img_scaled_torch ** 2, kernel_torch, stride=1, padding='same')

    # Clear memory for intermediate tensors after use
    del kernel_torch
    del noise_img_scaled_torch
    torch.cuda.empty_cache()
    gc.collect()

    # Calculate variance using in-place operations
    Y_var_torch = mean_squared_img_torch - mean_img_torch.pow(2)

    # Clear intermediate tensors to free memory
    del mean_img_torch
    del mean_squared_img_torch
    torch.cuda.empty_cache()
    gc.collect()

    # Move to CPU and convert to NumPy
    Y_var = Y_var_torch.squeeze().cpu().numpy()

    # Clear the remaining GPU tensors
    del Y_var_torch
    torch.cuda.empty_cache()
    gc.collect()

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

        # Index of the minimum variance value
        i_row, i_col = np.unravel_index(np.argmin(Y_var), Y_var.shape)

        # Extract patch, ensuring it remains box_sz x box_sz
        row_start = max(0, i_row - half_box_sz)
        row_end = min(i_row + half_box_sz, noise_img_scaled.shape[0])
        col_start = max(0, i_col - half_box_sz)
        col_end = min(i_col + half_box_sz, noise_img_scaled.shape[1])

        patch = noise_img_scaled[row_start:row_end, col_start:col_end]

        # Skip invalid or out-of-bound patches
        if patch.shape != (box_sz, box_sz):
            Y_var[i_row, i_col] = np.inf  # Mask the invalid location
            continue

        # Append patch and coordinates
        noise_patches.append(patch)
        patch_coordinates.append((i_row, i_col))
        cnt_p += 1

        # Mask the region to avoid overlap
        row_start = max(0, i_row - box_sz)
        row_end = min(Y_var.shape[0], i_row + box_sz + 1)
        col_start = max(0, i_col - box_sz)
        col_end = min(i_col + box_sz, Y_var.shape[1])
        Y_var[row_start:row_end, col_start:col_end] = np.inf  # Set the area to infinity

    # Convert noise patches to NumPy array
    noise_patches = np.array(noise_patches)

    # Clear cache one last time before returning results
    torch.cuda.empty_cache()
    gc.collect()

    return noise_patches, patch_coordinates

def extract_noise_patches_and_coor_return_min_variance_tf(noise_img_scaled, contamination_mask, box_sz ,num_of_patches=30):

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

    x =1
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
            patch = noise_img_scaled[
                max(0, i_row - half_box_sz): min(i_row + half_box_sz, noise_img_scaled.shape[0]),
                max(0, i_col - half_box_sz): min(i_col + half_box_sz, noise_img_scaled.shape[1])
            ]
        else:
            patch = noise_img_scaled[
                max(0, i_row - half_box_sz): min(i_row + half_box_sz - 1, noise_img_scaled.shape[0]),
                max(0, i_col - half_box_sz): min(i_col + half_box_sz - 1, noise_img_scaled.shape[1])
            ]

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

def extract_smaller_patches_with_coords(larger_patches, large_patch_coords, small_patch_size, num_smaller_patches):
    """
    Extract smaller patches with minimal variance and return their coordinates.

    Parameters:
    larger_patches      : ndarray : Array of larger noise patches [N, H, W].
    large_patch_coords  : list    : List of (row, col) coordinates of larger patch centers.
    small_patch_size    : int     : Desired size of smaller patches.
    num_smaller_patches : int     : Number of smaller patches to extract.

    Returns:
    smaller_patches : list : Extracted smaller patches.
    small_coords    : list : Coordinates of the smaller patches relative to the original micrograph.
    """
    half_small_sz = small_patch_size // 2
    smaller_patches = []
    small_coords = []

    # Compute variance for larger patches and sort
    larger_patch_variances = [np.var(patch) for patch in larger_patches]
    sorted_indices = np.argsort(larger_patch_variances)
    sorted_larger_patches = [larger_patches[i] for i in sorted_indices]
    sorted_large_coords = [large_patch_coords[i] for i in sorted_indices]

    # Debugging: Check the sorting of patches
    print(f"Number of larger patches: {len(larger_patches)}")
    print(f"Number of larger patch coordinates: {len(large_patch_coords)}")
    print(f"Computed variances: {larger_patch_variances}")
    print(f"Sorted variances: {[larger_patch_variances[i] for i in sorted_indices]}")

    # Extract smaller patches
    for patch_idx, (patch, (center_row, center_col)) in enumerate(zip(sorted_larger_patches, sorted_large_coords)):
        # Debugging: Check the current larger patch
        print(f"\nProcessing larger patch {patch_idx + 1}/{len(sorted_larger_patches)}")
        print(f"Center: ({center_row}, {center_col}), Patch shape: {patch.shape}")

        # Handle invalid patches
        if not np.isfinite(patch).all():
            print(f"Skipping larger patch {patch_idx + 1} due to invalid values.")
            continue

        # Create a variance map for the current larger patch
        var_map = np.zeros_like(patch)
        padded_patch = np.pad(patch, half_small_sz, mode='constant', constant_values=np.inf)

        for i in range(half_small_sz, patch.shape[0] + half_small_sz):
            for j in range(half_small_sz, patch.shape[1] + half_small_sz):
                small_patch = padded_patch[i - half_small_sz:i + half_small_sz,
                                           j - half_small_sz:j + half_small_sz]
                # Skip invalid patches
                if not np.isfinite(small_patch).all():
                    var_map[i - half_small_sz, j - half_small_sz] = np.inf
                    continue

                small_patch_mean = np.mean(small_patch)
                small_patch_variance = np.mean((small_patch - small_patch_mean) ** 2)
                var_map[i - half_small_sz, j - half_small_sz] = small_patch_variance

        # Debugging: Check variance map statistics
        print(f"Variance map min: {np.nanmin(var_map)}, max: {np.nanmax(var_map)}")

        # Extract the smallest variance smaller patches
        indices = np.argsort(var_map.flatten())[:num_smaller_patches]
        for idx in indices:
            i, j = np.unravel_index(idx, var_map.shape)
            small_patch = patch[i - half_small_sz:i + half_small_sz,
                                j - half_small_sz:j + half_small_sz]
            if small_patch.shape == (small_patch_size, small_patch_size):
                # Adjust coordinates to the original micrograph
                orig_row = center_row - patch.shape[0] // 2 + i
                orig_col = center_col - patch.shape[1] // 2 + j
                smaller_patches.append(small_patch)
                small_coords.append((orig_row, orig_col))

                # Debugging: Check extracted patch
                print(f"Extracted smaller patch at ({orig_row}, {orig_col}) with variance: {np.var(small_patch)}")

                if len(smaller_patches) >= num_smaller_patches:
                    print("Reached desired number of smaller patches.")
                    break

        # Stop once the desired number of smaller patches is reached
        if len(smaller_patches) >= num_smaller_patches:
            break

    # Debugging: Final results
    print(f"\nTotal smaller patches extracted: {len(smaller_patches)}")
    return np.array(smaller_patches), small_coords



def extract_high_variance_patches_with_skip(noise_img_scaled, box_sz, obj_sz_down_scaled, delta, num_of_patches=30):
    """
    Extract noise patches of size box_sz x box_sz from an image by skipping the top N patches with the highest variance,
    and then selecting the next num_of_patches patches with the highest variance.

    Parameters:
    - noise_img_scaled (2D NumPy array): Input noise image.
    - box_sz (int): Size of each square patch (box_sz x box_sz).
    - obj_sz_down_scaled (int): Downscaled object size, used to calculate deletion region.
    - delta (int): Additional parameter to adjust the size of the deletion region.
    - num_of_patches (int): Number of patches to extract after skipping (default: 30).

    Returns:
    - noise_patches (3D NumPy array): Extracted patches with shape (num_of_patches, box_sz, box_sz).
    """

    # Step 1: Calculate the side length of the deletion region
    sideLengthAlgorithmL = 2 * obj_sz_down_scaled + delta
    rDelAlgorithmL = sideLengthAlgorithmL // 2  # Radius for deletion

    print(f"Side Length Algorithm L: {sideLengthAlgorithmL}")
    print(f"Deletion Radius (rDelAlgorithmL): {rDelAlgorithmL}")

    # Step 2: Calculate the number of patches to skip
    N_skip = (noise_img_scaled.shape[0] // (sideLengthAlgorithmL)) * (
                noise_img_scaled.shape[1] // (sideLengthAlgorithmL))
    print(f"Number of patches to skip (N_skip): {N_skip}")

    # Step 3: Center the image by removing the mean
    Y_no_mean = noise_img_scaled - np.mean(noise_img_scaled)

    # Step 4: Compute the local variance map using convolution
    var_box = np.ones((box_sz, box_sz))
    Y_var = convolve2d(Y_no_mean ** 2, var_box, mode='same')

    # Step 5: Exclude boundary regions by setting their variance to -inf to prevent selection
    Y_var[:rDelAlgorithmL, :] = -np.inf
    Y_var[:, :rDelAlgorithmL] = -np.inf
    Y_var[-rDelAlgorithmL:, :] = -np.inf
    Y_var[:, -rDelAlgorithmL:] = -np.inf

    # Step 6: Initialize lists to store patches and counters
    noise_patches = []
    cnt_p = 0  # Counter for extracted patches
    cnt_skip = 0  # Counter for skipped patches

    # Step 7: Start the first phase - Skipping the top N_skip patches with highest variance
    print("Starting the skipping phase...")
    while cnt_skip < N_skip:
        p_max = np.max(Y_var)
        if p_max == -np.inf:
            print("No more valid patches available during skipping phase.")
            break

        # Find the index of the patch with the highest variance
        I = np.argmax(Y_var)
        i_row, i_col = np.unravel_index(I, Y_var.shape)

        # Define the boundaries of the patch
        half_box = box_sz // 2
        if box_sz % 2 == 1:
            # For odd box sizes, ensure the patch is centered correctly
            row_start = i_row - half_box
            row_end = i_row + half_box + 1
            col_start = i_col - half_box
            col_end = i_col + half_box + 1
        else:
            # For even box sizes, adjust indices accordingly
            row_start = i_row - half_box
            row_end = i_row + half_box
            col_start = i_col - half_box
            col_end = i_col + half_box

        # Extract the patch
        patch = noise_img_scaled[row_start:row_end, col_start:col_end]

        # Check if the patch has the correct size
        if patch.shape == (box_sz, box_sz):
            # Append the patch to the noise_patches list (though we're skipping them)
            # If you don't want to store skipped patches, you can omit this step
            # noise_patches.append(patch)  # Uncomment if you want to keep skipped patches
            cnt_skip += 1
        else:
            print(f"Skipped invalid patch at ({i_row}, {i_col}) with shape {patch.shape}")

        # Define the region to zero out around the selected patch
        del_row_start = max(i_row - rDelAlgorithmL, 0)
        del_row_end = min(i_row + rDelAlgorithmL, Y_var.shape[0])
        del_col_start = max(i_col - rDelAlgorithmL, 0)
        del_col_end = min(i_col + rDelAlgorithmL, Y_var.shape[1])

        # Zero out the defined region to skip this patch
        Y_var[del_row_start:del_row_end, del_col_start:del_col_end] = -np.inf

    # Step 8: Start the second phase - Extracting the next num_of_patches patches with highest variance
    print("Starting the extraction phase...")
    while cnt_p < num_of_patches:
        p_max = np.max(Y_var)
        if p_max == -np.inf:
            print("No more valid patches available during extraction phase.")
            break

        # Find the index of the patch with the highest variance
        I = np.argmax(Y_var)
        i_row, i_col = np.unravel_index(I, Y_var.shape)

        # Define the boundaries of the patch
        half_box = box_sz // 2
        if box_sz % 2 == 1:
            # For odd box sizes, ensure the patch is centered correctly
            row_start = i_row - half_box
            row_end = i_row + half_box + 1
            col_start = i_col - half_box
            col_end = i_col + half_box + 1
        else:
            # For even box sizes, adjust indices accordingly
            row_start = i_row - half_box
            row_end = i_row + half_box
            col_start = i_col - half_box
            col_end = i_col + half_box

        # Extract the patch
        patch = noise_img_scaled[row_start:row_end, col_start:col_end]

        # Check if the patch has the correct size
        if patch.shape == (box_sz, box_sz):
            # Append the patch to the noise_patches list
            noise_patches.append(patch)
            cnt_p += 1
            print(f"Extracted patch {cnt_p} at ({i_row}, {i_col}) with variance {p_max:.2f}")
        else:
            print(f"Skipped invalid patch at ({i_row}, {i_col}) with shape {patch.shape}")

        # Define the region to zero out around the selected patch
        del_row_start = max(i_row - rDelAlgorithmL, 0)
        del_row_end = min(i_row + rDelAlgorithmL, Y_var.shape[0])
        del_col_start = max(i_col - rDelAlgorithmL, 0)
        del_col_end = min(i_col + rDelAlgorithmL, Y_var.shape[1])

        # Zero out the defined region to prevent overlapping
        Y_var[del_row_start:del_row_end, del_col_start:del_col_end] = 0

    return np.array(noise_patches)


### Steerable Basis and Fourier-Bessel Processing ###

def compute_steerable_coefficients_simplified(patches, fb_basis):
    """
    Compute the Fourier-Bessel steerable coefficients for a set of patches.
    """
    num_patches = patches.shape[0]
    num_basis_functions = fb_basis.k_max[0] + 2 * np.sum(fb_basis.k_max[1:])
    all_patch_coefficients = np.zeros((num_patches, num_basis_functions), dtype=np.float32)
    x = 1
    for i, patch in enumerate(patches):
        patch_aspire = AspireImage(patch)
        fb_coefficients = fb_basis.evaluate_t(AspireImage(patch.astype(np.float32)))

        all_patch_coefficients[i, :] = fb_coefficients._data.flatten()

    return all_patch_coefficients


def compute_covariance_matrix(v_data_list):
    covariance_matrix = np.cov(v_data_list, rowvar=False)


### Steerable Basis and Fourier-Bessel Processing ###

def compute_steerable_coefficients_simplified(patches, fb_basis):
    """
    Compute the Fourier-Bessel steerable coefficients for a set of patches.
    """
    num_patches = patches.shape[0]
    num_basis_functions = fb_basis.k_max[0] + 2 * np.sum(fb_basis.k_max[1:])
    all_patch_coefficients = np.zeros((num_patches, num_basis_functions), dtype=np.float32)
    x = 1
    for i, patch in enumerate(patches):
        patch_aspire = AspireImage(patch)
        fb_coefficients = fb_basis.evaluate_t(AspireImage(patch.astype(np.float32)))

        all_patch_coefficients[i, :] = fb_coefficients._data.flatten()

    return all_patch_coefficients


def compute_covariance_matrix(v_data_list):
    covariance_matrix = np.cov(v_data_list, rowvar=False)
    return covariance_matrix


def extract_largest_eigenvectors(covariance_matrix, k):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    largest_eigenvectors = eigenvectors[:, sorted_indices[:k]]
    return largest_eigenvectors, eigenvalues[sorted_indices[:k]]


def check_orthonormality(eigenvectors):
    identity = np.eye(eigenvectors.shape[1])
    return np.allclose(np.dot(eigenvectors.T, eigenvectors), identity)


def reconstruct_image_from_eigenvector(eigenvector, fb_basis, obj_sz):
    temp = fb_basis.evaluate_t(AspireImage(np.zeros((obj_sz, obj_sz), dtype=np.float32)))
    temp._data[0] = eigenvector
    # coef_data = eigenvector.astype(np.float32)
    # coef_obj = Coef(data=coef_data, basis=fb_basis)
    reconstructed_img = fb_basis.evaluate(temp).asnumpy()[0]
    return reconstructed_img


def reconstruct_images_from_vectors(largest_eigenvectors, fb_basis, obj_sz):
    n_eigenvectors = largest_eigenvectors.shape[1]
    images = np.zeros((n_eigenvectors, obj_sz, obj_sz), dtype=np.float32)
    for i in range(n_eigenvectors):
        eigenvector = largest_eigenvectors[:, i]
        images[i] = reconstruct_image_from_eigenvector(eigenvector, fb_basis, obj_sz)
    return images


def qr_factorization_and_reconstruct_images(images, mean_noise_image):
    n_images = images.shape[0]
    img_height, img_width = images.shape[1], images.shape[2]
    flattened_mean_noise_image = mean_noise_image.flatten()
    # Step 3: Reshape images to (n_images, img_height * img_width)
    flattened_images = images.reshape(n_images, img_height * img_width)
    # Step 4: Add mean_noise_image as the first image to the set of flattened images
    # all_images = np.vstack([flattened_mean_noise_image, flattened_images])
    all_images = flattened_images
    # Step 5: Perform QR decomposition on the transposed matrix
    Q, R = np.linalg.qr(all_images.T)  # QR decomposition on the transposed matrix
    # Step 6: Reshape Q columns back into image shapes
    # We keep all n_images + 1 orthonormal vectors (which span the image space)
    #new_images = Q[:, :n_images + 1].T  # Transpose to get (n_images+1, img_height * img_width)
    new_images = Q[:, :n_images ].T  # Transpose to get (n_images+1, img_height * img_width)
   # new_images = new_images.reshape(n_images + 1, img_height, img_width)  # Reshape to original image dimensions
    new_images = new_images.reshape(n_images , img_height, img_width)
    # Step 7: Return all the new orthonormal images, including the first image (mean_noise_image)
    return new_images


def combine_templates_and_noise_basis(new_images_objects, orthogonal_images_eigen_vectors, template_weight=0.9):
    """
    Combine templates and noise eigenvectors with weights for steerable basis construction.
    
    Parameters:
    - new_images_objects: ndarray of shape (n_templates, image_height, image_width), noise-reduced templates.
    - orthogonal_images_eigen_vectors: ndarray of shape (n_noise_basis, image_height, image_width), noise basis.
    - template_weight: float, relative weight of templates (0 < template_weight < 1).
    
    Returns:
    - combined_data: ndarray of shape (n_combined, image_height, image_width).
    """
    w_templates = np.sqrt(template_weight)
    w_noise = np.sqrt(1 - template_weight)
    
    # Flatten and weight templates and noise eigenvectors
    n_templates, img_h, img_w = new_images_objects.shape
    n_noise_basis, _, _ = orthogonal_images_eigen_vectors.shape
    
    templates_flattened = new_images_objects.reshape(n_templates, -1) * w_templates
    noise_flattened = orthogonal_images_eigen_vectors.reshape(n_noise_basis, -1) * w_noise
    
    # Combine and reshape
    combined_data = np.vstack([templates_flattened, noise_flattened]).reshape(-1, img_h, img_w)
    return combined_data

def compute_combined_steerable_basis(combined_data, obj_sz, l_max):
    """
    Compute the steerable basis from combined templates and noise basis.
    
    Parameters:
    - combined_data: ndarray of shape (n_combined, image_height, image_width), combined dataset.
    - obj_sz: int, size of the downsampled object.
    - l_max: int, maximum angular frequency.
    
    Returns:
    - steerable_basis_vectors: ndarray of steerable basis vectors.
    """
    steerable_euclidian_l = compute_the_steerable_images(combined_data, obj_sz, l_max, norm_err=None)
    steerable_basis_vectors = compute_the_steerable_basis(steerable_euclidian_l)
    return steerable_basis_vectors


def compute_covariance_matrix_gpu(data):
    """
    Compute the covariance matrix using TensorFlow on the GPU.

    Parameters:
    data : np.ndarray
        2D array where rows represent samples and columns represent features.

    Returns:
    covariance_matrix : tf.Tensor
        The computed covariance matrix.
    """
    # Convert data to TensorFlow tensor
    data_tf = tf.convert_to_tensor(data, dtype=tf.float32)

    # Compute mean along the rows (axis=0)
    mean_tf = tf.reduce_mean(data_tf, axis=1, keepdims=True)

    # Center the data by subtracting the mean
    centered_data = data_tf - mean_tf

    # Compute the covariance matrix
    covariance_matrix = tf.matmul(centered_data, centered_data, transpose_a=True) / tf.cast(tf.shape(data_tf)[0] - 1, tf.float32)

    return covariance_matrix

def generate_new_images_objects(object_images, mean_noise_image, orthogonal_images_eigen_vectors):
    """
    Generate new images by subtracting the mean noise image and projecting the images onto the
    orthogonal eigenvectors and then subtracting the projection.

    Parameters:
    object_images : ndarray
        A 3D NumPy array of shape (n_images, image_height, image_width) representing the object images.
    mean_noise_image : ndarray
        A 2D NumPy array of shape (image_height, image_width) representing the mean noise image.
    orthogonal_images_eigen_vectors : ndarray
        A 3D NumPy array of shape (n_basis, image_height, image_width) representing the orthogonal eigenvectors.

    Returns:
    new_images_objects : ndarray
        A 3D NumPy array of the new images after projection and subtraction.
    """
    # Flatten the orthogonal eigenvector images for projection purposes
    n_basis, image_height, image_width = orthogonal_images_eigen_vectors.shape
    flattened_eigen_vectors = orthogonal_images_eigen_vectors.reshape(n_basis, -1)

    # Initialize an array to store the new images
    n_images = object_images.shape[0]
    new_images_objects = np.zeros_like(object_images)

    for i in range(n_images):
        # # Step 1: Subtract the mean noise image from the object image
        obj_img = object_images[i] - mean_noise_image
        #obj_img = object_images[i]
        # Step 2: Flatten the object image for projection
        obj_img_flattened = obj_img.flatten()

        # Step 3: Compute the projection of the object image onto the orthogonal eigenvectors
        projection_coefficients = np.dot(obj_img_flattened, flattened_eigen_vectors.T)

        # Step 4: Reconstruct the projection (sum of the projections of the image onto the eigenvectors)
        projection_sum = np.dot(projection_coefficients, flattened_eigen_vectors)
        projection_sum_2 = (flattened_eigen_vectors.T)@(flattened_eigen_vectors@obj_img_flattened)
        # Step 5: Subtract the projection from the object image
        new_img_flattened = obj_img_flattened - projection_sum

        # Step 6: Reshape the flattened new image back to the original image shape
        new_images_objects[i] = new_img_flattened.reshape(image_height, image_width)

    return new_images_objects


def compute_projection_norms(patches, basis_vectors):
    n_patches = patches.shape[0]
    projection_norms = np.zeros(n_patches)
    for i in range(n_patches):
        patch_flattened = patches[i].flatten()
        inner_products = np.dot(patch_flattened, basis_vectors)
        sum_of_squares = np.sum(inner_products ** 2)
        projection_norms[i] = np.sqrt(sum_of_squares)
    return projection_norms

# def compute_projection_norms(patches, basis_vectors):
#     """
#     Compute projection norms of patches onto a set of basis vectors.

#     Parameters:
#     - patches: Array of shape (n_patches, 4096)
#     - basis_vectors: Array of shape (4096, 50)

#     Returns:
#     - projection_norms: Array of shape (n_patches,)
#     """
    
#    # Flatten basis_vectors if it has only one column (shape (4096, 1))
#     if basis_vectors.shape[1] == 1:
#         basis_vectors = basis_vectors.flatten()  # Shape becomes (4096,)

#     # Compute inner products
#     inner_products = np.dot(patches, basis_vectors)  # Shape: (n_patches, k) or (n_patches,)
    
#     # If single basis vector, ensure inner_products is 2D for consistent processing
#     if inner_products.ndim == 1:
#         inner_products = inner_products[:, np.newaxis]  # Shape becomes (n_patches, 1)

#     # Compute the sum of squares of inner products for each patch
#     sum_of_squares = np.sum(inner_products ** 2, axis=1)  # Shape: (n_patches,)

#     # Compute projection norms
#     projection_norms = np.sqrt(sum_of_squares)  # Shape: (n_patches,)

#     return projection_norms


def project_patches_onto_basis(patches, basis_vectors):
    n_patches = patches.shape[0]
    projected_patches = np.zeros((n_patches, 64, 64), dtype=np.float32)
    for i in range(n_patches):
        patch_flattened = patches[i].flatten()
        inner_products = np.dot(patch_flattened, basis_vectors)
        reconstructed_flat = np.dot(inner_products, basis_vectors.T)
        projected_patches[i] = reconstructed_flat.reshape((64, 64))
    return projected_patches


def subtract_orthogonal_images(sorted_steerable_basis_vectors_new, orthogonal_images_eigen_vectors):
    # Flatten the orthogonal_images_eigen_vectors from (50, 64, 64) to (50, 64*64)
    flattened_orthogonal_images = orthogonal_images_eigen_vectors.reshape(orthogonal_images_eigen_vectors.shape[0],
                                                                          orthogonal_images_eigen_vectors.shape[1] *
                                                                          orthogonal_images_eigen_vectors.shape[2])

    # Initialize result array with the same shape as sorted_steerable_basis_vectors_new
    result = np.copy(sorted_steerable_basis_vectors_new)

    # Loop over each vector in sorted_steerable_basis_vectors_new
    for i in range(sorted_steerable_basis_vectors_new.shape[1]):
        v_i = sorted_steerable_basis_vectors_new[:, i]

        # Initialize the sum of projections
        projection_sum = np.zeros_like(v_i)

        # Loop over each orthogonal image vector
        for u_j in flattened_orthogonal_images:
            # Compute the projection of v_i onto u_j
            projection_coefficient = np.dot(v_i, u_j)
            projection = projection_coefficient * u_j
            projection_sum += projection

        # Subtract the sum of projections from v_i
        result[:, i] = v_i - projection_sum

    return result


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


def extract_number(filename):
    match = re.search(r'\d+', filename)  # Find the first number in the filename
    return int(match.group()) if match else float('inf')  # Return the number or infinity if no number is found

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

def plot_patches_from_coordinates(Y, object_coord, microName, patch_size, obj_sz_real, obj_sz_down_scaled, flipud=False, downsample_fn=None):
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
    
    coordinates = object_coord
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
            patch_downsampled = downsample_fn(patch, (dSampleSz[0], dSampleSz[1]))
            patches.append(patch_downsampled)
        else:
            print(f"[DEBUG] Skipped patch {i} with shape {patch.shape} (not {patch_size}x{patch_size}).")


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

def plot_boxes_and_star_on_micrograph(micrograph_path, box_path, star_path, output_path=None, star_box_size=300):
    """
    Plots a .box file (with top-left corner + dimensions) and a .star file (with center coordinates + boxes)
    on a micrograph with different colors.
    
    Parameters:
    micrograph_path : str : Path to the micrograph image (.mrc file).
    box_path : str : Path to the .box file.
    star_path : str : Path to the .star file.
    output_path : str : Path to save the output image. If None, display the plot.
    star_box_size : int : Size (width and height) of the box to draw around the .star coordinates.
    """
    # Load the micrograph
    with mrcfile.open(micrograph_path, permissive=True) as mrc:
        micrograph = mrc.data  # Load the data array

    # Normalize the micrograph for better visualization
    #micrograph = (micrograph - np.min(micrograph)) / (np.max(micrograph) - np.min(micrograph))

    # Load coordinates from the .box file
    def load_box(file_path):
        # Load top-left corner and dimensions
        boxes = np.loadtxt(file_path, delimiter='\t', usecols=(0, 1, 2, 3))
        centers = []
        for x, y, w, h in boxes:
            x_center = x + w / 2
            y_center = y + h / 2
            centers.append((x_center, y_center, w, h))
        return np.array(centers)

    boxes = load_box(box_path)

    # Load coordinates from the .star file
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

    stars = load_star(star_path)

    # Plot the micrograph
    plt.figure(figsize=(10, 10))
    plt.imshow(micrograph, cmap='gray', origin='upper')

    # Plot the boxes from the .box file (red rectangles)
    for x_center, y_center, w, h in boxes:
        x_top_left = x_center - w / 2
        y_top_left = y_center - h / 2
        rect = plt.Rectangle((x_top_left, y_top_left), w, h, linewidth=1, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)

    # Plot the star coordinates (blue circles and boxes)
    for x, y in stars:
        print(f"[DEBUG] Plot: Star Coordinate: ({x}, {y})")
        plt.plot(x, y, 'o', color='blue', markersize=5)  # Blue dots
        # Blue box around the star coordinate
        x_top_left = x - star_box_size / 2
        y_top_left = y - star_box_size / 2
        rect = plt.Rectangle((x_top_left, y_top_left), star_box_size, star_box_size, linewidth=1, edgecolor='blue', facecolor='none')
        plt.gca().add_patch(rect)

    # Add a legend
    plt.legend(['.star coordinates (center + boxes)', '.box regions'], loc='upper right')

    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

        

def csv_to_box(input_csv, output_box, box_size, mgScale):
    """
    Convert CSV to BOX format for EMAN2, ensuring coordinates align with the micrograph.

    Parameters:
    input_csv : str : Path to input CSV file
    output_box : str : Path to output BOX file
    box_size : int : Box size for particles
    mgScale : float : Scaling factor for micrograph
    """
    # Load the CSV file (assuming columns: 'X-Coordinate', 'Y-Coordinate')
    df = pd.read_csv(input_csv)

    # Open the BOX file for writing
    with open(output_box, 'w') as f:
        for _, row in df.iterrows():
            # Scale and adjust for EMAN2 format
            x = row['X-Coordinate'] * mgScale
            y = row['Y-Coordinate'] * mgScale

            # Calculate top-left corner of the box
            x_topleft = x - box_size // 2
            y_topleft = y - box_size // 2

            # Write to .box file
            f.write(f"{x_topleft:.0f}\t{y_topleft:.0f}\t{box_size}\t{box_size}\n")

def process_all_csvs_to_box(input_dir, output_dir, box_size, mgScale):
    """
    Process all CSV files in a directory and convert them to BOX files for EMAN2.

    Parameters:
    input_dir : str : Directory containing input CSV files
    output_dir : str : Directory to save the output BOX files
    box_size : int : Box size for particles
    mgScale : float : Scaling factor for micrograph
    """
    os.makedirs(output_dir, exist_ok=True)
    for csv_file in os.listdir(input_dir):
        if csv_file.endswith('.csv'):
            microName = os.path.splitext(csv_file)[0]
            input_csv_path = os.path.join(input_dir, csv_file)
            output_box_path = os.path.join(output_dir, f"{microName}.box")
            csv_to_box(input_csv_path, output_box_path, box_size, mgScale)

            
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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




# def spatial_acovf_stack(images, max_lag=None, normalize=True):
#     """
#     Computes the 2D spatial autocovariance function for a **stationary** process
#     using a stack of images from the same Gaussian process.

#     Parameters:
#         images (ndarray): 3D array of shape (num_images, Nx, Ny).
#         max_lag (int, optional): Maximum lag to compute autocovariance.
#         normalize (bool): If True, normalizes by variance.

#     Returns:
#         acov (ndarray): 2D autocovariance function averaged over all images.
#     """
#     num_images, Nx, Ny = images.shape
#     mean = np.mean(images)  # Global mean over all images
#     variance = np.var(images)  # Global variance over all images

#     # Set maximum lag if not provided
#     if max_lag is None:
#         max_lag = min(Nx // 2, Ny // 2)

#     acov = np.zeros((2 * max_lag + 1, 2 * max_lag + 1))

#     # Compute autocovariance for each lag (τx, τy)
#     for tau_x in range(-max_lag, max_lag + 1):
#         for tau_y in range(-max_lag, max_lag + 1):
#             valid_pairs = 0
#             sum_cov = 0.0

#             for img in range(num_images):  # Iterate over images
#                 for i in range(Nx - abs(tau_x)):
#                     for j in range(Ny - abs(tau_y)):
#                         sum_cov += (images[img, i, j] - mean) * (images[img, i + tau_x, j + tau_y] - mean)
#                         valid_pairs += 1

#             acov[tau_x + max_lag, tau_y + max_lag] = sum_cov / valid_pairs

#     # Normalize by variance if needed
#     if normalize:
#         acov /= variance

#     return acov




# def spatial_acovf_stack_fft(images, max_lag=None, normalize=True):
#     """
#     Computes the 2D spatial autocovariance function for a **stationary** process
#     using a stack of images from the same Gaussian process, optimized with FFT.

#     Parameters:
#         images (ndarray): 3D array of shape (num_images, Nx, Ny).
#         max_lag (int, optional): Maximum lag to compute autocovariance.
#         normalize (bool): If True, normalizes by variance.

#     Returns:
#         acov (ndarray): 2D autocovariance function averaged over all images.
#     """
#     num_images, Nx, Ny = images.shape
#     mean = np.mean(images)  # Global mean over all images
#     variance = np.var(images)  # Global variance over all images

#     # Set maximum lag if not provided
#     if max_lag is None:
#         max_lag = min(Nx // 2, Ny // 2)

#     # Subtract the mean for zero-mean process
#     images_centered = images - mean

#     # Compute autocovariance using FFT-based correlation
#     acov = np.zeros((2 * max_lag + 1, 2 * max_lag + 1))

#     for img in range(num_images):
#         corr = correlate2d(images_centered[img], images_centered[img], mode='full', boundary='wrap')  # Periodic BC
#         center = corr.shape[0] // 2, corr.shape[1] // 2  # Center index

#         # Extract the relevant region
#         acov += corr[center[0] - max_lag:center[0] + max_lag + 1,
#                      center[1] - max_lag:center[1] + max_lag + 1]

#     # Average over all images
#     acov /= num_images

#     # Normalize by variance if needed
#     if normalize:
#         acov /= variance

#     return acov


def select_best_gpu(min_free_memory=4 * 1024**3):
    best_gpu = None
    best_free_mem = 0
    num_gpus = cp.cuda.runtime.getDeviceCount()

    for i in range(num_gpus):
        try:
            free_mem, total_mem = cp.cuda.Device(i).mem_info
            if free_mem > min_free_memory and free_mem > best_free_mem:
                best_gpu = i
                best_free_mem = free_mem
        except cp.cuda.memory.OutOfMemoryError:
            continue

    return best_gpu if best_gpu is not None else 0  # Default to GPU 0 if none are available






def spatial_acovf_stack_gpu(images, max_lag=None, normalize=True):
    """
    Computes the 2D spatial autocovariance function for a **stationary** process
    using a stack of images from the same Gaussian process, optimized for GPU with CuPy.

    Parameters:
        images (ndarray): 3D NumPy array of shape (num_images, Nx, Ny).
        max_lag (int, optional): Maximum lag to compute autocovariance.
        normalize (bool): If True, normalizes by variance.

    Returns:
        acov (ndarray): 2D autocovariance function averaged over all images.
    """

    # Free GPU memory
    cp.cuda.Device(1).use()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()
    
    # Move data to GPU
    images_gpu = cp.asarray(images, dtype=cp.float32)
    num_images, Nx, Ny = images_gpu.shape

    # Compute mean and variance **per image**
    mean_per_image = cp.mean(images_gpu, axis=(1, 2), keepdims=True)
    images_centered = images_gpu - mean_per_image  # Subtract mean per image

    variance = cp.var(images_centered, axis=(1, 2))  # Variance per image
    variance_mean = cp.mean(variance)  # Average variance over images

    print(f"Variance per image (first 5): {variance[:5]}")
    print(f"Mean variance across images: {variance_mean}")

    # Check for negative variance
    if cp.any(variance < 0):
        raise ValueError("❌ Computed variance is negative! This should never happen.")

    # Set maximum lag if not provided
    if max_lag is None:
        max_lag = min(Nx // 2, Ny // 2)

    # Allocate memory for autocovariance
    acov = cp.zeros((2 * max_lag + 1, 2 * max_lag + 1), dtype=cp.float32)

    # Compute autocovariance using correlate2d
    for img in range(num_images):
        corr = correlate2d(images_centered[img], images_centered[img], mode='same', boundary='wrap')  # Avoid artifacts
        center = corr.shape[0] // 2, corr.shape[1] // 2  # Center index

        # Extract the relevant region
        acov += corr[center[0] - max_lag:center[0] + max_lag + 1,
                     center[1] - max_lag:center[1] + max_lag + 1]

    # Average over all images
    # Normalize correctly
    acov /= (num_images * Nx * Ny)  # Normalize by number of pixels


    # Print central value before normalization
    print(f"acov[0,0] before normalization: {acov[max_lag, max_lag]} (should match variance_mean)")
    x=1
    # Normalize by variance if needed
    if normalize:
        acov /= variance_mean
        print(f"acov[0,0] after normalization: {acov[max_lag, max_lag]} (should be 1)")

    # Move back to CPU
    return cp.asnumpy(acov)



def full_cov_from_autocov(acov):
    """
    Constructs the full covariance matrix (H*W, H*W) from the stationary autocovariance.
    Correctly centers the autocovariance so that the variance appears at index (0,0).

    Parameters:
        acov (cp.ndarray): Centered Autocovariance function of size (H, W),
                           where acov[max_lag, max_lag] corresponds to zero lag.

    Returns:
        cov_matrix (cp.ndarray): Full covariance matrix of shape (H*W, H*W).
    """
    cp.cuda.Device(4).use()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()
    H, W = acov.shape
    max_lag = H // 2  # Assuming autocovariance is square and centered

    # Ensure acov is on GPU
    acov = cp.asarray(acov)

    # Shift the autocovariance so that acov[max_lag, max_lag] (zero lag) aligns with index (0,0)
    acov_centered = cp.roll(acov, shift=(-max_lag, -max_lag), axis=(0, 1))

    # Construct block Toeplitz structure using CuPy
    blocks = []
    for i in range(H):
        row = cp.asarray(acov_centered[i, :])  # Convert each row to CuPy before passing to `toeplitz`
        blocks.append(toeplitz(row))

    # Properly stack blocks into a large matrix
    cov_matrix = cp.vstack([cp.hstack([blocks[i - j] if (0 <= i - j < H) else cp.zeros((W, W)) for j in range(H)]) for i in range(H)])

    return cov_matrix