# Standard Library Imports


# Parameters to change each datasets
import os
obj_sz_real = 300
micrograph_name_for_basis = None  # f None, the basis will be compute from each micrograph separately. with .mrc
# basis method 0 unsupervised, 1 2dclass
basis_method = 0
# contamination detection masks
use_contamination_datection = 0  # 0 no mask, 1 mask
# Directory paths
micrograph_directory = './data/10028/'
object_coord_dir = './data/10028'
output_folder_bh = './results/10028/'
cont_masks_directory = './'
os.makedirs(micrograph_directory, exist_ok=True)
os.makedirs(output_folder_bh, exist_ok=True)
use_gpu = 0  # 0 no gpu, 1 gpu

if use_gpu==0:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU before TensorFlow is imported
    import tensorflow as tf  # Now import TensorFlow

import math
import tensorflow as tf  # Now import TensorFlow
# from aspire.basis import Coef, FBBasis2D
# from aspire.image import Image as AspireImage
from scipy.sparse.linalg import eigsh
import numpy as np
import mrcfile
# Custom Modules
# from common import steerable_utils_new
# from common.steerable_utils_new import *
from common import functions
from common.functions import *
import logging
# Redirect logging to nowhere
logging.getLogger('aspire').handlers = [logging.FileHandler(os.devnull)]

# Parameters
alpha = 0.05  #BH
delta = 1  #Centering

# Object


obj_sz_down_scaled = 64
mgScale = obj_sz_down_scaled / obj_sz_real
obj_sz_plots = obj_sz_real
obj_sz_down_scaled_plots = obj_sz_plots * mgScale
# Side length of the algorithm
sideLengthAlgorithm = math.ceil(2 * (obj_sz_down_scaled) + delta)
golden_ratio = 2 / 3
sideLengthAlgorithmL = golden_ratio * sideLengthAlgorithm
# Steerable basis paramerters
l_max_objects = 20  # Maximum order of the Fourier-Bessel basis
num_of_objects = 30  # Number of objects to use for basis computation
num_of_noise_patches = 30  # Number of noise patches to use for basis computation
# compute_basis_from_specific_micrograph

# in the end. I
# noise simulation
box_sz_Sz = np.min([int(sideLengthAlgorithmL / 2), 100])  # box size for S_z
noise_patch_sz_sim = box_sz_Sz + obj_sz_down_scaled  # box_sz needed due to convulution of S_z
num_noise_patch_sim = 15  # number of noise patches to simulate the noise from



class_adress_10028 = './class_averages.mrcs'





# process_all_csvs_to_box('./data/particle_coordinates/', './data/particle_coordinates/', 300, 1, 7676)

# Main
first_run_for_basis = 1  # This is just a flag to implement computing basis from a specific micrograph.

# Initialize the Fourier-Bessel basis (independent of the micrograph)
fb_basis_objects = FBBasis2D(size=(obj_sz_down_scaled, obj_sz_down_scaled), ell_max=l_max_objects)

# Step 1: Load the micrographs
# Get the list of micrographs in micrograph_directory
files_micro = [f for f in os.listdir(micrograph_directory) if f.endswith('.mrc')]
# Check and reorder files_micro
if micrograph_name_for_basis is not None:
    if micrograph_name_for_basis in files_micro:
        # Remove the element and insert it at the first position
        files_micro.remove(micrograph_name_for_basis)
        files_micro.insert(0, micrograph_name_for_basis)
        compute_basis_from_specific_micrograph = True
    else:
        print(f"Warning: '{micrograph_name_for_basis}' is not in the list of micrographs")
        compute_basis_from_specific_micrograph = False
else:
    compute_basis_from_specific_micrograph = False

for mrc_file in files_micro:
    file_path = os.path.join(micrograph_directory, mrc_file)
    if use_contamination_datection:
        file_path_mask = os.path.join(cont_masks_directory, mrc_file)
    microName = mrc_file.replace(".mrc", "")  # Extract the micrograph name
    with mrcfile.open(file_path, permissive=True) as mrc:

        mgBig = mrc.data
        mgBigSz = mgBig.shape
        mgScale = obj_sz_down_scaled / obj_sz_real
        mic_sz = 4000  # micrograph size
        M_L_est = ((mgScale * max(mgBigSz)) / (box_sz_Sz / np.sqrt(2))) ** 2
        # Think about it
        num_of_exp_noise = np.min(
            [10 ** 3, int(0.2 * M_L_est / alpha)])  # Test value, should be: 1/num_of_exp_noise~alpha/M_L

        dSampleSz = (int(np.floor(mgScale * mgBig.shape[0])), int(np.floor(mgScale * mgBig.shape[1])))
        Y = downsample(mgBig, (dSampleSz[1], dSampleSz[0]))
    # contamination detection
    if use_contamination_datection:
        with mrcfile.open(file_path_mask, permissive=True) as mrc_mask:
            mgBig_mask = np.flipud(mrc_mask.data)
            # contamination_mask = AspireImage(mgBig_mask).downsample(dSampleSz[0]).asnumpy()[0]
            contamination_mask = downsample(mgBig_mask, (dSampleSz[0], dSampleSz[1]))
            contamination_mask_binary = contamination_mask.copy()
            threshold = 0.5
            contamination_mask_binary[contamination_mask_binary < threshold] = 0
            contamination_mask_binary[contamination_mask_binary >= threshold] = 1
    else:
        contamination_mask_binary = np.zeros(Y.shape)


    if basis_method == 0:
        if first_run_for_basis:
            if compute_basis_from_specific_micrograph:
                first_run_for_basis = 0

            # extract noise_patches for denoising the objects
            noise_patches, noise_patches_coor = extract_noise_patches_and_coor_return_min_variance_tf(Y, contamination_mask_binary, obj_sz_down_scaled, num_of_noise_patches)
            num_patches, height, width = noise_patches.shape
            mean_noise_image = np.mean(noise_patches)
            plot_patches_on_micrograph(Y, noise_patches_coor, obj_sz_down_scaled, microName, output_folder=None)
            # Step 2: Compute the steerable basis or use 2dclasses
            # Extract object patches
            objects_down_scaled = extract_patches_from_any(Y, object_coord_dir, microName, obj_sz_down_scaled, mgScale,contamination_mask_binary)
            # Compute the steerable basis
            eigen_vectors_per_ang_lst, mean_noise_per_ang_lst = fourier_bessel_pca_per_angle(noise_patches,
                                                                                             fb_basis_objects)
            steerable_euclidian_l_new_objects, denoised_objects = compute_the_steerable_images(
                objects_down_scaled[:num_of_objects, :, :],
                obj_sz_down_scaled,
                fb_basis_objects,
                eigen_vectors_per_ang_lst,
                mean_noise_per_ang_lst)
            steerable_basis_vectors = compute_the_steerable_basis(steerable_euclidian_l_new_objects)

            flattened_noise_patches = noise_patches.reshape(num_patches, -1).T - mean_noise_image
            flattened_denoised_objects = denoised_objects.reshape(denoised_objects.shape[0], -1).T
            for n in range(flattened_noise_patches.shape[1]):
                flattened_noise_patches[:, n] = flattened_noise_patches[:, n] / np.linalg.norm(
                    flattened_noise_patches[:, n])
            for n in range(flattened_denoised_objects.shape[1]):
                flattened_denoised_objects[:, n] = flattened_denoised_objects[:, n] / np.linalg.norm(
                    flattened_denoised_objects[:, n])

            sorted_basis_vectors_full, num_of_basis, projected_snr_per_dim, basis_idx = sort_steerable_basis_by_obj_then_snr(
                steerable_basis_vectors, flattened_denoised_objects, flattened_noise_patches,
                100)
            # plt.plot(projected_snr_per_dim)
            # plt.show()
            sorted_steerable_basis_vectors = sorted_basis_vectors_full[:, :num_of_basis + 1]
            sorted_basis_images = np.reshape(sorted_steerable_basis_vectors, (
                obj_sz_down_scaled, obj_sz_down_scaled, sorted_steerable_basis_vectors.shape[1]))

    if basis_method == 1:
        if first_run_for_basis:
            first_run_for_basis = 0
            # Step 2: Compute the steerable basis or use 2dclasses
            with mrcfile.open(class_adress_10028, permissive=True) as class_2d:
                class_2d_img = class_2d.data
                class_2d_img = class_2d_img.transpose(1, 2, 0)
                mgScale = obj_sz_down_scaled / class_2d_img.shape[0]
                dSampleSz = (
                    int(np.floor(mgScale * class_2d_img.shape[0])), int(np.floor(mgScale * class_2d_img.shape[1])))
                # Y = AspireImage(mgBig).downsample(dSampleSz[0]).asnumpy()[0]
                class_2d_img_scaled = np.zeros(
                    downsample(class_2d_img[:, :, 0], (dSampleSz[0], dSampleSz[1])).shape + (class_2d_img.shape[2],))
                for i_img in range(class_2d_img.shape[2]):
                    class_2d_img_scaled[:, :, i_img] = downsample(class_2d_img[:, :, i_img],
                                                                  (dSampleSz[0], dSampleSz[1]))
                matrix = class_2d_img_scaled.reshape(-1, class_2d_img_scaled.shape[2])
                Q, R = np.linalg.qr(matrix, mode='reduced')
                sorted_basis_images = Q.reshape(class_2d_img_scaled.shape[0], class_2d_img_scaled.shape[1],
                                                class_2d_img_scaled.shape[2])

    # Step 3: Simulate noise
    print("Simulating noise")
    noise_patch_diam_scaled = int(np.floor(obj_sz_real * mgScale))
    num_of_patches = ((Y.shape[0] - obj_sz_down_scaled) // (obj_sz_down_scaled)) ** 2
    noise_patches_sim, coor_sim = extract_noise_patches_and_coor_return_min_variance_tf(Y, contamination_mask_binary,
                                                                                        noise_patch_sz_sim,
                                                                                        num_noise_patch_sim)
    plot_patches_on_micrograph(Y, coor_sim, noise_patch_sz_sim, microName, output_folder=None)
    n_images, img_height, img_width = noise_patches_sim.shape
    flattened_images = noise_patches_sim.reshape(n_images, (img_height) * (img_width))  # Shape: (n_images, n_pixels)
    covariance_matrix = np.cov(flattened_images, rowvar=False)
    eigenvalues, eigenvectors = eigsh(covariance_matrix, k=n_images - 1, which='LM')  # 'LM' -> Largest Magnitude
    # plt.plot(eigenvalues)
    # plt.show()
    Lambda_sqrt = np.diag(np.sqrt(eigenvalues))
    Z = np.random.randn(len(eigenvalues), num_of_exp_noise)
    noise_vec_sim = (eigenvectors @ Lambda_sqrt) @ Z
    # Simulate S_z from noise patches
    S_z_tf = projected_noise_simulation_from_noise_patches_tf(noise_vec_sim, sorted_basis_images, num_of_exp_noise)
    z_max = np.max(S_z_tf, axis=(0, 1)).reshape(-1, 1)

    # Step 4: Compute Peaks
    centerd_Y = Y - np.mean(noise_patches_sim)
    Y_peaks, Y_peaks_loc, S = peak_algorithm_cont_mask_tf(
        centerd_Y, sorted_basis_images, np.floor(sideLengthAlgorithmL),contamination_mask_binary,
        obj_sz_down_scaled=obj_sz_down_scaled)
    plt.plot(z_max)
    plt.plot(Y_peaks)
    plt.show()

    # Step 5: BH algorithm
    test_val = test_function_real_data(z_max, Y_peaks)
    M_L = (Y.shape[0] * Y.shape[1]) / ((S_z_tf.shape[0] / (np.sqrt(2))) ** 2)

    K_bh = BH(test_val, alpha, M_L)
    print(microName, K_bh, sorted_basis_images.shape[2])

    # Step 6: Save the coordinates and plots
    coords_output(Y_peaks_loc, output_folder_bh, microName, mgScale, mgBigSz, K_bh, obj_sz_real)
    plot_and_save(S, circles=None, obj_sz_down_scaled=obj_sz_down_scaled,
                                      filename=f'scoringMap_{microName}.jpg', output_folder=output_folder_bh)
    plot_and_save(Y, circles=None, obj_sz_down_scaled=obj_sz_down_scaled,
                                      filename=f'Y_{microName}.jpg',
                                      output_folder=output_folder_bh)
    plot_and_save(Y, circles=np.flip(Y_peaks_loc[:K_bh, :], axis=1),
                                      obj_sz_down_scaled=obj_sz_down_scaled_plots,
                                      filename=f'circles_with_bh_{microName}.jpg',
                                      output_folder=output_folder_bh)
    plot_and_save(Y, circles=np.flip(Y_peaks_loc, axis=1), obj_sz_down_scaled=obj_sz_down_scaled,
                                      filename=f'circles_without_bh_{microName}.jpg', output_folder=output_folder_bh)
