## This is the main code used as a proof of cnמcept
import os.path
import os.path
from aspire.image import *
from aspire.basis import FBBasis2D
import os
from scipy.ndimage import rotate
from scipy.interpolate import interp1d
from numpy.fft import fft2, ifft2
from scipy.linalg import eigh
from scipy.signal import convolve
from scipy.signal import convolve2d
import scipy.linalg
import matplotlib.pyplot as plt
from numba import jit, prange
from scipy.spatial.distance import pdist, squareform
import pickle
from matplotlib.patches import Circle
from scipy.signal import convolve2d
import tensorflow as tf
from scipy.signal import wiener
# Set device to MPS if available
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
import os







def read_coordinates(coord_file):
    """
    Read coordinates from a .box, .star, or .csv file.

    Parameters:
    coord_file : str
        Path to the coordinate file.

    Returns:
    coordinates : list of tuple
        Array of coordinates [(x, y), ...].
    file_type : str
        'box', 'star', or 'csv' depending on the file type.
    """
    coordinates = []
    if coord_file.endswith('.box'):
        file_type = 'box'
    elif coord_file.endswith('.star'):
        file_type = 'star'
    elif coord_file.endswith('.csv'):
        file_type = 'csv'
    else:
        raise ValueError(
            "Unsupported file type. Please use .box, .star, or .csv files.")

    with open(coord_file, 'r') as f:
        for line in f:
            tokens = line.strip().split(',')

            # Skip header or invalid lines
            if file_type in ['box', 'csv'] and not tokens[0].replace('.', '', 1).isdigit():
                continue
            if file_type == 'star' and line.startswith('_'):
                continue

            try:
                if file_type == 'box':
                    # Box files store the top-left corner; convert to center coordinates
                    if len(tokens) < 4:
                        continue
                    x, y, box_size_x, box_size_y = int(tokens[0]), int(
                        tokens[1]), int(tokens[2]), int(tokens[3])
                    x_center = x + box_size_x // 2
                    y_center = y + box_size_y // 2
                    coordinates.append((x_center, y_center))
                elif file_type == 'star':
                    # Star files typically store the center directly
                    if len(tokens) < 2:
                        continue
                    x, y = int(float(tokens[0])), int(float(tokens[1]))
                    coordinates.append((x, y))
                elif file_type == 'csv':
                    # Assuming csv format contains center coordinates in two columns: x, y
                    if len(tokens) < 2:
                        continue
                    x, y = int(float(tokens[0])), int(float(tokens[1]))
                    coordinates.append((x, y))
            except ValueError:
                # Skip lines with invalid numerical values
                continue

    if not coordinates:
        print(f"No valid coordinates found in {coord_file}.")
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
    
def ensure_positive_definite(matrix, epsilon=1e-6, max_attempts=10):
    """
    Ensure the input matrix is positive definite by attempting to add a small value to the diagonal.

    Parameters:
    matrix : ndarray
        Input matrix to check.
    epsilon : float, optional
        Initial value to add to the diagonal if the matrix is not positive definite.
    max_attempts : int, optional
        Maximum number of attempts to adjust the matrix.

    Returns:
    matrix : ndarray
        Adjusted positive definite matrix.

    Raises:
    LinAlgError:
        If the matrix cannot be made positive definite after max_attempts.
    """
    for attempt in range(max_attempts):
        try:
            _ = scipy.linalg.cholesky(matrix, lower=True)
            return matrix
        except np.linalg.LinAlgError:
            diag_increment = epsilon * (10 ** attempt)  # Increase epsilon exponentially
            matrix += np.eye(matrix.shape[0]) * diag_increment
            print(f"Matrix adjusted with diagonal increment: {diag_increment:.2e}")
    
    raise np.linalg.LinAlgError("Matrix could not be made positive definite after multiple attempts.")


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
        if p_val[-(l + 1)] <= ((len(p_val) - l) * alpha) / M_L:
            K = len(p_val) - l
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



def compute_projection_norms(patches, basis_vectors):
    """
    Computes the norm of the projection of each patch onto the given 30 basis vectors.
    Patches are of shape (64, 64), and basis_vectors is of shape (64, 64, 30).
    """
    n_patches = patches.shape[0]
    n_basis = basis_vectors.shape[2]  # Number of basis vectors (30)

    projection_norms = np.zeros(n_patches)
    for i in range(n_patches):
        patch_flattened = patches[i].flatten()

        # For each basis vector, compute the projection
        inner_products = np.zeros(n_basis)
        for j in range(n_basis):
            basis_flattened = basis_vectors[:, :, j].flatten()
            inner_products[j] = np.dot(patch_flattened, basis_flattened)

        # Compute the norm of the projection onto the basis set
        sum_of_squares = np.sum(inner_products ** 2)
        projection_norms[i] = np.sqrt(sum_of_squares)

    return projection_norms



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
            img_copy = img.copy()
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

    # Reshape the array to add the channel dimension: (num_of_exp_noise, sz, sz, 1)
    noise_imgs_tf = tf.convert_to_tensor(noise_imgs, dtype=tf.float32)

    for j in range(basis.shape[2]):
        flipped_basis = np.flip(np.flip(basis[:, :, j], 0), 1)  # Flip the basis along both axes
        flipped_basis_tf = tf.convert_to_tensor(
            tf.reshape(flipped_basis, (basis.shape[0], basis.shape[1], 1, 1)),
            dtype=tf.float32
        )

        # Perform the convolution using tf.nn.conv2d
        conv_result = tf.nn.conv2d(noise_imgs_tf, flipped_basis_tf, strides=[1, 1, 1, 1], padding='VALID')**2

        # Accumulate the sum of squared results
        S_z_n += tf.squeeze(conv_result).numpy()  # Convert TensorFlow tensor to NumPy array

        # Explicitly release GPU memory for the basis tensor and convolution result
       # del flipped_basis_tf, conv_result
        #tf.keras.backend.clear_session()
        #gc.collect()

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

    # Extract unique distances and their indices
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
    Q, R = np.linalg.qr(flattened_matrices, mode='reduced')
    Q_dom = dominant_vectors(Q, R)
    return Q_dom
def dominant_vectors(Q, R):
    # Calculate the absolute maximum value on the diagonal of R
    max_diag_value = np.max(np.abs(np.diag(R)))

    # Set the threshold based on this maximum value
    adapted_threshold = 0.0001 * max_diag_value

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



def sort_steerable_basis_keren(class_averages, steerable_basis_vectors, obj_sz, gamma):
    num_templates = class_averages.shape[0]
    num_basis_vectors = steerable_basis_vectors.shape[1]
    coeff_dict = {}

    # Compute coefficients (inner products squared) and store in a dictionary
    for n_img in range(num_templates):
        img = Image(class_averages[n_img])
        scaled_img = img.downsample(obj_sz)
        scaled_img = scaled_img.asnumpy()[0]
        coeffs = {}
        for i in range(num_basis_vectors):
            basis_vector = steerable_basis_vectors[:, i]
            coeff = np.dot(basis_vector.T, scaled_img.flatten())
            coeffs[i] = coeff**2
        coeff_dict[n_img] = coeffs

    # Calculate the original norms of the templates
    original_norms = {}
    for n_img in range(num_templates):
        img = Image(class_averages[n_img])
        scaled_img = img.downsample(obj_sz)
        scaled_img = scaled_img.asnumpy()[0]
        original_norms[n_img] = np.linalg.norm(scaled_img.flatten())

    # Select the template with the minimal original norm
    f_idx = min(original_norms, key=original_norms.get)

    selected_vectors = []
    projected_norms = []

    # Iterative basis vector selection
    while True:
        # Find the largest coefficient squared for the current template
        largest_coeff_idx = max(coeff_dict[f_idx], key=coeff_dict[f_idx].get)
        selected_vectors.append(largest_coeff_idx)

        # Check stopping condition
        selected_basis = steerable_basis_vectors[:, selected_vectors]
        projected_templates = np.array([np.dot(selected_basis, [coeffs[idx] for idx in selected_vectors]) for coeffs in coeff_dict.values()])
        projected_norms = np.linalg.norm(projected_templates, axis=1)
        min_projected_norm = np.min(projected_norms)
        if min_projected_norm > gamma * np.min(list(original_norms.values())):
            break

        # Update coefficients for the remaining templates
        for coeffs in coeff_dict.values():
            coeffs.pop(largest_coeff_idx, None)

        # Find the next template f_idx to update based on the minimum of maximum coefficients
        max_coeffs_per_template = {n_img: coeff_dict[n_img][largest_coeff_idx] for n_img in coeff_dict if largest_coeff_idx in coeff_dict[n_img]}
        f_idx = min(max_coeffs_per_template, key=max_coeffs_per_template.get)

    # Return the selected basis vectors
    sorted_steerable_basis_vectors = steerable_basis_vectors[:, selected_vectors]
    return sorted_steerable_basis_vectors

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


def compute_the_steerable_images(class_averages,obj_sz,l_max,norm_err):
    # compute the radial Fourier component to each class average
    fb_basis = FBBasis2D(size=(obj_sz, obj_sz), ell_max=l_max)
    # get the coefficients of the class averages in the FB basis
    steerable_euclidian_l = np.zeros((obj_sz, obj_sz, 1+2*(fb_basis.ell_max), class_averages.shape[0]))
    for n_img in range(class_averages.shape[0]):
        img = Image(class_averages[n_img])
        scaled_img = np.asarray(img.downsample(obj_sz)).astype(np.float32)
        img_fb_coeff = fb_basis.evaluate_t(scaled_img)
        v_0 = img_fb_coeff.copy()
        v_0._data[0,fb_basis.k_max[0]:] = 0
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
            vsin = img_fb_coeff.copy()
            vsin._data[0,:coeff_k_index_end_cos]=0
            vsin._data[0,coeff_k_index_end_sin:] = 0
            vcos_img = fb_basis.evaluate(vcos).asnumpy()[0]
            vsin_img = fb_basis.evaluate(vsin).asnumpy()[0]
            steerable_euclidian_l[:,:,l_idx,n_img] = vcos_img
            steerable_euclidian_l[:, :, l_idx+1, n_img] = vsin_img
            #steerable_euclidian_l[:,:,l_idx,n_img] = vcos_img + 1j*vsin_img
            #steerable_euclidian_l[:, :, l_idx+1, n_img] = vcos_img - 1j * vsin_img
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

