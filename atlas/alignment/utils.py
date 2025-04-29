"""
Utility functions for alignment tasks.
"""

import numpy as np
import pandas as pd
from typing import NamedTuple
import tifffile as tiff
from skimage.registration import phase_cross_correlation
from skimage.transform import downscale_local_mean

class ROI(NamedTuple):
    """
    Defines a Region of Interest (ROI) with x and y boundaries.
    """
    x0: int
    x1: int
    y0: int
    y1: int

def first_last_true(bool_vector):
    """
    Finds the first and last indices where a boolean vector is True.

    Parameters:
    ----------
    bool_vector : np.ndarray
        Boolean 1D numpy array.
    
    Returns:
    -------
    tuple (int or None, int or None)
        - Index of the first True value.
        - Index of the last True value.
        - Returns (None, None) if no True values are found.
    """
    assert isinstance(bool_vector, np.ndarray), "bool_vector must be a numpy array"
    assert bool_vector.dtype == np.bool_, "bool_vector must be a boolean array"
    
    if np.any(bool_vector):
        first_true = np.argmax(bool_vector)
        last_true = len(bool_vector) - 1 - np.argmax(np.flip(bool_vector))
    else:
        first_true, last_true = None, None
    
    return first_true, last_true

def check_roi_limits(img_shape, ROI_input):
    """
    Checks if the given ROI fits within the image dimensions.
    If the ROI exceeds the image boundaries, the function suggests correction values.
    
    Parameters:
    ----------
    img_shape : tuple
        Shape of the image as (height, width).
    ROI_input : ROI
        NamedTuple defining the ROI with (x0, x1, y0, y1).
    
    Returns:
    -------
    tuple (int, int, int, int)
        - Adjustments needed for x0, x1, y0, y1 to fit within the image.
    """
    assert isinstance(img_shape, tuple) and len(img_shape) == 2, "img_shape must be a tuple (height, width)"
    assert isinstance(ROI_input, ROI), "ROI_input must be an instance of ROI"
    
    max_y, max_x = img_shape
    d_y1 = max(0, ROI_input.y1 - max_y)
    d_y0 = max(0, -ROI_input.y0)
    d_x1 = max(0, ROI_input.x1 - max_x)
    d_x0 = max(0, -ROI_input.x0)
    
    return d_x0, d_x1, d_y0, d_y1

from atlas.image_analysis.utils import mask_low_and_saturation

def get_translation(reference_img, moving_img):
    """
    Computes the translation shift between two images using phase cross-correlation.

    This function first determines the largest central object in both images to focus 
    the alignment on relevant regions. It returns the Regions of Interest (ROI) used 
    for alignment and the computed translation shift.

    Parameters:
    ----------
    reference_img : np.ndarray
        The reference image.
    moving_img : np.ndarray
        The moving image to be aligned.
    
    Returns:
    -------
    ROI
        The ROI used from the reference image.
    ROI
        The ROI used from the moving image.
    np.ndarray
        The final pixel shift (row, col) needed to align the moving image to the reference.
    """
    assert isinstance(reference_img, np.ndarray), "reference_img must be a numpy array"
    assert isinstance(moving_img, np.ndarray), "moving_img must be a numpy array"
    
    # Create masks for saturation and low values
    mask_ref = np.logical_not(mask_low_and_saturation(reference_img))
    mov_mask = np.logical_not(mask_low_and_saturation(moving_img))

    # find object in the centre of the image
    x0, x1 = first_last_true(np.any(mask_ref, axis=0))
    y0, y1 = first_last_true(np.any(mask_ref, axis=1))
    ref_ROI = ROI(x0=x0, x1=x1, y0=y0, y1=y1)

    x0, x1 = first_last_true(np.any(mov_mask, axis=0))
    y0, y1 = first_last_true(np.any(mov_mask, axis=1))
    mov_ROI = ROI(x0=x0, x1=x1, y0=y0, y1=y1)

    crop_shift = np.array([ref_ROI.y0 -mov_ROI.y0, ref_ROI.x0 - mov_ROI.x0])
    print(f"crop pixel shift: {crop_shift}")

    # crop, so we focus only where we have data
    crop_ref_img  = reference_img[ref_ROI.y0:ref_ROI.y1,ref_ROI.x0:ref_ROI.x1]
    crop_ref_mask = mask_ref[ref_ROI.y0:ref_ROI.y1,ref_ROI.x0:ref_ROI.x1]

    crop_mov_img = moving_img[mov_ROI.y0:mov_ROI.y1,mov_ROI.x0:mov_ROI.x1]
    crop_mov_mask = mov_mask[mov_ROI.y0:mov_ROI.y1,mov_ROI.x0:mov_ROI.x1]

    # Compute phase cross-correlation shift
    detected_shift, _, _ = phase_cross_correlation(crop_ref_img, crop_mov_img, 
                                                reference_mask=crop_ref_mask, moving_mask=crop_mov_mask, 
                                                upsample_factor=1)

    detected_shift = np.round(detected_shift)
    print(f"Detected pixel offset based on crops (row, col): {detected_shift}")

    current_shift = detected_shift + crop_shift

    return ref_ROI, mov_ROI, current_shift

def initialize_alignment_df(tif_list_sorted, down_scale=10):
    """
    Initializes a DataFrame to store results for z-alignment processing.

    This function creates a structured DataFrame for storing image alignment data,
    including paths to images, downscaling factor, region of interest (ROI) details,
    and translation shifts.

    Parameters:
    ----------
    tif_list_sorted : list of Path
        A sorted list of file paths to the TIFF images to be processed.
    down_scale : int, optional
        The downscaling factor used to speed up pairwise image comparisons (default: 10).

    Returns:
    -------
    pd.DataFrame
        A DataFrame with the following columns:
        - "moving_path": Path to the current (moving) image.
        - "reference_path": Path to the reference image.
        - "moving_ROI": ROI NamedTuple for the moving image.
        - "reference_ROI": ROI NamedTuple for the reference image.
        - "current_shift": np.ndarray (shape: (2,)) for row and column shift.
        - "cumulative_ROI": ROI NamedTuple representing the cumulative ROI shift.
        - "cumulative_shift": np.ndarray (shape: (2,)) for accumulated shift.
        - "down_scale": int, the downscaling factor used.

    Notes:
    ------
    - The first row (index 0) has the **same moving and reference image**.
    - `current_shift` and `cumulative_shift` start as `[0, 0]` NumPy arrays.
    - `ROI` values are initialized as `None` and should be updated later.
    """

    num_images = len(tif_list_sorted)

    # ✅ Ensure correct input type
    if not isinstance(tif_list_sorted, list):
        raise TypeError("Expected 'tif_list_sorted' to be a list of file paths.")

    # Define column names and data types
    z_alignment_df = pd.DataFrame({
        #"index": pd.Series(np.arange(num_rows), dtype=int),
        "moving_path": pd.Series([None] * num_images, dtype="object"),  # Paths stored as objects
        "reference_path": pd.Series([None] * num_images, dtype="object"),
        "moving_ROI": pd.Series([None] * num_images, dtype="object"),  # ROI is a NamedTuple, so store as object
        "reference_ROI": pd.Series([None] * num_images, dtype="object"),
        "current_shift": pd.Series([np.array([0., 0.])] * num_images, dtype="object"),
        "cumulative_ROI": pd.Series([None] * num_images, dtype="object"),
        "cumulative_shift": pd.Series([np.array([0., 0.])] * num_images, dtype="object"),  # NumPy array of 2 integers
        "down_scale": pd.Series([None] * num_images, dtype="object"),  # NaN-filled float column
    })
    # add information about corresponding images, at the moment this is based on the sorted tif names
    for index in range(z_alignment_df.shape[0]):
        # get moving image
        mov_index = index
        mov_path = tif_list_sorted[mov_index]
        # we consider here the special case of index 0, in this case moving and reference image are the same.
        if index==0:
            ref_index = 0
        else:
            ref_index = index-1
        ref_path = tif_list_sorted[ref_index]
        
        # ✅ Now fill the DataFrame
        # Fill DataFrame efficiently
        z_alignment_df.loc[index, "moving_path"] = mov_path
        z_alignment_df.loc[index, "reference_path"] = ref_path
        z_alignment_df.loc[index, "down_scale"] = down_scale

    return z_alignment_df

def pairwise_alignment(alignment_df):
    """
    Computes the pairwise alignment shifts between moving and reference images.

    This function iterates through the provided DataFrame, loads the reference and moving images,
    applies downscaling (with per-image factors stored in the DataFrame), and computes the translation shift.
    The computed shifts and updated Regions of Interest (ROIs) are stored back in the DataFrame.

    Parameters:
    ----------
    alignment_df : pd.DataFrame
        A Pandas DataFrame containing at least the following columns:
        - "moving_path": Path to the moving image.
        - "reference_path": Path to the reference image.
        - "down_scale": int, downscaling factor used for speed optimization.
        - "moving_ROI": ROI NamedTuple (to be updated).
        - "reference_ROI": ROI NamedTuple (to be updated).
        - "current_shift": np.ndarray (to be updated).

    Returns:
    -------
    pd.DataFrame
        The input DataFrame updated with:
        - "moving_ROI": ROI NamedTuple for the moving image.
        - "reference_ROI": ROI NamedTuple for the reference image.
        - "current_shift": np.ndarray (2D) representing translation shift.

    Notes:
    ------
    - The first row typically aligns to itself, meaning `current_shift = [0, 0]`.
    - Each row's downscale factor is **retrieved from the DataFrame**, allowing flexibility per image.
    - This function can be parallelized in the future, as each row’s calculation is independent.
    """

    # ✅ Ensure input is a valid DataFrame with required columns
    required_cols = {"moving_path", "reference_path", "down_scale", "moving_ROI", "reference_ROI", "current_shift"}
    if not isinstance(alignment_df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")
    if not required_cols.issubset(alignment_df.columns):
        raise ValueError(f"Missing required columns. Expected: {required_cols}, Found: {set(alignment_df.columns)}")

    # ✅ Ensure correct data type for 'current_shift'
    alignment_df["current_shift"] = alignment_df["current_shift"].astype(object)

    # TODO: Implement parallel processing (each calculation is independent)
    for index, row in alignment_df.iterrows():
        # ✅ Retrieve downscale factor from DataFrame
        down_scale = row["down_scale"]
        if not isinstance(down_scale, (int, float)) or down_scale <= 0:
            raise ValueError(f"Invalid 'down_scale' value at index {index}: {down_scale}. Must be a positive number.")

        # Load moving image
        mov_path = row["moving_path"]
        mov_img = tiff.imread(mov_path)

        # Load reference image
        ref_path = row["reference_path"]
        ref_img = tiff.imread(ref_path)

        # ✅ Downscale for performance optimization
        mov_img_ds = downscale_local_mean(mov_img, (down_scale, down_scale)).astype(mov_img.dtype)
        ref_img_ds = downscale_local_mean(ref_img, (down_scale, down_scale)).astype(ref_img.dtype)

        print(f"Processing alignment: ref -> {ref_path.name}, moving -> {mov_path.name}")

        # ✅ Compute translation shift
        ref_ROI, mov_ROI, current_shift = get_translation(ref_img_ds, mov_img_ds)
    

        # ✅ Update the DataFrame correctly
        alignment_df.loc[index, "moving_ROI"] = mov_ROI
        alignment_df.loc[index, "reference_ROI"] = ref_ROI
        print(f"current shift: {current_shift}")
        alignment_df.at[index, "current_shift"] = current_shift

    return alignment_df

def calculate_cumulative_shifts(input_df):
    """
    Computes the cumulative shift values for a DataFrame containing pairwise shifts.

    This function sequentially calculates the cumulative translation shifts across multiple images
    based on their pairwise shifts. It also updates the corresponding cumulative Region of Interest (ROI)
    for each image in the DataFrame.

    Parameters:
    ----------
    input_df : pd.DataFrame
        A Pandas DataFrame containing at least the following columns:
        - "current_shift" : np.ndarray (shape: (2,)) representing row and column shifts.
        - "moving_ROI" : ROI (NamedTuple) defining the region of interest.

    Returns:
    -------
    pd.DataFrame
        The input DataFrame with updated columns:
        - "cumulative_shift" : np.ndarray (shape: (2,)) cumulative shift applied.
        - "cumulative_ROI" : ROI (NamedTuple) updated based on cumulative shift.

    Notes:
    ------
    - The first row (index 0) has a cumulative shift of [0, 0] since it's the reference.
    - Each row's cumulative shift is calculated as the sum of the previous cumulative shift and
      the current pairwise shift.
    """

    # ✅ Ensure input is a DataFrame with the required columns
    required_cols = {"current_shift", "moving_ROI"}
    if not isinstance(input_df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")
    if not required_cols.issubset(input_df.columns):
        raise ValueError(f"Missing required columns. Expected: {required_cols}, Found: {set(input_df.columns)}")

    # ✅ Ensure 'cumulative_shift' column exists and is set to object dtype
    if "cumulative_shift" not in input_df.columns:
        input_df["cumulative_shift"] = None
    input_df["cumulative_shift"] = input_df["cumulative_shift"].astype(object)

    # ✅ Ensure 'cumulative_ROI' column exists
    if "cumulative_ROI" not in input_df.columns:
        input_df["cumulative_ROI"] = None

    # ✅ Process each row sequentially
    for index, row in input_df.iterrows():
        if index == 0:
            cumulative_shift = np.array([0., 0.])  # Reference image has no shift
        else:
            current_shift = row["current_shift"]
            if not isinstance(current_shift, np.ndarray) or current_shift.shape != (2,):
                raise ValueError(f"Invalid 'current_shift' at index {index}. Expected np.ndarray of shape (2,), got {type(current_shift)}.")
            cumulative_shift = input_df.loc[index - 1, "cumulative_shift"] + current_shift  # Accumulate shifts

        input_df.at[index, "cumulative_shift"] = cumulative_shift  # ✅ Store cumulative shift

        # ✅ Compute and store cumulative ROI
        moving_ROI = row["moving_ROI"]
        if not isinstance(moving_ROI, ROI):
            raise ValueError(f"Invalid 'moving_ROI' at index {index}. Expected ROI instance, got {type(moving_ROI)}.")
        
        cumulative_ROI_x0 = int(moving_ROI.x0 + cumulative_shift[1])  # Shift in x-direction
        cumulative_ROI_x1 = int(moving_ROI.x1 + cumulative_shift[1])
        cumulative_ROI_y0 = int(moving_ROI.y0 + cumulative_shift[0])  # Shift in y-direction
        cumulative_ROI_y1 = int(moving_ROI.y1 + cumulative_shift[0])
        cumulative_ROI = ROI(x0=cumulative_ROI_x0, x1=cumulative_ROI_x1, y0=cumulative_ROI_y0, y1=cumulative_ROI_y1)

        input_df.loc[index, "cumulative_ROI"] = cumulative_ROI  # ✅ Store cumulative ROI
    
    return input_df

