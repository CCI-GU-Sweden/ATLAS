"""
Utility functions for alignment tasks.
"""

import numpy as np
import pandas as pd
from typing import NamedTuple
import tifffile as tiff
from skimage.registration import phase_cross_correlation
from skimage.transform import downscale_local_mean, estimate_transform, SimilarityTransform

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
    mask_ref = np.logical_not(mask_low_and_saturation(reference_img, max_factor = 0.8))
    mov_mask = np.logical_not(mask_low_and_saturation(moving_img, max_factor = 0.8))

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

    # Find minimal common shape
    min_y = min(crop_ref_img.shape[0], crop_mov_img.shape[0])
    min_x = min(crop_ref_img.shape[1], crop_mov_img.shape[1])

    # Crop both to (min_y, min_x) from top-left (0,0)
    crop_ref_img  = crop_ref_img[:min_y, :min_x]
    crop_ref_mask = crop_ref_mask[:min_y, :min_x]
    crop_mov_img  = crop_mov_img[:min_y, :min_x]
    crop_mov_mask = crop_mov_mask[:min_y, :min_x]

    # Compute phase cross-correlation shift
    detected_shift, _, _ = phase_cross_correlation(crop_ref_img, crop_mov_img, 
                                                reference_mask=crop_ref_mask, moving_mask=crop_mov_mask, 
                                                upsample_factor=1,
                                                overlap_ratio=.7)

    detected_shift = np.round(detected_shift)
    print(f"Detected pixel offset based on crops (row, col): {detected_shift}")

    current_shift = detected_shift + crop_shift

    return ref_ROI, mov_ROI, current_shift

def get_translation_old(reference_img, moving_img):
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

def initialize_alignment_df_points(tif_list_sorted, down_scale=10):
    """
    Initialize a DataFrame for pairwise and cumulative image alignment based on matched points.

    This function creates and pre-populates a structured pandas DataFrame used
    to manage z-alignment between consecutively ordered TIFF images. Each row
    represents the alignment of one "moving" image against a "reference" image
    (typically the previous image in the sorted list).

    The DataFrame is designed to store matched feature points and the estimated
    geometric transforms (both full resolution and downsampled) required to
    register the image stack into a common reference frame.

    Parameters
    ----------
    tif_list_sorted : list of pathlib.Path
        Sorted list of file paths to the TIFF images to be aligned.
        The order defines the pairwise alignment sequence.
    down_scale : int, optional (default=10)
        Downsampling factor used during feature matching and transform
        estimation to accelerate computation.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row per image and the following columns:

        - "moving_path": pathlib.Path
            Path to the moving image at the current index.
        - "reference_path": pathlib.Path
            Path to the reference image (previous image in the list,
            except for index 0 where moving == reference).
        - "moving_shape": tuple or None
            Shape of the moving image (to be filled later).
        - "reference_shape": tuple or None
            Shape of the reference image (to be filled later).
        - "moving_points": list or np.ndarray
            Matched feature coordinates in the moving image.
        - "reference_points": list or np.ndarray
            Corresponding matched feature coordinates in the reference image.
        - "image_transform": object
            Transform mapping the moving image to the reference image
            (e.g., Euclidean or affine transform).
        - "cumulative_transform": object
            Transform mapping the moving image into the global reference
            frame (typically image 0).
        - "image_transform_ds": object
            Transform estimated on downsampled images.
        - "cumulative_transform_ds": object
            Cumulative transform estimated on downsampled images.
        - "down_scale": int
            Downsampling factor used for this alignment row.

    Notes
    -----
    - Row 0 represents the base reference frame: the moving and reference
      images are identical.
    - All transform-related fields are initialized as None and are expected
      to be filled during the alignment process.
    - The function assumes pairwise alignment between consecutive images
      in `tif_list_sorted`.
    - This structure supports later chaining of transforms to express
      all images in a common coordinate system.
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
        "moving_shape": pd.Series([None] * num_images, dtype="object"),
        "reference_shape": pd.Series([None] * num_images, dtype="object"),
        "moving_points": pd.Series([None] * num_images, dtype="object"),  # list of matching points in the moving image
        "reference_points": pd.Series([None] * num_images, dtype="object"), # list of matching points in the reference image
        "image_transform": pd.Series([None] * num_images, dtype="object"), # transformation object for moving to reference image
        "cumulative_transform": pd.Series([None] * num_images, dtype="object"), # transformation object for moving image to common reference frame
        "image_transform_ds": pd.Series([None] * num_images, dtype="object"), # transformation object for moving to reference image down sampled
        "cumulative_transform_ds": pd.Series([None] * num_images, dtype="object"), # transformation object for moving image to common reference frame down samples
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

def find_largest_centered_rectangle(mask):
    """
    Finds the largest rectangle centered at the image center
    that is fully contained in a binary mask.
    """
    h, w = mask.shape
    cy, cx = h // 2, w // 2

    max_up = max_down = max_left = max_right = 0

    # Vertical extent
    for i in range(cy, -1, -1):
        if mask[i, cx]:
            max_up += 1
        else:
            break
    for i in range(cy + 1, h):
        if mask[i, cx]:
            max_down += 1
        else:
            break

    # Horizontal extent
    for j in range(cx, -1, -1):
        if mask[cy, j]:
            max_left += 1
        else:
            break
    for j in range(cx + 1, w):
        if mask[cy, j]:
            max_right += 1
        else:
            break

    top = cy - max_up + 1
    bottom = cy + max_down
    left = cx - max_left + 1
    right = cx + max_right

    # Now shrink until the whole region is valid
    while True:
        region = mask[top:bottom, left:right]
        if region.shape[0] == 0 or region.shape[1] == 0:
            raise ValueError("No valid center-aligned rectangle found.")
        if np.all(region):
            break
        # shrink evenly
        if bottom - top > 1:
            top += 1
            bottom -= 1
        if right - left > 1:
            left += 1
            right -= 1

    return top, bottom, left, right

def crop_to_centered_valid_rectangle(img1, img2):
    """
    Crop two images to their largest common, valid center-aligned rectangle,
    and return the crops along with their top-left coordinates in the original images.

    Returns:
    --------
    crop1 : np.ndarray
        Cropped region from img1.
    crop2 : np.ndarray
        Cropped region from img2.
    origin1 : np.ndarray
        (y, x) position of the top-left corner of crop1 in img1.
    origin2 : np.ndarray
        (y, x) position of the top-left corner of crop2 in img2.
    """
    shape = np.minimum(img1.shape, img2.shape)
    y, x = shape

    # Compute valid data masks
    mask1 = np.logical_not(mask_low_and_saturation(img1)[:y, :x])
    mask2 = np.logical_not(mask_low_and_saturation(img2)[:y, :x])

    common_mask = mask1 & mask2

    # Find largest center-aligned rectangle in valid region
    top, bottom, left, right = find_largest_centered_rectangle(common_mask)

    # Compute cropping coordinates in original images
    center1 = np.array(img1.shape) // 2
    center2 = np.array(img2.shape) // 2

    origin1 = np.array([
        center1[0] - (y // 2 - top),
        center1[1] - (x // 2 - left)
    ])
    origin2 = np.array([
        center2[0] - (y // 2 - top),
        center2[1] - (x // 2 - left)
    ])

    # Crop the images
    crop1 = img1[
        origin1[0]:origin1[0] + (bottom - top),
        origin1[1]:origin1[1] + (right - left)
    ]
    crop2 = img2[
        origin2[0]:origin2[0] + (bottom - top),
        origin2[1]:origin2[1] + (right - left)
    ]

    return crop1, crop2, origin1, origin2

def split_into_regions_with_centers(ref_img: np.ndarray, mov_img: np.ndarray, rows: int = 2, cols: int = 2):
    """
    Splits two identically shaped 2D images into a grid of regions (e.g. 2x2, 3x4).
    Returns matching region pairs and their center coordinates in (y, x) order.

    Parameters:
    -----------
    ref_img : np.ndarray
        Reference image.
    mov_img : np.ndarray
        Moving image (must match shape and dtype).
    rows : int
        Number of vertical splits (rows).
    cols : int
        Number of horizontal splits (columns).

    Returns:
    --------
    List of tuples: (ref_region, mov_region, center_yx)
        ref_region : np.ndarray
        mov_region : np.ndarray
        center_yx : np.ndarray (y, x) in full image coordinates
    """
    if ref_img.shape != mov_img.shape:
        raise ValueError("Input images must have the same shape.")
    if ref_img.dtype != mov_img.dtype:
        raise ValueError("Input images must have the same dtype.")

    h, w = ref_img.shape
    region_height = h // rows
    region_width = w // cols

    regions = []

    for r in range(rows):
        for c in range(cols):
            y_start = r * region_height
            y_end = (r + 1) * region_height if r < rows - 1 else h

            x_start = c * region_width
            x_end = (c + 1) * region_width if c < cols - 1 else w

            sl_y = slice(y_start, y_end)
            sl_x = slice(x_start, x_end)

            ref_q = ref_img[sl_y, sl_x]
            mov_q = mov_img[sl_y, sl_x]

            center_y = (y_start + y_end - 1) / 2
            center_x = (x_start + x_end - 1) / 2
            center_yx = np.array([center_y, center_x])

            regions.append((ref_q, mov_q, center_yx))

    return regions

def estimate_translation_no_rotation(reference_img, moving_img, upsample_factor=10):
    """
    Estimate pure translational shift between two 2D images using phase cross-correlation.

    This function computes the relative translation required to align
    `moving_img` to `reference_img`, assuming no rotational or scaling
    differences between them.

    A 2D Hanning window is applied to both images prior to phase
    cross-correlation to reduce edge artifacts and improve robustness.

    Parameters
    ----------
    reference_img : np.ndarray
        Reference image (2D array).
    moving_img : np.ndarray
        Moving image (2D array) to be aligned to the reference image.
        Must have the same shape as `reference_img`.
    upsample_factor : int, optional (default=10)
        Subpixel precision factor passed to `phase_cross_correlation`.
        Higher values increase accuracy at the cost of computation time.

    Returns
    -------
    rotation_deg : float
        Always 0.0 (included for API compatibility with functions that
        estimate both rotation and translation).
    translation_shift : np.ndarray
        Array of shape (2,) containing the estimated shift in (row, col)
        order. The shift corresponds to the translation that should be
        applied to `moving_img` to align it with `reference_img`.

    Notes
    -----
    - The shift follows the convention of `skimage.registration.phase_cross_correlation`,
      meaning it is expressed in (row, col) coordinates.
    - The images are windowed using a separable Hanning window to
      suppress boundary discontinuities.
    - This function assumes no rotational or scale differences between
      the images. If rotation is present, a more general transform
      estimation method should be used.
    """
        
    assert reference_img.shape == moving_img.shape, "Images must be the same shape"
    #orig_dtype = moving_img.dtype

    # optional: reduce edge effects (often helps a lot)
    wy = np.hanning(reference_img.shape[0])
    wx = np.hanning(reference_img.shape[1])
    win = wy[:, None] * wx[None, :]
    refw = reference_img * win
    movw = moving_img * win

    translation_shift, _, _ = phase_cross_correlation(refw, movw)
    #print(f"shit: {translation_shift}")

    rotation_deg = 0.0 # to be compatible with other implementations

    # Return rotation and translation (both in original image coordinates)
    return rotation_deg, translation_shift

def get_matched_points_by_quadrant(ref_img, mov_img, rows=2, cols=2):
    """
    Compute corresponding point pairs between two images using local region-based translation.

    The function performs the following steps:

    1. Crops both images to their largest centered overlapping valid region.
    2. Splits the cropped images into a grid of regions defined by (rows × cols).
    3. Estimates a local translation for each region independently.
    4. Converts the regional center coordinates and matched points back to
       full image coordinates.

    Parameters
    ----------
    ref_img : np.ndarray
        Reference image (2D array).
    mov_img : np.ndarray
        Moving image to align (2D array).
    rows : int, optional (default=2)
        Number of region divisions along the vertical axis.
    cols : int, optional (default=2)
        Number of region divisions along the horizontal axis.

    Returns
    -------
    reference_points : list of np.ndarray
        List of (y, x) coordinates in the reference image (full image coordinates).
    matched_points : list of np.ndarray
        List of (y, x) coordinates in the moving image corresponding to
        the reference points (full image coordinates).

    Notes
    -----
    - Local alignment is currently based on translation only (no rotation).
    - Region centers are defined in the cropped image coordinate system
      and are shifted back to original image coordinates using the crop origins.
    - The quality of the matched points depends on the reliability of
      local translation estimation; outlier rejection may be required
      for robust global transform estimation.
    - Increasing (rows, cols) provides more local correspondences but may
      reduce stability if regions contain insufficient structure.
    """
    # Crop to valid overlapping region, and get origin shifts
    image_reference, image_moving, ref_origin, mov_origin = crop_to_centered_valid_rectangle(ref_img, mov_img)

    # Split into nxm regionsand estimate local alignment
    #quadrants = split_into_quadrants_with_centers(image_reference, image_moving)
    quadrants = split_into_regions_with_centers(image_reference, image_moving, rows=rows, cols=cols)

    reference_points = []
    matched_points = []

    for ref_q, mov_q, center in quadrants:
        # Estimate local rotation/translation
        #angle, shift = estimate_rotation_translation_fourier_no_rotation(ref_q, mov_q)
        angle, shift = estimate_translation_no_rotation(ref_q, mov_q)
        matched = center - shift 
        # activate during testing
        #TODO: some blocks are less accurate, I need a strategy to dismiss outliers
        print(f"the shift for quadrant was: {shift}")

        # Convert to full image coordinates
        reference_points.append(center + ref_origin)
        print(f"ref origin: {ref_origin}")
        matched_points.append(matched + mov_origin)

    return reference_points, matched_points

def estimate_alignment_transform(reference_points, moving_points, model='similarity'):
    """
    Estimate a geometric transform that aligns moving points to reference points.

    Parameters:
    -----------
    reference_points : list or np.ndarray of shape (N, 2)
        Target points (from the reference image), in (x, y).
    moving_points : list or np.ndarray of shape (N, 2)
        Source points (from the moving image), in (x, y).
    model : str
        Transform type. One of:
        - 'translation'
        - 'euclidean'
        - 'similarity'
        - 'affine'

    Returns:
    --------
    tform : skimage.transform._geometric.GeometricTransform
        Fitted transformation object.
    """
    reference_points = np.asarray(reference_points)
    moving_points = np.asarray(moving_points)

    if reference_points.shape != moving_points.shape:
        raise ValueError("Point lists must have the same shape")

    if model not in ('translation', 'euclidean', 'similarity', 'affine'):
        raise ValueError(f"Invalid model '{model}'. Must be one of: translation, euclidean, similarity, affine")
    
    if model == "translation":
        print("checking!!!!!!")
        # points must be (x, y)
        shift = reference_points.mean(axis=0) - moving_points.mean(axis=0)

        tform = SimilarityTransform(translation=shift[::-1])
        #model = "euclidean"
        #tform = TranslationTransform()
        #tform.estimate(src=moving_points[:, ::-1], dst=reference_points[:, ::-1])
    else:  
        # Estimate transform
        tform = estimate_transform(model, src=moving_points[:, ::-1], dst=reference_points[:, ::-1])

    return tform

def compute_cumulative_transforms(df, transform_col="image_transform", output_col="cumulative_transform"):
    """
    Compute cumulative transforms by composing pairwise transforms.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with one row per image and transform information.
    transform_col : str
        Name of the column containing the pairwise (relative) transforms.
    output_col : str
        Name of the column to store the resulting cumulative transforms.

    Returns:
    --------
    None (modifies the DataFrame in place)
    """
    if df.empty:
        return

    # Initialize first cumulative transform
    df.at[0, output_col] = df.at[0, transform_col]

    # Compute composition for subsequent rows
    for i in range(1, len(df)):
        prev_cum = df.at[i - 1, output_col]
        this_pair = df.at[i, transform_col]
        df.at[i, output_col] = prev_cum + this_pair  # Transform composition

    print(f"Cumulative transforms computed in column '{output_col}'.")

def print_similarity_parameters(tform):
    """
    Print translation, rotation (deg), and scale from a similarity transform.
    """
    # Extract matrix
    M = tform.params  # shape (3, 3)

    # Translation
    tx = M[0, 2]
    ty = M[1, 2]

    # Rotation + scale
    # Upper-left 2×2 matrix: [a, -b; b, a] where a = s·cosθ, b = s·sinθ
    a = M[0, 0]
    b = M[1, 0]
    scale = np.sqrt(a**2 + b**2)
    rotation_rad = np.arctan2(b, a)
    rotation_deg = np.degrees(rotation_rad)

    print("Estimated transform parameters:")
    print(f"  Translation: x = {tx:.2f}, y = {ty:.2f}")
    print(f"  Rotation: {rotation_deg:.2f} degrees")
    print(f"  Scale: {scale:.5f}")

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

def compute_canvas_from_df(z_align_df, transform_column="cumulative_transform"):
    """
    Compute a common canvas bounding box that can hold all transformed images
    based on their cumulative transforms.

    Parameters:
    -----------
    z_align_df : pd.DataFrame
        The alignment DataFrame containing image shapes and transforms.
    transform_column : str
        The name of the column with the transform objects (e.g., "cumulative_transform" or "cumulative_transform_ds").

    Returns:
    --------
    canvas_shape : tuple (h, w)
        The final canvas shape encompassing all images.
    offset : np.ndarray
        The (y, x) offset to apply to align the canvas to positive coordinates.
    """
    all_transformed_corners = []
    use_downscaled_shape = transform_column.endswith("ds")

    for idx, row in z_align_df.iterrows():
        shape = row["moving_shape"]
        tform = row[transform_column]
        down_scale = int(row["down_scale"])

        if tform is None or shape is None:
            continue

        h, w = shape
        if use_downscaled_shape:
            h = h // down_scale
            w = w // down_scale

        corners = np.array([
            [0, 0],
            [0, w - 1],
            [h - 1, 0],
            [h - 1, w - 1]
        ])

        # Transform (y, x) → (x, y) → apply → back to (y, x)
        # transformed = tform.inverse(corners[:, ::-1])[:, ::-1]
        transformed = tform.inverse(corners)
        print(f"Image {idx:02d} transformed corners (rounded):")
        print('\n'.join(['  ' + str(np.round(row).astype(int)) for row in transformed]))
        print_similarity_parameters(tform)

        all_transformed_corners.append(transformed)

    all_corners = np.vstack(all_transformed_corners)
    min_yx = np.floor(all_corners.min(axis=0)).astype(int)
    max_yx = np.ceil(all_corners.max(axis=0)).astype(int)

    canvas_shape = tuple((max_yx - min_yx).astype(int))
    offset = - min_yx

    print(f"\nFinal canvas shape (h, w): {canvas_shape}")
    print(f"Global offset to apply (y, x): {offset}")

    #TODO: check that it works after corrections, so far tested with simple translations

    return canvas_shape, offset
