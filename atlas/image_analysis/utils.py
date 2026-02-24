"""
Utility functions for image analysis tasks.
"""

import numpy as np

def image_dtype_min_max(img):
    """
    Determines the minimum and maximum possible values for the given image's data type.

    Parameters:
    ----------
    img : np.ndarray
        Input image array.
    
    Returns:
    -------
    tuple (int or float, int or float)
        - The minimum and maximum possible values for the image dtype.
    """
    assert isinstance(img, np.ndarray), "img must be a numpy array"
    dtype_info = np.iinfo(img.dtype) if np.issubdtype(img.dtype, np.integer) else np.finfo(img.dtype)
    return dtype_info.min, dtype_info.max

def mask_low_and_saturation(img, min_factor = 1.0, max_factor = 1.0):
    """
    Creates a mask highlighting saturated pixels (min/max values) in the image.

    Parameters:
    ----------
    img : np.ndarray
        Input image array.
    
    Returns:
    -------
    np.ndarray
        Boolean mask where True represents saturated pixels.
    """
    assert isinstance(img, np.ndarray), "img must be a numpy array"
    min_val, max_val = image_dtype_min_max(img)
    return (img <= min_val*min_factor) | (img >= max_val*max_factor)

def rescale_image_intensity(input_image, percentile_low=1, percentile_high=99):
    """
    Rescales the intensity of an image based on given percentiles while maintaining its dtype.

    Parameters:
    ----------
    input_image : np.ndarray
        The input image array.
    percentile_low : float, optional
        Lower percentile for contrast stretching (default: 1).
    percentile_high : float, optional
        Upper percentile for contrast stretching (default: 99).

    Returns:
    -------
    np.ndarray
        The rescaled image with the same dtype as the input.
    float
        The lower percentile intensity value.
    float
        The upper percentile intensity value.
    """
    assert isinstance(input_image, np.ndarray), "input_image must be a numpy array"
    assert 0 <= percentile_low < percentile_high <= 100, "Percentiles must be in range [0, 100] and percentile_low < percentile_high"
    
    # Create saturation mask
    saturation_mask = mask_low_and_saturation(input_image)
    
    # Compute percentile values, ignoring saturated pixels
    p_low, p_high = np.percentile(input_image[np.logical_not(saturation_mask)], 
                                    (percentile_low, percentile_high))
    
    # Get min/max possible values for the image dtype
    img_min_val, img_max_val = image_dtype_min_max(input_image)
    
    # Copy the input image to avoid modifying the original
    img_rescale = input_image.astype(np.float32)  # Convert to float for computation
    
    # Apply rescaling transformation
    img_rescale = ((img_rescale - p_low) / (p_high - p_low)) * img_max_val
    
    # Clip values to ensure they stay within the valid dtype range
    img_rescale = np.clip(img_rescale, img_min_val, img_max_val)
    
    # Convert back to original dtype
    return img_rescale.astype(input_image.dtype), p_low, p_high