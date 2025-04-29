"""
Utility functions for IO tasks in the Atlas package.
"""

import re
import shutil
import zarr
from pathlib import Path
from pylibCZIrw import czi as pyczi
import numpy as np
import pandas as pd
import tifffile as tiff
from skimage.transform import downscale_local_mean

from atlas.image_analysis import rescale_image_intensity
from atlas.alignment import ROI

def apply_alignment(alignment_df, buffer_pixels=20, percentile_low=.5, percentile_high=99.9, use_down_sample=False):
    """
    Applies alignment transformations to a set of images and saves the result as a Zarr array.

    This function processes an alignment DataFrame that contains precomputed cumulative shifts
    and downscaling factors. It reconstructs an aligned image stack, optionally applying
    downsampling for visualization and intensity rescaling for contrast enhancement.
    A small buffer region is added around the images for safety.

    Parameters:
    ----------
    alignment_df : pd.DataFrame
        A Pandas DataFrame containing at least the following columns:
        - "moving_path" : Path to the moving image.
        - "cumulative_ROI" : ROI NamedTuple defining the transformed region.
        - "down_scale" : int, downscaling factor per image.
    buffer_pixels : int, optional
        Number of extra pixels to add around the reconstructed image for safety (default: 20).
    percentile_low : float, optional
        Lower percentile for contrast stretching (default: 0.5).
    percentile_high : float, optional
        Upper percentile for contrast stretching (default: 99.9).
    use_down_sample : bool, optional
        If True, applies downsampling during processing for visualization (default: False).

    Returns:
    -------
    tuple
        - zarr.Array : The created Zarr array containing the aligned images.
        - Path : The path to the saved Zarr file.

    Notes:
    ------
    - If `use_down_sample` is True, images will be downscaled according to their respective factors in the DataFrame.
    - If `use_down_sample` is False, full-resolution images are processed.
    - The function ensures that all images fit within the same global coordinate system.
    - Intensity rescaling improves visualization and prevents artifacts due to saturated pixel values.
    """
    # Ensure input is a Pandas DataFrame
    if not isinstance(alignment_df, pd.DataFrame):
        raise TypeError("alignment_df must be a Pandas DataFrame.")

    # Required columns for processing
    required_cols = {"moving_path", "cumulative_ROI", "down_scale", "reference_path"}
    missing_cols = required_cols - set(alignment_df.columns)
    if missing_cols:
        raise ValueError(f"alignment_df is missing required columns: {missing_cols}")

    # Ensure 'moving_path' and 'reference_path' contain valid Paths
    if not all(isinstance(p, (Path, str)) for p in alignment_df["moving_path"]):
        raise TypeError("Column 'moving_path' must contain Path or string file paths.")
    if not all(isinstance(p, (Path, str)) for p in alignment_df["reference_path"]):
        raise TypeError("Column 'reference_path' must contain Path or string file paths.")

    # Ensure 'cumulative_ROI' contains valid ROI objects
    if not all(isinstance(roi, ROI) for roi in alignment_df["cumulative_ROI"] if roi is not None):
        raise TypeError("Column 'cumulative_ROI' must contain ROI objects or None.")

    # Ensure percentile values are valid
    if not (0 <= percentile_low < percentile_high <= 100):
        raise ValueError("percentile_low must be < percentile_high, both in range [0, 100].")


    if use_down_sample:
        print("we will downsample during saving of the zarr, this is good during testing for visual inspection")
    else:
        print("doing full scale alignment, it is recomended to check results before with downscale")

    n_z = alignment_df.shape[0]
    #assert n_z == z_align_df.shape[0], "images do not match stitch frame"

    x0_values = []
    x1_values = []
    y0_values = []
    y1_values = []

    for roi, ds in zip(alignment_df['cumulative_ROI'], alignment_df['down_scale']):
        if use_down_sample:
            ds = 1

        if roi is not None:
            x0_values.append(roi.x0 * ds)
            x1_values.append(roi.x1 * ds)
            y0_values.append(roi.y0 * ds)
            y1_values.append(roi.y1 * ds)
        

    x0_min = min(x0_values) - buffer_pixels
    x1_max = max(x1_values) + buffer_pixels
    y0_min = min(y0_values) - buffer_pixels
    y1_max = max(y1_values) + buffer_pixels

    """
    if x0_min < 0 or y0_min < 0:
        print("this could give me an error at the moment, TODO TEST!!!!!")
        print(f"xmin:{x0_min}, ymin{y0_min}")
    """

    y_shape = y1_max - y0_min
    x_shape = x1_max - x0_min
    print(f"shape of the zarr array to create: {y_shape, x_shape, n_z}")

    path_0 = alignment_df["reference_path"][0]
    # Open the TIFF file and read the dtype without loading image data
    with tiff.TiffFile(path_0) as tif:
        original_image_dtype = tif.pages[0].dtype  # Get dtype of the first image page

    z0_path = path_0.parent.joinpath(f"{path_0.parent.name}.zarr")

    print(f"creating zarr at: {z0_path}")

    if z0_path.exists():
        rm_tree(z0_path)

    store = zarr.DirectoryStore(z0_path)
    chunk_size = 1024
    #print(f'Chunk size: {chunk_size},{chunk_size},1')
    # I will keep the z, y, x arrangement
    z = zarr.creation.open_array(store=store, mode='a', shape=(y_shape, x_shape, n_z), chunks=(chunk_size,chunk_size,1), dtype=original_image_dtype)

    for idx in range(z.shape[2]):
        current_tif_path = alignment_df['moving_path'][idx]
        img_idx = tiff.imread(current_tif_path)

        if use_down_sample:
            down_scale = alignment_df['down_scale'][idx]
            img_idx = downscale_local_mean(img_idx, (down_scale, down_scale)).astype(img_idx.dtype)
            ds_factor = 1
        else:
            ds_factor = alignment_df['down_scale'][idx]
        

        # rescale image intensity
        img_rescale, p_low, p_high = rescale_image_intensity(img_idx, percentile_low=percentile_low, percentile_high=percentile_high)


        # Determine target and reference ROIs
        if idx == 0:
            #print(f"idx {idx}, special case")
            ds_target_ROI = alignment_df['moving_ROI'][idx]
        else:
            #print(f"idx: {idx}, do as expected")
            ds_target_ROI = alignment_df['cumulative_ROI'][idx]

        # Assign ds_img_ROI in both cases (makes logic consistent)
        ds_img_ROI = alignment_df['moving_ROI'][idx]


        # Target index calculation (applies to both cases)
        tx0 = ds_target_ROI.x0 * ds_factor - x0_min
        tx1 = ds_target_ROI.x1 * ds_factor - x0_min
        ty0 = ds_target_ROI.y0 * ds_factor - y0_min
        ty1 = ds_target_ROI.y1 * ds_factor - y0_min
        #print(ty0, ty1, tx0, tx1)

        # Reference index calculation (applies to both cases)
        ix0 = ds_img_ROI.x0 * ds_factor
        ix1 = ds_img_ROI.x1 * ds_factor
        iy0 = ds_img_ROI.y0 * ds_factor
        iy1 = ds_img_ROI.y1 * ds_factor
        #print(iy0, iy1, ix0, ix1)

        
        z[ty0:ty1,tx0:tx1, idx] = img_rescale[iy0:iy1,ix0:ix1]
        print(f"Done for idx {idx}")

    return z, z0_path

def zarr_array_to_czi(zarr_path, pixel_size, end_str):
    """
    Converts a Zarr array into a CZI file.

    This function reads a Zarr array, extracts its image data, and saves it
    as a `.czi` file while preserving pixel size information.

    Parameters:
    ----------
    zarr_path : Path
        Path to the Zarr array.
    pixel_size : dict
        Dictionary containing pixel size information with keys:
        - 'Value' : XY pixel size in microns.
        - 'Axial' : Z step size in microns.
        - 'Unit' : Must be "µm".
    end_str : str
        String to append to the output CZI filename.

    Returns:
    -------
    Path
        Path to the created CZI file.

    Raises:
    -------
    AssertionError
        If the pixel size unit is not microns.
    ValueError
        If the Zarr array has an unexpected shape.
    """
    assert isinstance(zarr_path, Path), "zarr_path must be a Path object"
    assert isinstance(pixel_size, dict), "pixel_size must be a dictionary"
    assert 'Value' in pixel_size and 'Axial' in pixel_size and 'Unit' in pixel_size, "pixel_size must contain 'Value', 'Axial', and 'Unit' keys"
    assert pixel_size['Unit'] == "µm", "Pixel size unit must be in microns (µm)"
    
    pix_x = pixel_size['Value']
    pix_y = pixel_size['Value']
    pix_z = pixel_size['Axial']
    
    zarr_array = zarr.open_array(zarr_path, mode="r")  # Open as a Zarr array
    
    if len(zarr_array.shape) < 3:
        raise ValueError(f"Expected a 3D Zarr array, got shape {zarr_array.shape}")
    
    newczi = zarr_path.parent.joinpath(f"{zarr_path.parent.name}{end_str}.czi")
    print(f"Saving CZI file to: {newczi}")
    
    with pyczi.create_czi(newczi, exist_ok=True) as czidoc_w:
        # Loop over Z-planes and channels
        for frame in range(zarr_array.shape[2]):
            print(f"Processing frame {frame}")
            tmp_plane = zarr_array[:, :, frame].squeeze()
            czidoc_w.write(data=tmp_plane[..., np.newaxis], plane={"Z": frame})
        
        # Write metadata
        czidoc_w.write_metadata(
            document_name=newczi.stem,
            channel_names={0: "White"},
            scale_x=float(pix_x) * 10 ** -6,
            scale_y=float(pix_y) * 10 ** -6,
            scale_z=float(pix_z) * 10 ** -6,
        )
    
    return newczi

def is_there_a_single_tif(folder_path):
    """
    Checks if there is exactly one .tif file in the given folder.

    Parameters:
    ----------
    folder_path : Path
        The folder to search for .tif files.

    Returns:
    -------
    tuple (bool, Path or None)
        - True and the .tif file if exactly one is found.
        - False and None if no .tif or multiple .tif files are found.
    """
    assert isinstance(folder_path, Path), "folder_path must be a pathlib.Path object"

    counter = 0
    tiffout = None

    # Iterate over files in the folder
    for file in folder_path.iterdir():
        if file.is_file() and file.suffix.lower() == ".tif":  # Case-insensitive check
            counter += 1
            tiffout = file

        # If more than one .tif is found, return early
        if counter > 1:
            print("Folder has multiple .tif files.")
            return False, None

    # Handle cases after iterating through the folder
    if counter == 1:
        return True, tiffout  # Return the single .tif file found
    else:
        print("Folder has no .tif files.")
        return False, None  # No .tif files found
    
def extract_s_number(filepath):
    """
    Extracts the S number from a filename using a specific pattern (_S_\d+_).
    
    Parameters:
    ----------
    filepath : Path
        The file path from which to extract the S number.
    
    Returns:
    -------
    int or None
        - The extracted S number if found.
        - None if the pattern does not exist in the filename.
    """
    assert isinstance(filepath, Path), "filepath must be a pathlib.Path object"
    assert filepath.is_file(), "filepath must be an existing file"
    
    pattern = re.compile(r"_S_(\d+)_")
    match = pattern.search(filepath.name)  # Only search in the filename
    
    return int(match.group(1)) if match else None  # Return None if no match found

def rm_tree(pth):
    """
    Recursively removes a directory and all its contents.

    Parameters:
    ----------
    pth : Path
        The directory to be removed.
    """
    assert isinstance(pth, Path), "pth must be a pathlib.Path object"
    assert pth.is_dir(), "pth must be an existing directory"
    
    for child in pth.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()

def create_empty_folder(folder_path):
    """
    Creates an empty folder. If the folder exists, it clears all its contents.

    Parameters:
    ----------
    folder_path : Path
        Path to the folder that needs to be created or emptied.
    """
    assert isinstance(folder_path, Path), "folder_path must be a pathlib.Path object"
    
    if folder_path.exists():
        rm_tree(folder_path)  # Use rm_tree to remove contents
    folder_path.mkdir(parents=True, exist_ok=True)
    print(f"Created new (or emptied) folder: {folder_path}")