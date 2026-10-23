"""
Utility functions for IO tasks in the Atlas package.
"""

import re
import shutil
import zarr
from pathlib import Path, PurePath
from pylibCZIrw import czi as pyczi
import numpy as np
import pandas as pd
import tifffile as tiff
from skimage.transform import downscale_local_mean
import json
from atlas.image_analysis import rescale_image_intensity
from atlas.alignment import ROI

c

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
    assert isinstance(zarr_path, PurePath), "zarr_path must be a Path object"
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
    assert isinstance(folder_path, PurePath), "folder_path must be a pathlib.Path object"

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
    
# def extract_s_number(filepath):
#     """
#     Extracts the S number from a filename using a specific pattern (_S_\d+_).
    
#     Parameters:
#     ----------
#     filepath : Path
#         The file path from which to extract the S number.
    
#     Returns:
#     -------
#     int or None
#         - The extracted S number if found.
#         - None if the pattern does not exist in the filename.
#     """
#     assert isinstance(filepath, PurePath), "filepath must be a pathlib.Path object"
#     assert filepath.is_file(), "filepath must be an existing file"
    
#     pattern = re.compile(r"_S_(\d+)")
#     match = pattern.search(filepath.name)  # Only search in the filename
    
#     return int(match.group(1)) if match else None  # Return None if no match found

def extract_s_number(filepath):
    """
    Extract the S number from a file or directory name.

    Parameters
    ----------
    filepath : str or Path
        Path or name containing an S number.

    Returns
    -------
    int or None
        The extracted S number, or None if no match is found.
    """
    _filepath = Path(filepath)

    pattern = re.compile(r"(?:^|_)S_(\d+)")
    match = pattern.search(_filepath.name)

    return int(match.group(1)) if match else None

def reorder_files_by_s_number(file_list, new_order):
    """
    Reorders a list of file paths based on the S-number in the filename,
    using a user-defined new order.

    Parameters:
    ----------
    file_list : list of Path
        List of pathlib.Path objects with filenames containing '_S_<number>.tiff'
    new_order : list of int
        Desired order of the S-numbers.

    Returns:
    -------
    list of Path
        Reordered list of Path objects matching the specified new_order.
    """
    s_number_pattern = re.compile(r"_S_(\d+)\.tiff$", re.IGNORECASE)

    # Build mapping: S-number -> file
    s_to_file = {}
    for f in file_list:
        match = s_number_pattern.search(f.name)
        if match:
            s_to_file[int(match.group(1))] = f
        else:
            raise ValueError(f"Filename does not match expected pattern: {f.name}")

    # Reorder using the provided order
    reordered_files = []
    for s in new_order:
        if s not in s_to_file:
            raise ValueError(f"S-number {s} not found in file list")
        reordered_files.append(s_to_file[s])

    return reordered_files

def parse_shorthand_order(filepath):
    """
    Parses a shorthand order file using '...' to indicate a range.

    Example line: 1,...,5,7,9,8,6 → [1, 2, 3, 4, 5, 7, 9, 8, 6]

    Parameters:
    ----------
    filepath : str or Path
        Path to the .txt file containing the shorthand order.

    Returns:
    -------
    list of int
        Expanded list of integers following the shorthand notation.
    """
    from pathlib import Path

    text = Path(filepath).read_text().strip()
    tokens = [t.strip() for t in text.split(',') if t.strip()]
    
    result = []
    i = 0
    while i < len(tokens):
        if tokens[i] == '...':
            raise ValueError("Order string cannot start with '...'")
        elif i + 2 < len(tokens) and tokens[i + 1] == '...':
            start = int(tokens[i])
            end = int(tokens[i + 2])
            step = 1 if start <= end else -1
            result.extend(range(start, end + step, step))
            i += 3  # Skip 'start', '...', 'end'
        else:
            result.append(int(tokens[i]))
            i += 1
    return result

def rm_tree(pth):
    """
    Recursively removes a directory and all its contents.

    Parameters:
    ----------
    pth : Path
        The directory to be removed.
    """
    assert isinstance(pth, PurePath), "pth must be a pathlib.Path object"
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
    assert isinstance(folder_path, PurePath), "folder_path must be a pathlib.Path object"
    
    if folder_path.exists():
        rm_tree(folder_path)  # Use rm_tree to remove contents
    folder_path.mkdir(parents=True, exist_ok=True)
    print(f"Created new (or emptied) folder: {folder_path}")

