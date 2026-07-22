"""
Functions for handling Fibics metadata.
"""

import xmltodict
import tifffile as tiff
from pathlib import Path, PureWindowsPath, PurePosixPath
import numpy as np

def extract_tif_metadata(tif_path):
    """
    Extracts metadata from a `.tif` file and returns it as a dictionary.

    Parameters:
    ----------
    tif_path : Path
        Path to the `.tif` file.

    Returns:
    -------
    dict
        A dictionary containing the extracted metadata. If the 'FibicsXML' key is
        missing, an empty dictionary is returned instead.
    """

    # print(f"in extract_tif_metadata: {tif_path}")

    assert isinstance(tif_path, Path), "tif_path must be a pathlib.Path object"
    assert tif_path.is_file(), "tif_path must be an existing file"
    
    with tiff.TiffFile(tif_path) as tif:
        metadata = tif.pages[0].tags  # Extract metadata from the first page
        metadata_dict = {tag.name: tag.value for tag in metadata.values()}

    if "FibicsXML" not in metadata_dict:
        print("Warning: 'FibicsXML' metadata not found in the TIFF file.")
        metadata_dict['FibicsDict'] = {}

    try:
        fibics_dict = xmltodict.parse(metadata_dict['FibicsXML'])
        fibics_dict = fibics_dict.get('Fibics', {})
        metadata_dict['FibicsDict'] = fibics_dict
    except Exception as e:
        print(f"Error parsing FibicsXML: {e}")
        metadata_dict['FibicsDict'] = {}  
    
    return metadata_dict

def get_pixel_size_from_tif(filename, raw_data_folder):
    """
    Extracts the pixel size (in microns) from the TIFF metadata.

    Parameters:
    ----------
    filename : str
        The TIFF filename from the DataFrame.
    raw_data_folder : Path
        The directory where the TIFF files are stored.

    Returns:
    -------
    float
        The pixel size in microns, or NaN if extraction fails.
    """

    tif_name = filename_helper(filename)
    tif_file = raw_data_folder.joinpath(tif_name)

    # if "\\" in filename:
    #     tif_name = PureWindowsPath(filename).name
    #     tif_file = raw_data_folder.joinpath(tif_name)
    # else:
    #     tif_name = PurePosixPath(filename).name
    #     tif_file = raw_data_folder.joinpath(tif_name)

    try:
        tif_metadata = extract_tif_metadata(tif_file)
        fibics_dict = tif_metadata.get('FibicsDict', {})
        pix_size_micron = float(fibics_dict['Scan']['Ux'])
        print('')
        return pix_size_micron
    except (KeyError, ValueError, FileNotFoundError) as e:
        print(f"Warning: Could not extract pixel size for {filename}: {e}")
        print('')
        return float('nan')


def get_image_size_from_tif(filename, raw_data_folder):
    """
    Extracts ImageWidth and ImageHeight from the Fibics metadata in the TIFF file.

    Parameters:
    ----------
    filename : str
        The TIFF filename from the DataFrame.
    raw_data_folder : Path
        The directory where the TIFF files are stored.

    Returns:
    -------
    tuple (int, int)
        The image width and height extracted from the TIFF metadata.
        Returns (NaN, NaN) if extraction fails.
    """

    tif_name = filename_helper(filename)
    tif_file = raw_data_folder.joinpath(tif_name)
    
    # if "\\" in filename:
    #     tif_name = PureWindowsPath(filename).name
    #     tif_file = raw_data_folder.joinpath(tif_name)
    # else:
    #     tif_name = PurePosixPath(filename).name
    #     tif_file = raw_data_folder.joinpath(tif_name)

    try:
        # Extract metadata
        tif_metadata = extract_tif_metadata(tif_file)
        fibics_dict = tif_metadata.get('FibicsDict', {})

        # Extract image width & height if available
        image_width = int(fibics_dict['Image']['Width'])
        image_height = int(fibics_dict['Image']['Height'])

        return image_width, image_height

    except (KeyError, ValueError, FileNotFoundError) as e:
        print(f"Warning: Could not extract image size for {filename}: {e}")
        return np.nan, np.nan  # Return NaN values for missing metadata
    

def filename_helper(path_to_file):
    if "\\" in path_to_file:
        return PureWindowsPath(path_to_file).name
    else:
        return PurePosixPath(path_to_file).name