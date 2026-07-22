"""
Functions for handling Fibics metadata.
"""

import xmltodict
import tifffile as tiff
from pathlib import Path, PureWindowsPath, PurePosixPath, PurePath
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
    

def find_slice_valid_path(raw_data_folder):

    # find path to the ve-mif file which stores the slice's metadata
    mif_found = False
    for file in raw_data_folder.iterdir():  
        # Check if it's a file with the desired extension
        if file.is_file() and file.suffix == ".ve-mif":
            mif_file = file
            mif_found = True
            break
    
    if not mif_found:
        return None, ".ve-mif file not found"

    # get number of tif files in the data folder
    tif_files_list = list(raw_data_folder.glob("*.tif"))
    for i, tf in enumerate(tif_files_list):
        tif_files_list[i] = filename_helper(str(tf))
    
    num_tif_files = len(tif_files_list)

    # get a dict of the metadata
    with open(mif_file, "r", encoding="utf-8") as f:
        mif_dict = xmltodict.parse(f.read())

    # reject a data folder if the status is not completed
    if mif_dict['MosaicInfo']['Status'].lower() != "completed":
        return None, "Incomplete acquisition"
    
    # check if the tiff files for all the tiles exist in the folder
    # first check if the numbers match
    tile_list = mif_dict['MosaicInfo']['Tiles']['Tile']
    if len(tile_list) != num_tif_files:
        return None, "unequal number of tif files and metadata tif files"
    
    # check whether the filenames in the list exist in the folder
    for tile in tile_list:
        filename = filename_helper(tile["Filename"])
        if filename not in tif_files_list:
            return None, "tif files and metadata tif files do not agree"
    
    # check whether the pixel sizes are equal for all the tiles
    # no need as there is just a single pixel size in the metadata

    return raw_data_folder, "" # maybe return None instead of empty string

def get_valid_slice_folders(series_folder):

    series_list = []
    for folder in series_folder.iterdir():  # Iterate over all items in the folder
        if folder.is_dir() and folder.name.startswith("S_"):
            series_list.append(folder)

    valid_folders = []
    failed_folders = []
    for raw_data_folder in series_list:
        folder_name, failure_condition = find_slice_valid_path(raw_data_folder)
        if folder_name is not None:
            valid_folders.append(folder_name)
        else:
            failed_folders.append([folder_name, failure_condition])
    
    return valid_folders, failed_folders
    
def filename_helper(path_to_file: str | PurePath) -> str:
    if isinstance(path_to_file, PurePath):
        path_to_file = str(path_to_file)
    elif not isinstance(path_to_file, str):
        raise TypeError(
            "path_to_file must be a string or pathlib path object, "
            f"not {type(path_to_file).__name__}"
        )

    if "\\" in path_to_file:
        return PureWindowsPath(path_to_file).name

    return PurePosixPath(path_to_file).name