"""
Functions for handling Fibics metadata.
"""

import xmltodict
import tifffile as tiff
from pathlib import Path, PureWindowsPath, PurePosixPath, PurePath
import numpy as np
import warnings


def get_stitched_pixel_size_microns(tif_path):
    """Read the stitched TIFF's acquisition pixel size in microns."""
    with tiff.TiffFile(tif_path) as tif:
        try:
            metadata = tif.shaped_metadata[0]["AcquisitionMetadata"]
            value = float(metadata["Pixel size"])
            unit = metadata["Pixel size unit"].strip().lower()
        except (TypeError, IndexError, KeyError, AttributeError, ValueError) as exc:
            raise ValueError(f"Missing or invalid pixel-size metadata in {tif_path}") from exc

    factors = {"nm": 0.001, "µm": 1.0, "μm": 1.0, "um": 1.0, "mm": 1000.0, "m": 1_000_000.0}
    if unit not in factors:
        raise ValueError(f"Unsupported pixel-size unit {unit!r} in {tif_path}")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Pixel size must be positive and finite in {tif_path}")
    return value * factors[unit]

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

    assert isinstance(tif_path, PurePath), "tif_path must be a pathlib.Path object"
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


def get_fibics_tif_scan_pixel_size(tif_path):
    """Return scan vectors and their magnitudes from a raw FIBICS TIFF.

    FIBICS Ux/Uy and Vx/Vy contain no explicit units in the examples examined;
    this uses the existing stitching code's assumption that they are microns.
    Vector components retain their signs. X and Y sizes are vector magnitudes.
    """
    tif_path = Path(tif_path)
    try:
        scan = extract_tif_metadata(tif_path)["FibicsDict"]["Scan"]
        ux, uy, vx, vy = (float(scan[key]) for key in ("Ux", "Uy", "Vx", "Vy"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid FIBICS scan vectors in {tif_path}") from exc
    if not np.isfinite([ux, uy, vx, vy]).all():
        raise ValueError(f"Non-finite FIBICS scan vectors in {tif_path}")
    return {
        "u_vector": [ux, uy], "v_vector": [vx, vy],
        "pixel_size_x": float(np.hypot(ux, uy)),
        "pixel_size_y": float(np.hypot(vx, vy)),
        "unit": "µm", "unit_source": "assumed by existing stitching code",
    }


def get_fibics_mif_pixel_size(mif_path):
    """Return the scalar pixel size and its unit as recorded in a .ve-mif."""
    mif_path = Path(mif_path)
    with mif_path.open(encoding="utf-8") as f:
        metadata = xmltodict.parse(f.read())
    try:
        pixel = metadata["MosaicInfo"]["PixelSize"]
        value, unit = float(pixel["#text"]), pixel["@unit"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid MIF pixel size in {mif_path}") from exc
    if not np.isfinite(value) or value <= 0 or not isinstance(unit, str) or not unit.strip():
        raise ValueError(f"Invalid MIF pixel size or unit in {mif_path}")
    return {"pixel_size": value, "unit": unit}

def filename_helper(path_to_file: str | PurePath) -> PurePath:

    """
    Gets a filepath which is valid for Windows and Linux

    Parameters:
    ----------
    filename : str
        The TIFF filename from the DataFrame.
    raw_data_folder : Path
        The directory where the TIFF files are stored.

    Returns:
    -------
    path_to_file converted to PurePosixPath | PureWindowsPath
        The file path in either format.
    """
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

    """
    Finds whether or not a slice folder is valid through a series of metadata checks

    Parameters:
    ----------
    raw_data_folder : Path()
                    The path to a single slice folder.

    Returns:
    -------
    valid_folder : Path()
                    this is raw_data_folder if all checks are passed, else None
    failure_condition : str
                    a string describing the check that failed, if success then an empty string
    """

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

    # TODO: Consider validating per-tile "Tile Complete" events in MosaicInfo/Log
    # and TIFF readability. Currently only overall completion, tile counts,
    # and filename presence are checked.
    
    # Count acquired tiles, since some positions in the grid may be omitted.
    tile_list = mif_dict['MosaicInfo']['Tiles']['Tile']
    # xmltodict returns a dictionary for a single tile.
    if isinstance(tile_list, dict):
        tile_list = [tile_list]
    ntiles = len(tile_list)
    if ntiles != num_tif_files:
        return None, "unequal number of actual tif files and tif files listed in .ve-mif metadata"
    
    # check whether the filenames in the list exist in the folder
    for tile in tile_list:
        filename = filename_helper(tile["Filename"])
        if filename not in tif_files_list:
            return None, "tif filenames and metadata tif files do not agree"
    
    # check whether the pixel sizes are equal for all the tiles
    # no need as there is just a single pixel size in the metadata

    return raw_data_folder, "" # maybe return None instead of empty string

def get_valid_slice_folders(series_folder: Path) -> tuple[list[Path], list[list[Path | str]]]:


    """
    Runs find_slice_valid path() for all the slice folders to check the validity through metadata

    Parameters:
    ----------
    series_folder : Path()
                    The path to the dataset's root i.e. the dir which contains folders for all the slices.

    Returns:
    -------
    valid_folders : list[str]
                    a list of slice folders which pass the metadata checks
    failed_folders : list[list[Path | str]]
                    a list of lists where each entries first value is the failed path and the second value
                    is the string for why it failed
    """

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

def get_imaging_duration(mif_dict: dict):

    """
    Gets imaging time from the start and end of acquisition in the mif metadata (which is already in a dict)
    """

    from datetime import datetime

    start = datetime.fromisoformat(mif_dict['MosaicInfo']['Date'])
    end = datetime.fromisoformat(mif_dict['MosaicInfo']['DateFinished'])

    duration = end - start

    total_seconds = int(duration.total_seconds())

    days, remainder = divmod(total_seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds = divmod(remainder, 60)

    formatted_duration = f"D{days}T{hours:02d}:{minutes:02d}:{seconds:02d}"

    return formatted_duration
    
def get_metadata_for_stitched_tif(mif_file: PurePath):

    """
    Gets acquisition metadata for the stitched tif for a single slice

    Parameters:
    ----------
    mif_file : PurePath
                The path to an existinf .ve-mif file.

    Returns:
    -------
    metadict : dict
                A dictionary with acquisition metadata to be included with the stitched tif
    """

    import re

    with open(mif_file, "r", encoding="utf-8") as f:
        mif_dict = xmltodict.parse(f.read())

    import re
    metadict = {}

    metadict['Acquisition date'], _ = mif_dict['MosaicInfo']['Date'].split('T')
    metadict['Protocol name'] = mif_dict['MosaicInfo']['SetupInfo']['SetupFilename']

    numtiles_x = mif_dict['MosaicInfo']['TileInfo']['NumTilesX']
    numtiles_y = mif_dict['MosaicInfo']['TileInfo']['NumTilesY']
    metadict['Tile arrangement'] = f"R{numtiles_y}C{numtiles_x}"

    metadict['Imaging duration'] = get_imaging_duration(mif_dict)

    metadict["Pixel size"] = float(mif_dict['MosaicInfo']['PixelSize']['#text'])
    metadict['Pixel size unit'] = mif_dict['MosaicInfo']['PixelSize']['@unit']
    metadict["Pixel size MIF"] = metadict["Pixel size"]
    metadict["Pixel size MIF unit"] = metadict["Pixel size unit"]

    # Keep each tile's scan vectors so differences between tiles remain visible.
    tiles = mif_dict['MosaicInfo']['Tiles']['Tile']
    if isinstance(tiles, dict):
        tiles = [tiles]
    tile_calibrations = []
    for tile in tiles:
        filename = tile.get("Filename")
        entry = {"filename": filename_helper(filename) if filename else None}
        try:
            if not filename:
                raise ValueError("Tile filename is missing from MIF metadata")
            entry.update(get_fibics_tif_scan_pixel_size(Path(mif_file).parent / entry["filename"]))
        except (AssertionError, OSError, ValueError) as exc:
            entry["error"] = str(exc)
            warnings.warn(f"Could not read tile pixel size for {filename}: {exc}", stacklevel=2)
        tile_calibrations.append(entry)
    metadict["Pixel size tile TIFF per tile"] = tile_calibrations
    # The first listed tile is the representative value; all tile values
    # remain available above. Missing metadata must not prevent stitching.
    representative = tile_calibrations[0] if tile_calibrations else {}
    for axis in ("X", "Y"):
        metadict[f"Pixel size tile TIFF {axis}"] = representative.get(f"pixel_size_{axis.lower()}")
        metadict[f"Pixel size tile TIFF {axis} unit"] = representative.get("unit")

    dwell_unit = mif_dict['MosaicInfo']['DwellTime']['@unit']
    if dwell_unit.lower() == 'ns':
        dwellscale = 0.001
    elif dwell_unit.lower() == "µs":
        dwellscale = 1
    elif dwell_unit.lower() == "ms":
        dwellscale = 1000
    metadict["Dwell time"] = dwellscale * float(mif_dict['MosaicInfo']['DwellTime']['#text'])
    metadict['Dwell time unit'] = "µs"

    metadict['Line average'] = int(mif_dict['MosaicInfo']['LineAveraging'])

    nrows = float(mif_dict['MosaicInfo']['TileInfo']['NumTilesY'])
    ncols = float(mif_dict['MosaicInfo']['TileInfo']['NumTilesX'])
    ntiles = nrows * ncols
    
    if ntiles != 1:
        wd = 0
        brightness = 0
        contrast = 0
        for tile in mif_dict['MosaicInfo']['Tiles']['Tile']:
            wd += float(tile['WD'])
            brightness += float(tile['Brightness'])
            contrast += float(tile['Contrast'])
        wd /= ntiles
        contrast /= ntiles
        brightness /= ntiles

    else:
        wd = float(mif_dict['MosaicInfo']['Tiles']['Tile']['WD'])
        brightness = float(mif_dict['MosaicInfo']['Tiles']['Tile']['Brightness'])
        contrast = float(mif_dict['MosaicInfo']['Tiles']['Tile']['Contrast'])

    metadict['Working distance'] = wd * 1000 # WD is unitless, we assume it is stored in meters
    metadict['Working distance unit'] = 'mm'

    metadict['Brightness'] = brightness

    metadict['Contrast'] = contrast

    metadict['Working distance'] = wd * 1000 # WD is unitless, we assume it is stored in meters
    metadict['Working distance unit'] = 'mm'

    text = mif_dict['MosaicInfo']['SetupInfo']['Beam']['Aperture']
    units = re.findall(r"(?<=\d)\s*([a-zA-Z]+)", text)
    metadict['EHT voltage'] = float(mif_dict['MosaicInfo']['SetupInfo']['Beam']['AccV']) / 1000
    metadict['EHT voltage unit'] = units[0]
    metadict['EHT current'] = float(mif_dict['MosaicInfo']['SetupInfo']['Beam']['BeamI'])
    metadict['EHT current unit'] = units[1]

    metadict['Detector'] = mif_dict['MosaicInfo']['SetupInfo']['Signal']['Detector']

    bsd_gain = None
    detector_items = mif_dict['MosaicInfo']['SetupInfo']['Signal']['DetectorInfo']['item']
    # xmltodict returns a dict for one item, and a list for repeated items.
    # SE2 acquisitions can have only a Collector setting and no BSD Gain.
    if isinstance(detector_items, dict):
        detector_items = [detector_items]
    for detector_item in detector_items:
        if detector_item['@name'].lower() == "bsd gain":
            bsd_gain = detector_item['#text']
            break

    metadict['BSD Gain'] = bsd_gain

    autofocusfreq = float(mif_dict['MosaicInfo']['TileInfo']['AutoTune']['AutoFocusFrequency'])
    autofocusbool = True
    if not autofocusfreq > 0:
        autofocusbool = False
        autofocusfreq = 0
    metadict['Autofocus'] = autofocusbool
    metadict['Autofocus frequency'] = autofocusfreq

    autostigfreq = float(mif_dict['MosaicInfo']['TileInfo']['AutoTune']['AutoStigFrequency'])
    autostigbool = True
    if not autostigfreq > 0:
        autostigbool = False
        autostigfreq = 0
    metadict['Autostig'] = autostigbool
    metadict['Autostig frequency'] = autostigfreq

    return metadict
