import numpy as np
import xmltodict
import pandas as pd

from collections import namedtuple
from shapely.geometry import box
from dateutil import parser

from atlas.io import get_pixel_size_from_tif, get_image_size_from_tif

def normalize_angle(angle):
    """
    Normalizes an angle to the range [-180, 180] degrees.

    Parameters:
    ----------
    angle : float
        The input angle in degrees.

    Returns:
    -------
    float
        The normalized angle within [-180, 180] range.
    """
    return ((angle + 180) % 360) - 180

def rotate_points(points, angle_degrees):
    """
    Rotates one or multiple 2D points (x, y) by a given angle in degrees.

    Parameters:
    ----------
    points : tuple (x, y) or list of tuples [(x1, y1), (x2, y2), ...]
        The original point(s) to be rotated.
    angle_degrees : float
        The rotation angle in degrees.

    Returns:
    -------
    tuple (x', y') or list of tuples [(x1', y1'), (x2', y2'), ...]
        The rotated point(s).
    """
    # Convert input to NumPy array
    points_array = np.array(points, dtype=np.float64)  # Ensure it's float for precision

    # If a single point was given, reshape it to (1, 2)
    if points_array.ndim == 1:
        points_array = points_array.reshape(1, 2)

    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # Define the rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians),  np.cos(angle_radians)]
    ])

    # Apply rotation (matrix multiplication)
    rotated_points = points_array @ rotation_matrix.T  # Transpose for correct multiplication

    # Convert back to original format (tuple or list of tuples)
    if len(rotated_points) == 1:
        return tuple(rotated_points[0])  # Return a single tuple for single input
    return [tuple(point) for point in rotated_points]  # Return a list of tuples for multiple points

# Define NamedTuple for Rectangle
Rectangle = namedtuple("Rectangle", ["top", "bot", "left", "right"])

# Function to create both Rectangle and Shapely box
def create_geometry(row):
    """
    Creates a Rectangle namedtuple and a Shapely box geometry from a row in the DataFrame.

    Parameters:
    ----------
    row : pandas.Series
        A row from the DataFrame.

    Returns:
    -------
    tuple(Rectangle, shapely.geometry.box)
        A tuple containing the namedtuple Rectangle and the Shapely box object.
    """
    rect = Rectangle(top=row['Y0_pix'], bot=row['Y1_pix'], left=row['X0_pix'], right=row['X1_pix'])
    geom = box(minx=rect.left, maxx=rect.right, miny=rect.top, maxy=rect.bot)
    return rect, geom

def get_overlap_relative(box_reference, box_moving):
    """
    Computes the overlapping region between two Shapely boxes and expresses 
    the overlap relative to each box's own coordinate system.

    Parameters:
    ----------
    box_reference : shapely.geometry.Polygon
        The reference box (e.g., from the fixed image).
    box_moving : shapely.geometry.Polygon
        The moving box (e.g., from the image being aligned).

    Returns:
    -------
    tuple (shapely.geometry.Polygon, shapely.geometry.Polygon)
        A tuple containing two Shapely box geometries:
        - The first box (`ref_overlap`) represents the overlap relative to the 
          `box_reference` coordinate system.
        - The second box (`mov_overlap`) represents the overlap relative to the 
          `box_moving` coordinate system.
    """
    # Compute the intersection
    overlap = box_reference.intersection(box_moving)

    # Convert the overlapping box to integer pixel indices
    min_x, min_y, max_x, max_y = map(int, overlap.bounds)

    # overlap is in final image space (boxes reference frame), 
    # now I have to change it to the pixel space of each tif.
    img0_x_0 = min_x - int(box_reference.bounds[0])
    img0_x_1 = max_x - int(box_reference.bounds[0])
    img0_y_0 = min_y - int(box_reference.bounds[1])
    img0_y_1 = max_y - int(box_reference.bounds[1])

    ref_overlap = box(minx=img0_x_0, miny=img0_y_0, maxx=img0_x_1, maxy=img0_y_1)
    print(f"overlap img0 x: {img0_x_0}-{img0_x_1}, y {img0_y_0}-{img0_y_1}")


    img1_x_0 = min_x - int(box_moving.bounds[0])
    img1_x_1 = max_x - int(box_moving.bounds[0])
    img1_y_0 = min_y - int(box_moving.bounds[1])
    img1_y_1 = max_y - int(box_moving.bounds[1])
    mov_overlap = box(minx=img1_x_0, miny=img1_y_0, maxx=img1_x_1, maxy=img1_y_1)
    print(f"overlap img1 x: {img1_x_0}-{img1_x_1}, y {img1_y_0}-{img1_y_1}")

    return ref_overlap, mov_overlap

def get_tiles_dataframe(mif_file, buffer_microns):
    """
    Loads a .mif (Mosaic Information File) and parses tile metadata into a DataFrame 
    suitable for image stitching and mosaicking.

    This function:
    - Parses the .mif file into a structured dictionary.
    - Extracts tile metadata (stage positions, acquisition times, filenames, etc.).
    - Corrects for scan rotation by rotating stage coordinates.
    - Re-centers the coordinate system to start at (0,0), with an additional buffer.
    - Converts stage positions to pixel units based on per-tile pixel size.
    - Precomputes bounding boxes and Shapely geometries for each tile.
    - Sorts tiles by acquisition time to simplify stitching later.

    Parameters:
    ----------
    mif_file : pathlib.Path
        Path to the .mif file containing tile metadata.
    
    buffer_microns : float
        Extra buffer (in microns) added around the stitched image to avoid boundary clipping 
        during stitching.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame where each row corresponds to a tile and includes:
        - Stage coordinates (original and rotated)
        - Pixel coordinates and dimensions
        - Image filenames
        - Scan rotation angle
        - Pixel size
        - Acquisition start time
        - Shapely geometries representing each tile for spatial operations
        - A copy of the geometry ('geometry_shifted') to store shifts applied during stitching

    Notes:
    -----
    - Rotation is necessary to correctly align exported tile images when scan rotation is applied.
    - Buffer ensures that all tiles fit into a pre-allocated canvas without risk of going out of bounds.
    - Geometry columns ('geometry', 'geometry_shifted') are used for overlap calculation and later stitching.
    - Pixel size and image dimensions are extracted from the image file metadata, but no pixel data is loaded.
    """


    raw_data_folder = mif_file.parent
    with open(mif_file, "r", encoding="utf-8") as f:
        mif_dict = xmltodict.parse(f.read())
    
    # Extract tile list (or single dictionary)
    mif_tile_list = mif_dict['MosaicInfo']['Tiles']['Tile']

    # Ensure mif_tile_list is always a list
    if isinstance(mif_tile_list, dict):  # If it's a single dictionary, convert to a list
        mif_tile_list = [mif_tile_list]

    # Convert to DataFrame
    mif_tile_df = pd.DataFrame(mif_tile_list)

        # Convert all timestamps in 'StartTime' column to datetime objects
    mif_tile_df['StartTime'] = mif_tile_df['StartTime'].apply(parser.isoparse)

    mif_tile_df['ScanRotationDeg'] = normalize_angle(float(mif_dict['MosaicInfo']['ReferenceInfo']['Beam']['ScanRot']))

    # get pixel size to each row in the DataFrame
    mif_tile_df['PixelSizeMicron'] = mif_tile_df['Filename'].apply(
        lambda fname: get_pixel_size_from_tif(fname, raw_data_folder))
    # loads image size based on tif metadata, we do not open the pixel info
    mif_tile_df[['ImageWidth', 'ImageHeight']] = mif_tile_df['Filename'].apply(
        lambda fname: get_image_size_from_tif(fname, raw_data_folder)
    ).to_list()
    # Convert columns to float
    mif_tile_df['StageX'] = pd.to_numeric(mif_tile_df['StageX'], errors='coerce')
    mif_tile_df['StageY'] = pd.to_numeric(mif_tile_df['StageY'], errors='coerce')

    # Apply rotation to each row, this is important to keep the square exports of FIBICs in the correct frame of ref
    mif_tile_df[['StageX_rot', 'StageY_rot']] = mif_tile_df.apply(
        lambda row: pd.Series(rotate_points((row['StageX'], row['StageY']), row['ScanRotationDeg'] * -1)),
        axis=1  # Apply function row-wise
    )

    # now we set the reference frame to 0,0 and not to the actual stage position
    min_stage_X = mif_tile_df['StageX_rot'].min()
    min_stage_Y = mif_tile_df['StageY_rot'].min()

    # buffer is needed for later stitching, if not some images might go out of the pre alocated canvas
    buffer_pixels = int(np.round(buffer_microns/mif_tile_df['PixelSizeMicron'][0]))
    buffer_microns = buffer_pixels * mif_tile_df['PixelSizeMicron'][0]

    mif_tile_df['X0_micron'] = mif_tile_df['StageX_rot']-min_stage_X + buffer_microns
    mif_tile_df['Y0_micron'] = mif_tile_df['StageY_rot']-min_stage_Y + buffer_microns
    mif_tile_df['X0_pix'] = np.round(mif_tile_df['X0_micron']/mif_tile_df['PixelSizeMicron']).astype('uint')
    mif_tile_df['X1_pix'] = mif_tile_df['X0_pix'] + mif_tile_df['ImageWidth']
    mif_tile_df['Y0_pix'] = np.round(mif_tile_df['Y0_micron']/mif_tile_df['PixelSizeMicron']).astype('uint')
    mif_tile_df['Y1_pix'] = mif_tile_df['Y0_pix'] + mif_tile_df['ImageHeight']

    # I sort by time due to my simple tiling strategy later on
    mif_tile_df.sort_values('StartTime', ascending=True, inplace=True)

    # Apply geometry function to all rows and create new columns, probably geometry is enough, check later
    mif_tile_df[['rectangle', 'geometry']] = mif_tile_df.apply(lambda row: pd.Series(create_geometry(row)), axis=1)
    # during stitching I will modify the geomtry to account for the iamge shifts
    mif_tile_df['geometry_shifted'] = mif_tile_df['geometry'].copy()

    return mif_tile_df