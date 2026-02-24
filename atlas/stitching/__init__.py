import numpy as np
import xmltodict
import pandas as pd
from pathlib import Path
import tifffile as tiff

from collections import namedtuple
from shapely.geometry import box
from dateutil import parser

from atlas.io import get_pixel_size_from_tif, get_image_size_from_tif, extract_s_number
from atlas.image_analysis import mask_low_and_saturation
from atlas.alignment.utils import first_last_true

from skimage.registration import phase_cross_correlation

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

def add_tile_overlap_columns(df, geometry_column="geometry"):
    """
    Computes pairwise overlap between tiles based on spatial geometry and adds two new columns:
    - 'overlaps_bool': List of booleans per row indicating which other tiles it overlaps with.
    - 'overlap_percent': List of integers per row showing the percentage overlap with each tile.

    Notes:
    - A tile is not considered to overlap with itself (overlap is False and 0%).
    - The percentage is computed relative to the tile's own area (not the intersected tile's).

    Parameters:
        df (pd.DataFrame): A DataFrame containing a column with shapely geometries.
        geometry_column (str): Name of the column containing shapely geometry boxes. Default is "geometry".

    Returns:
        pd.DataFrame: The same DataFrame with two new columns added.
    """

    # --- Validations ---
    assert geometry_column in df.columns, f"'{geometry_column}' column not found in DataFrame."
    from shapely.geometry.base import BaseGeometry
    assert all(isinstance(g, BaseGeometry) for g in df[geometry_column]), \
        f"All entries in '{geometry_column}' must be shapely geometry objects."

    n = len(df)
    overlaps_bool = []
    overlaps_percent = []

    for current_idx in range(n):
        current_box = df[geometry_column][current_idx]
        bool_row = []
        percent_row = []
        for query_idx in range(n):
            if current_idx == query_idx:
                bool_row.append(False)
                percent_row.append(0)
                continue

            query_box = df[geometry_column][query_idx]
            if current_box.intersects(query_box):
                overlap_area = current_box.intersection(query_box).area
                overlap_percent = int(round((overlap_area / current_box.area) * 100))
                bool_row.append(True)
                percent_row.append(overlap_percent)
            else:
                bool_row.append(False)
                percent_row.append(0)

        overlaps_bool.append(bool_row)
        overlaps_percent.append(percent_row)

    df["overlaps_bool"] = overlaps_bool
    df["overlap_percent"] = overlaps_percent

    return df

def get_total_canvas_size(tile_df):
    # based on the tiles dataframe returns the total size of the canvas needed for stitching
    buffer_pixels = tile_df['X0_pix'].min()
    print(f"buffer in pixels based on DF output: {buffer_pixels}")

    # here we generate the full canvas size
    max_X_row = tile_df.loc[tile_df["X0_pix"].idxmax()]
    max_Y_row = tile_df.loc[tile_df["Y0_pix"].idxmax()]
    total_img_width = max_X_row['X0_pix'] + max_X_row['ImageWidth'] + buffer_pixels
    total_img_height = max_Y_row['Y0_pix'] + max_Y_row['ImageHeight'] + buffer_pixels

    print(f"Total image will be of size including buffer: {total_img_width}x{total_img_height}")

    return int(total_img_width), int(total_img_height)

def stitch_ATLAS_tiles(tiles_df, raw_data_folder, max_shift_pixels=100):
    # Step 1: Initialize the full canvas and add the first image (reference)
    first_tif_path = raw_data_folder.joinpath(Path(tiles_df.iloc[0]['Filename']).name)
    with tiff.TiffFile(first_tif_path) as tif:
        image_dtype = tif.pages[0].dtype  # Read dtype from metadata
    


    total_img_width, total_img_height = get_total_canvas_size(tiles_df)

    # Initialize full canvas
    #print(f"shape: {total_img_height, total_img_width}")
    stitched_img = np.zeros([total_img_height, total_img_width], dtype=image_dtype)

    # First image: Never shifted, it's the reference
    row0 = tiles_df.iloc[0]
    box0 = row0['geometry']
    tif_0 = raw_data_folder.joinpath(Path(row0['Filename']).name)
    img0 = np.flipud(tiff.imread(tif_0))  # Flip image

    # Place reference image in full image
    y0, y1 = int(box0.bounds[1]), int(box0.bounds[3])
    x0, x1 = int(box0.bounds[0]), int(box0.bounds[2])
    stitched_img[y0:y1, x0:x1] = img0

    # Step 2: Iterate over all remaining images and align them
    for moving_index in range(1, len(tiles_df)):
        print(f"\nProcessing tile {moving_index}/{len(tiles_df) - 1}...")

        # Reference tile (previous row)
        row_ref = tiles_df.iloc[moving_index - 1]
        box_ref = row_ref['geometry_shifted']
        tif_ref = raw_data_folder.joinpath(Path(row_ref['Filename']).name)
        #w_ref = row_ref['ImageWidth']
        h_ref = row_ref['ImageHeight']

        # Moving tile (current row)
        row_mov = tiles_df.iloc[moving_index]
        box_mov = row_mov['geometry_shifted']
        tif_mov = raw_data_folder.joinpath(Path(row_mov['Filename']).name)
        img_mov = np.flipud(tiff.imread(tif_mov))  # Flip image

        # Compute the intersection area between reference and moving image
        ref_box, mov_box = get_overlap_relative(box_reference=box_ref, box_moving=box_mov)

        # Load only the overlapping part of the reference image
        with tiff.TiffFile(tif_ref) as tif:
            #due to flip
            y0_tmp,y1_tmp = int(ref_box.bounds[1]), int(ref_box.bounds[3])
            y0 = h_ref - y1_tmp
            y1 = h_ref - y0_tmp
            x0,x1 = int(ref_box.bounds[0]), int(ref_box.bounds[2])
            crop_ref = tif.asarray()[y0:y1, x0:x1]
            crop_ref = np.flipud(crop_ref)

        # Extract the overlapping part of the moving image
        crop_mov = img_mov[int(mov_box.bounds[1]):int(mov_box.bounds[3]),
                        int(mov_box.bounds[0]):int(mov_box.bounds[2])]

        # Create masks for saturation and low values
        mask_ref = np.logical_not(mask_low_and_saturation(crop_ref))
        mask_mov = np.logical_not(mask_low_and_saturation(crop_mov))

        # Compute phase cross-correlation shift
        detected_shift, _, _ = phase_cross_correlation(crop_ref, crop_mov, reference_mask=mask_ref, moving_mask=mask_mov)

        print(f"Detected pixel offset (row, col): {-detected_shift}")

        if any(np.abs(detected_shift) > max_shift_pixels):
            print('detected_shift >  max_shift_pixels, weird, setting offset to 0,0')
            detected_shift[0] = 0
            detected_shift[1] = 0

        # Apply shift to position the moving image correctly
        y0 = int(box_mov.bounds[1] + detected_shift[0])
        y1 = int(box_mov.bounds[3] + detected_shift[0])
        x0 = int(box_mov.bounds[0] + detected_shift[1])
        x1 = int(box_mov.bounds[2] + detected_shift[1])

        # Insert the moved image into the full image
        print(f"target location - y:{y0},{y1}; x:{x0},{x1}")
        
        if y0 < 0:
            print("buffer was too small, y0")
            img_mov = img_mov[-int(y0):,:]
        if y1 > stitched_img.shape[0]:
            print("buffer was too small, y1")
            d = y1 - stitched_img.shape[0]
            img_mov = img_mov[0:-int(d),:]
        if x0 < 0:
            print("buffer was too small, x0")
            img_mov = img_mov[:,-int(x0):]
        if x1 > stitched_img.shape[1]:
            print("buffer was too small, x1, cropping")
            d = x1 - stitched_img.shape[1]
            img_mov = img_mov[:, 0:-int(d)]
        print(f"inserted img shape: {img_mov.shape}")

        stitched_img[y0:y1, x0:x1] = img_mov

        # Update the geometry_shifted column
        shifted_box = box(minx=x0, miny=y0, maxx=x1, maxy=y1)
        tiles_df.loc[moving_index, 'geometry_shifted'] = shifted_box

    stitched_img = np.flipud(stitched_img)

    stitched_mask = np.logical_not(mask_low_and_saturation(stitched_img))

    # Find ROI from rotated image
    x0, x1 = first_last_true(np.any(stitched_mask, axis=0))
    y0, y1 = first_last_true(np.any(stitched_mask, axis=1))

    stitched_img = stitched_img[y0:y1, x0:x1]

    print("\n✅ Stitching process completed. now saving")
    extracted_number = extract_s_number(first_tif_path)

    # Define the output file path
    output_tif_path = raw_data_folder.parent.joinpath(f"stitched_image_S_{extracted_number}.tiff")
    output_cc_path = raw_data_folder.parent.joinpath(f"phaseCC_stitching_S_{extracted_number}.csv")
    # Save the full image as a TIFF file
    tiff.imwrite(output_tif_path, stitched_img)

    tiles_df.to_csv(output_cc_path, index=False)

    print("saving done!")

    return stitched_img, tiles_df

# TODO: get a helper function that can stitch based on the csv output, 
# the current method is memory efficient because I have to load the iamges
# however I could just load the overlap region in the moving image as I do
# for the ref. This would split efforts and could perhaps be easier to maintain