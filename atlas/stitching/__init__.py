import numpy as np
import xmltodict
import pandas as pd
from pathlib import Path, PureWindowsPath, PurePosixPath
import tifffile as tiff

from collections import namedtuple
from shapely.geometry import box
from dateutil import parser

from atlas.io import get_pixel_size_from_tif, get_image_size_from_tif, extract_s_number
from atlas.image_analysis import mask_low_and_saturation
from atlas.alignment.utils import first_last_true

from skimage.registration import phase_cross_correlation
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from collections import defaultdict, deque

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

def get_total_canvas_size(tile_df, buffer_pixels):
    # based on the tiles dataframe returns the total size of the canvas needed for stitching
    if buffer_pixels == 0:
        buffer_pixels = tile_df['X0_pix'].min()
        print(f"buffer in pixels based on DF output: {buffer_pixels}")
    else:
        buffer_pixels = np.ceil(buffer_pixels).astype(int)
        print(f"buffer in pixels based on displacement: {buffer_pixels}")
        

    # here we generate the full canvas size
    max_X_row = tile_df.loc[tile_df["X0_pix"].idxmax()]
    max_Y_row = tile_df.loc[tile_df["Y0_pix"].idxmax()]
    total_img_width = max_X_row['X0_pix'] + max_X_row['ImageWidth'] + buffer_pixels
    total_img_height = max_Y_row['Y0_pix'] + max_Y_row['ImageHeight'] + buffer_pixels

    print(f"Total image will be of size including buffer: {total_img_width}x{total_img_height}")

    return int(total_img_width), int(total_img_height)

def stitch_ATLAS_tiles_old(tiles_df, raw_data_folder, max_shift_pixels=100):
    # NOTE:
    # `stitch_ATLAS_tiles_old` is the legacy stitching implementation. It aligns
    # tiles sequentially, using the preceding tile as the reference.
    # we now replace it with overlap matching followed by a
    # minimum spanning tree, allowing transforms to be calculated relative
    # to a common reference tile. The legacy function remains in for historical reference.


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

def match_tiles(input_df, reference_idx, min_overlap_percent=2, std_th=2, do_hanning=True):
    """
    Compute stitching cost and pixel-shift vectors between a reference tile and all other tiles.

    For each tile, the function:
      1) checks the precomputed overlap percentage against `min_overlap_percent`,
      2) extracts the overlapping image regions using the tiles' geometries,
      3) builds masks to exclude low-value and saturated pixels,
      4) optionally refines the masks by keeping only pixels above a global
         intensity threshold derived from mean + std_th * std of valid pixels,
      5) estimates translation via masked phase cross-correlation,
      6) computes a heuristic stitching cost.

    Parameters
    ----------
    input_df : pandas.DataFrame
        DataFrame containing tile metadata. Must include:
        - 'geometry' : shapely geometry (tile bounding box in global space)
        - 'ImageWidth' : width in pixels
        - 'ImageHeight' : height in pixels
        - 'Filename' : TIFF filename
        - 'raw_data_folder' : pathlib.Path to folder containing the raw TIFF
        - 'overlap_percent' : list-like of overlap percentages with all other tiles
    reference_idx : int
        Index of the tile used as the reference for cost and shift computation.
    min_overlap_percent : float, optional (default=2)
        Minimum overlap (%) required to attempt matching. Tiles below this threshold
        get cost=1.0 and shift=[0, 0].
    std_th : float or None or False, optional (default=2)
        Optional intensity-based mask refinement.
        - If None or False: do not apply intensity thresholding.
        - If a number: compute pix_th = mean + std_th * std using valid pixels
          from both overlap crops and keep only pixels > pix_th in both masks.
        Note: This assumes foreground is brighter than background; for inverted
        contrast (e.g., some BSD images), this may reject signal.
    do_hanning : bool, optional (default=True)
        If True, multiply each overlap crop by a two-dimensional Hanning window
        before phase cross-correlation. This can reduce boundary artifacts, but it
        also affects the intensity standard deviations used to compute the cost.

    Returns
    -------
    cost_list : list of float
        One stitching cost per tile, in the positional row order of ``input_df``.
        Lower values indicate stronger candidate connections. Tiles below
        ``min_overlap_percent`` receive a cost of 1.0.
    shift_list : list of np.ndarray, shape (2,)
        One detected pixel shift per tile, in the positional row order of
        ``input_df``. Each shift is ``(row, col)`` and aligns that tile to the
        reference tile. If overlap is too small, the shift is ``np.zeros(2)``.

    Raises
    ------
    AssertionError
        If a required DataFrame column is missing, ``reference_idx`` is outside
        the positional bounds of ``input_df``, or ``min_overlap_percent`` is not
        numeric.

    Notes
    -----
    - Shifts follow the `skimage.registration.phase_cross_correlation` convention
      (row, col). Apply with care regarding sign depending on your downstream usage.
    - Masked phase cross-correlation may return NaNs for the error metric (known behavior);
      this function does not use that error value.
    - The heuristic cost is inversely proportional to the average masked intensity
      standard deviation, the fraction of valid moving-mask pixels, and the overlap
      percentage. It is intended for relative edge ranking, not as a normalized
      registration error.
    - If no valid intensities remain after the initial low/saturation masks, the
      optional intensity-threshold refinement is skipped.
    """

    # ------------------------------------------------------------------
    # Assertions: Validate DataFrame structure
    # ------------------------------------------------------------------
    required_cols = [
        'geometry', 'ImageWidth', 'ImageHeight',
        'Filename', 'raw_data_folder', 'overlap_percent'
    ]
    for col in required_cols:
        assert col in input_df.columns, f"Missing required column: '{col}'"

    assert 0 <= reference_idx < len(input_df), "reference_idx is out of DataFrame bounds"
    assert isinstance(min_overlap_percent, (int, float)), "min_overlap_percent must be numeric"

    def load_tiff_array(tif_path):
        """Memory-map a TIFF when possible, otherwise read it normally."""
        try:
            return tiff.memmap(tif_path, page=0, mode="r")
        except ValueError:
            with tiff.TiffFile(tif_path) as tif_file:
                return tif_file.asarray()

    def close_tiff_array(image):
        """Close the file mapping owned by a NumPy memmap array."""
        if isinstance(image, np.memmap):
            image._mmap.close()

    # ------------------------------------------------------------------
    # Setup reference tile
    # ------------------------------------------------------------------
    row_ref = input_df.iloc[reference_idx]
    geometry_ref = row_ref['geometry']
    w_ref = row_ref['ImageWidth']
    h_ref = row_ref['ImageHeight']

    # Load dtype from TIFF
    ref_tif_name = filename_helper(row_ref.Filename)
    ref_tif_path = row_ref.raw_data_folder.joinpath(ref_tif_name)
    ref_image = None
    # if "\\" in row_ref.Filename:
    #     ref_tif_path = row_ref.raw_data_folder.joinpath(PureWindowsPath(row_ref.Filename).name)
    # else:
    #     ref_tif_path = row_ref.raw_data_folder.joinpath(PurePosixPath(row_ref.Filename).name)
    # with tiff.TiffFile(ref_tif_path) as tif:
    #     image_dtype = tif.pages[0].dtype

    print(f"\nProcessing reference tile {reference_idx}...")

    # Prepare outputs
    n_tiles = len(input_df)
    cost_list = []
    shift_list = []

    # ------------------------------------------------------------------
    # Compare reference tile to all other tiles
    # ------------------------------------------------------------------
    for query_idx in range(n_tiles):
        print(f"\nComparing reference {reference_idx} to tile {query_idx}...")

        row_mov = input_df.iloc[query_idx]
        overlap_percentage = row_ref.overlap_percent[query_idx]

        print(f"Overlap %: {overlap_percentage}")

        # Not enough overlap → default cost/shift
        if overlap_percentage < min_overlap_percent:
            print("Too little overlap: assigning cost=1.0 and shift=[0,0]")
            cost_list.append(np.float64(1.0))
            shift_list.append(np.zeros(2))
            continue

        # ------------------------------------------------------------------
        # Compute overlapping bounding boxes
        # ------------------------------------------------------------------
        geometry_mov = row_mov['geometry']
        mov_tif_name = filename_helper(row_mov.Filename)
        mov_tif_path = row_ref.raw_data_folder.joinpath(mov_tif_name)
        # if "\\" in row_mov.Filename:
        #     mov_tif_path = row_ref.raw_data_folder.joinpath(PureWindowsPath(row_mov.Filename).name)
        # else:
        #     mov_tif_path = row_ref.raw_data_folder.joinpath(PurePosixPath(row_mov.Filename).name)

        ref_box, mov_box = get_overlap_relative(
            box_reference=geometry_ref,
            box_moving=geometry_mov
        )

        # ------------------------------------------------------------------
        # Load overlapping region from reference image
        # ------------------------------------------------------------------
        y0_tmp, y1_tmp = int(ref_box.bounds[1]), int(ref_box.bounds[3])
        y0, y1 = h_ref - y1_tmp, h_ref - y0_tmp  # flip correction
        x0, x1 = int(ref_box.bounds[0]), int(ref_box.bounds[2])
        if ref_image is None:
            ref_image = load_tiff_array(ref_tif_path)
        crop_ref = np.flipud(ref_image[y0:y1, x0:x1]).copy()

        # ------------------------------------------------------------------
        # Load overlapping region from moving image
        # ------------------------------------------------------------------
        mov_image = load_tiff_array(mov_tif_path)
        try:
            y0_tmp, y1_tmp = int(mov_box.bounds[1]), int(mov_box.bounds[3])
            y0, y1 = h_ref - y1_tmp, h_ref - y0_tmp
            x0, x1 = int(mov_box.bounds[0]), int(mov_box.bounds[2])
            crop_mov = np.flipud(mov_image[y0:y1, x0:x1]).copy()
        finally:
            close_tiff_array(mov_image)

        # ------------------------------------------------------------------
        # Mask & threshold computation
        # ------------------------------------------------------------------
        mask_ref = ~mask_low_and_saturation(crop_ref)
        mask_mov = ~mask_low_and_saturation(crop_mov)

        current_vals = np.concatenate([
            crop_ref[mask_ref].ravel(),
            crop_mov[mask_mov].ravel()
        ])

        if current_vals.size == 0:
            # fall back to no intensity thresholding
            pix_mean = pix_std = None
        else:
            pix_mean = current_vals.mean()
            pix_std = current_vals.std()

        if (std_th is not None and std_th is not False) and (current_vals.size > 0):
            pix_th = pix_mean + float(std_th) * pix_std
            mask_mov = np.logical_and(mask_mov, crop_mov > pix_th)
            mask_ref = np.logical_and(mask_ref, crop_ref > pix_th)

        mask_pixels = mask_mov.sum()
        mask_pixels_per = mask_pixels / mask_mov.size

        # optional: reduce edge effects (often helps a lot)
        if do_hanning:
                
            wy = np.hanning(crop_ref.shape[0])
            wx = np.hanning(crop_ref.shape[1])
            win = wy[:, None] * wx[None, :]
            crop_ref = crop_ref * win

            wy = np.hanning(crop_mov.shape[0])
            wx = np.hanning(crop_mov.shape[1])
            win = wy[:, None] * wx[None, :]
            crop_mov = crop_mov * win

        # ------------------------------------------------------------------
        # Phase cross-correlation (shift detection)
        # ------------------------------------------------------------------
        #show_two_images(mask_ref, mask_mov)
        detected_shift, _, _ = phase_cross_correlation(
            crop_ref,
            crop_mov,
            reference_mask=mask_ref,
            moving_mask=mask_mov
        )

        print(f"Detected pixel offset (row, col): {-detected_shift}")

        # ------------------------------------------------------------------
        # Cost computation
        # ------------------------------------------------------------------
        ref_std = crop_ref[mask_ref].std()
        mov_std = crop_mov[mask_mov].std()
        avg_std = (ref_std + mov_std) / 2

        cost = 1.0 / (1e-6 + avg_std * mask_pixels_per * overlap_percentage)
        cost_list.append(cost)
        shift_list.append(detected_shift)

    close_tiff_array(ref_image)
    return cost_list, shift_list

def build_adjacency_matrix_from_costs(df, cost_column='stitching_costs'):
    """
    Construct an adjacency matrix from stitching costs to be used for MST computation.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing stitching cost vectors for each tile.
    cost_column : str, optional
        Name of the column containing lists/arrays of stitching costs (default: 'stitching_costs').

    Returns
    -------
    adj_matrix : np.ndarray, shape (n_tiles, n_tiles)
        Adjacency matrix with np.inf for non-edges and stitching costs for valid connections.
    """
    assert cost_column in df.columns, f"Column '{cost_column}' not found in DataFrame."
    
    n_tiles = len(df)
    adj_matrix = np.ones((n_tiles, n_tiles))  # start with 1s (worst or no connection)

    # Fill the matrix with actual stitching costs
    for i in range(n_tiles):
        cost_vector = df['stitching_costs'][i]
        for j in range(n_tiles):
            if i != j and cost_vector[j] < 1.0:
                adj_matrix[i][j] = cost_vector[j]
            else:
                adj_matrix[i][j] = np.inf  # No edge (self or non-overlap)
    
    return adj_matrix

def find_all_paths_to_root(mst_sparse, root):
    mst_edges = np.transpose(mst_sparse.nonzero())
    adj = defaultdict(list)

    for i, j in mst_edges:
        adj[i].append(j)
        adj[j].append(i)  # because it's an undirected tree
    #adj = build_adjacency_from_mst(mst)

    paths = {}

    def dfs(node, parent, path):
        path = path + [node]
        paths[node] = path[::-1]  # reversed: from node to root
        for neighbor in adj[node]:
            if neighbor != parent:
                dfs(neighbor, node, path)

    dfs(root, None, [])
    return paths

def path_to_pairs(path):
    """
    Given a list of tile indices representing a stitching path,
    return a list of (moving_tile, reference_tile) pairs.
    The last pair is a self-reference (e.g., (0, 0)).
    """
    pairs = []
    for i in range(len(path)):
        current = path[i]
        if i < len(path) - 1:
            next_ = path[i + 1]
        else:
            next_ = current  # last: self-reference
        pairs.append((current, next_))
    return pairs

def build_transform_dict_from_mst(mif_tile_df, mst, reference_tile=0, stitching_shift_column='stitching_shifts'):
    """
    Build a dictionary of stitching transformations for all tiles, using a minimum spanning tree (MST)
    and a selected reference tile.

    Parameters
    ----------
    mif_tile_df : pd.DataFrame
        DataFrame with per-tile data, must contain a column of stitching shifts (vectors),
        and be indexable by tile index.
    mst : scipy.sparse.csr_matrix
        Minimum spanning tree in sparse matrix form (output of minimum_spanning_tree).
    reference_tile : int, optional
        Tile index to use as the reference (default is 0).
    stitching_shift_column : str, optional
        Column name in the DataFrame that stores per-tile stitching shift vectors (default: 'stitching_shifts').

    Returns
    -------
    transform_dict : dict
        Dictionary with keys like "t21" meaning transform from tile 2 to tile 1, and values as np.array of shape (2,).
    """

    # The MST will be used to calculate the transofrmation matrices between each tile and a reference tile.
    # For the moment I just pick 0 as reference but maybe there is a better way, in general I dont think it matters much.

    # User input reference_tile and mst, output transform dictionary
    paths_to_reference = find_all_paths_to_root(mst, reference_tile)

    mst_dense = mst.toarray()
    # print("MST adjacency matrix (only connected edges):")
    # print(mst_dense)

    # now calculate inital transforms and store them in a dictionary, by initial I mean they are given by the local conection between tiles
    transform_dict = {}
    mst_edges = np.transpose(mst.nonzero())  # array of (idx_ref, idx_mov) pairs

    for idx_ref, idx_mov in mst_edges:

        cost = mst_dense[idx_ref][idx_mov]
        # print(f"Edge: Reference Tile {idx_ref} - Moving Tile {idx_mov} with cost {cost}")

        # fetch the transform from the dataframe, in this case is a simple translation so a x,y shift vector
        row_ref = mif_tile_df.iloc[idx_ref]
        t_value = row_ref[stitching_shift_column][idx_mov]

        # Create key in the transform dictionary using the format "t{mov}{ref}"
        key = f"t{idx_mov}{idx_ref}"
        transform_dict[key] = t_value

        # now add the oposite for symetry, so we can move back and forth
        key = f"t{idx_ref}{idx_mov}"
        transform_dict[key] = -t_value

    # now add the special case of the reference tile to itsef as all paths should lead to this
    key = f"t{reference_tile}{reference_tile}"
    transform_dict[key] = np.zeros(2)

    # now add all the long range connections where I need to traverse more than one tile edge
    n_tiles = mif_tile_df.shape[0]
    for query_tile in range(n_tiles):
        t_desired = f"t{query_tile}{reference_tile}"
        if t_desired in transform_dict:
            pass
            # print(f"Key {t_desired} exists!, I can do the transform!")
        else:
            # print(f"Key {t_desired} does not exists!, calculating path")
            chain = paths_to_reference[query_tile]

            t_value = np.zeros(2)
            pairs = path_to_pairs(chain)
            for pair in pairs:
                key = f"t{pair[0]}{pair[1]}"
                # print(transform_dict[key])
                t_value = t_value + transform_dict[key]

            transform_dict[t_desired] = t_value

    return transform_dict

def apply_transforms_and_stitch(mif_tile_df, transform_dict, reference_tile=0):
    """
    Apply computed transforms to all tiles and create a single stitched image.

    Parameters
    ----------
    mif_tile_df : pd.DataFrame
        DataFrame containing tile metadata, including:
        - 'Filename' : filename of the tile
        - 'raw_data_folder' : path to folder containing TIFF files
        - 'geometry' : shapely box describing where tile is placed
    transform_dict : dict
        Dictionary of shifts (2D vectors) keyed as 'tXY' meaning from tile X to tile Y.
    reference_tile : int,
        Index of the reference tile (default is 0).

    Returns
    -------
    stitched_img : np.ndarray
        The final stitched image with all tiles placed using their transforms.
    """

    # apply transform, user inputs are transform_dict and mif_tile_df, output is the stitched_img

    max_shift_pixels = max(np.abs(val).max() for val in transform_dict.values())

    #TODO: after testing remove this part is not used any longer
    total_img_width, total_img_height = get_total_canvas_size(mif_tile_df, max_shift_pixels)

    stitched_img = None  # initialize later

    ymin = 0
    ymax = 0
    xmin = 0
    xmax = 0

    for index, row in mif_tile_df.iterrows():
        #print(f"\nIndex: {index}, Filename: {row['Filename']}")
        t_key = f"t{index}{reference_tile}"
        #print(t_key)
        t_val = transform_dict[t_key]
        #print(t_val)
        
        # Compute shifted bounding box position
        box_mov = row.geometry
        y0 = int(box_mov.bounds[1] + t_val[0])
        y1 = int(box_mov.bounds[3] + t_val[0])
        x0 = int(box_mov.bounds[0] + t_val[1])
        x1 = int(box_mov.bounds[2] + t_val[1])

        if y0 < ymin:
            ymin = y0
        if y1 > ymax:
            ymax = y1

        if x0 < xmin:
            xmin = x0
        if x1 > xmax:
            xmax = x1

        #print(f"idx: {y0}-{y1}; {x0}-{x1}")
    ycorr = 0
    xcorr = 0
    if ymin < 0:
        ycorr = -ymin
    if xmin < 0:
        xcorr = -xmin
    
    #print(f"img idx based on geometries: {ymin}-{ymax}; {xmin}-{xmax}")
    img_width = xmax + xcorr
    img_height = ymax + ycorr
    print(f"img shape based on geometries: h-{img_height}; w-{img_width}")

    for index, row in mif_tile_df.iterrows():
        print(f"\nIndex: {index}, Filename: {row['Filename']}")

        t_key = f"t{index}{reference_tile}"

        t_val = transform_dict[t_key]

        # Load current tile image and flip vertically
        tif_current_name = filename_helper(row.Filename)
        tif_current = row.raw_data_folder.joinpath(tif_current_name)
        # if "\\" in row.Filename:
        #     tif_current = row.raw_data_folder.joinpath(PureWindowsPath(row.Filename).name)
        # else:
        #     tif_current = row.raw_data_folder.joinpath(PurePosixPath(row.Filename).name)

        img_current = np.flipud(tiff.imread(tif_current))  # Flip image

        # Initialize stitched canvas on first iteration
        if index == 0:
            stitched_img = np.zeros(
                [img_height, img_width],
                dtype=img_current.dtype
            )
            print(stitched_img.shape)
            #extracted_number = extract_s_number(tif_current)

        # Compute shifted bounding box position
        box_mov = row.geometry
        y0 = int(box_mov.bounds[1] + t_val[0])+ycorr
        y1 = int(box_mov.bounds[3] + t_val[0])+ycorr
        x0 = int(box_mov.bounds[0] + t_val[1])+xcorr
        x1 = int(box_mov.bounds[2] + t_val[1])+xcorr

        print(f"idx: {y0}-{y1}; {x0}-{x1}")

        stitched_img[y0:y1, x0:x1] = img_current

    return stitched_img


def stitch_ATLAS_tiles(
    mif_file,
    *,
    buffer_microns=1,
    max_shift_pixels=200,
    min_overlap_percent=2,
    std_th=0,
    do_hanning=False,
    reference_tile=0,
):
    """
    Stitch the tiles described by a FIBICS .ve-mif file.

    The function calculates pairwise tile registrations, constructs a
    minimum spanning tree, and places every tile relative to a selected
    reference tile.

    It performs no output-file naming or saving.

    Parameters
    ----------
    mif_file : pathlib.Path
        Path to the .ve-mif file. The corresponding TIFF tiles are
        expected in the same directory.
    buffer_microns : float, default=1
        Buffer added around the nominal tile positions.
    max_shift_pixels : float, default=200
        Maximum accepted Euclidean registration shift. Larger shifts
        are replaced with a zero shift and a weak matching cost.
    min_overlap_percent : float, default=2
        Minimum tile overlap required for registration.
    std_th : float or None, default=0
        Intensity threshold passed to ``match_tiles``.
    do_hanning : bool, default=False
        Whether ``match_tiles`` applies a Hanning window.
    reference_tile : int, default=0
        Tile used as the common transformation reference.

    Returns
    -------
    stitched_img : numpy.ndarray
        Cropped stitched image in the internal stitching orientation.
    mif_tile_df : pandas.DataFrame
        Parsed tile metadata with overlap, cost, and shift columns.
    transform_dict : dict[str, numpy.ndarray]
        Tile transformations calculated from the minimum spanning tree.
    """
    mif_file = Path(mif_file)

    if not mif_file.is_file():
        raise FileNotFoundError(f"MIF file does not exist: {mif_file}")

    if mif_file.suffix.lower() != ".ve-mif":
        raise ValueError(f"Expected a .ve-mif file, got: {mif_file.name}")

    if max_shift_pixels < 0:
        raise ValueError("max_shift_pixels must be non-negative")

    raw_data_folder = mif_file.parent

    # Resetting the index is important because subsequent stitching
    # functions treat DataFrame indices as positional tile indices.
    mif_tile_df = get_tiles_dataframe(
        mif_file,
        buffer_microns=buffer_microns,
    ).reset_index(drop=True)

    if mif_tile_df.empty:
        raise ValueError(f"No tiles found in {mif_file}")

    if not 0 <= reference_tile < len(mif_tile_df):
        raise ValueError(
            f"reference_tile must be between 0 and "
            f"{len(mif_tile_df) - 1}"
        )

    mif_tile_df = add_tile_overlap_columns(mif_tile_df)
    mif_tile_df["raw_data_folder"] = raw_data_folder

    all_costs = []
    all_shifts = []

    for current_idx in range(len(mif_tile_df)):
        costs, shifts = match_tiles(
            mif_tile_df,
            reference_idx=current_idx,
            min_overlap_percent=min_overlap_percent,
            std_th=std_th,
            do_hanning=do_hanning,
        )

        for index, shift in enumerate(shifts):
            if np.linalg.norm(shift) > max_shift_pixels:
                # Retain a weak edge so the graph has a chance to
                # remain connected, but do not apply the bad shift.
                costs[index] = 0.9
                shifts[index] = np.zeros(2)

        all_costs.append(costs)
        all_shifts.append(shifts)

    mif_tile_df["stitching_costs"] = all_costs
    mif_tile_df["stitching_shifts"] = all_shifts

    adjacency_matrix = build_adjacency_matrix_from_costs(
        mif_tile_df,
        cost_column="stitching_costs",
    )

    mst = minimum_spanning_tree(csr_matrix(adjacency_matrix))

    expected_edges = max(0, len(mif_tile_df) - 1)
    if mst.nnz != expected_edges:
        raise ValueError(
            "The tile-overlap graph is disconnected: "
            f"expected {expected_edges} MST edges, found {mst.nnz}."
        )

    transform_dict = build_transform_dict_from_mst(
        mif_tile_df,
        mst,
        reference_tile=reference_tile,
    )

    stitched_img = apply_transforms_and_stitch(
        mif_tile_df,
        transform_dict,
        reference_tile=reference_tile,
    )

    valid_mask = ~mask_low_and_saturation(stitched_img)
    valid_rows, valid_columns = np.nonzero(valid_mask)

    if valid_rows.size == 0:
        raise ValueError("The stitched image contains no valid pixels")

    stitched_img = stitched_img[
        valid_rows.min() : valid_rows.max() + 1,
        valid_columns.min() : valid_columns.max() + 1,
    ]

    return stitched_img, mif_tile_df, transform_dict


def filename_helper(path_to_file):
    if "\\" in path_to_file:
        return PureWindowsPath(path_to_file).name
    else:
        return PurePosixPath(path_to_file).name
