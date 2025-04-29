import numpy as np
from collections import namedtuple
from shapely.geometry import box

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