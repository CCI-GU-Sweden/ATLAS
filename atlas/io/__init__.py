"""
IO operations for the Atlas package.
"""
from .utils import is_there_a_single_tif, extract_s_number, create_empty_folder, rm_tree, zarr_array_to_czi, apply_alignment, reorder_files_by_s_number, parse_shorthand_order
from .fibics_metadata import extract_tif_metadata, get_pixel_size_from_tif, get_image_size_from_tif

__all__ = ["is_there_a_single_tif", "extract_s_number", "extract_tif_metadata", "get_pixel_size_from_tif", "create_empty_folder",
           "rm_tree", "zarr_array_to_czi", "apply_alignment", "get_image_size_from_tif", "reorder_files_by_s_number", "parse_shorthand_order"]