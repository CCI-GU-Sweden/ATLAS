"""Write aligned ZYX arrays from a master stitched Zarr store."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from skimage.transform import downscale_local_mean

from atlas.alignment import ROI
from atlas.image_analysis import rescale_image_intensity
from .utils import rm_tree


def apply_alignment_zarr(
    alignment_df, store, buffer_pixels=0, percentile_low=0.5,
    percentile_high=99.9, use_down_sample=False, *, output_path,
):
    """Write aligned ZYX sections from a master stitched Zarr.

    Use the same ROI placement, downsampling, intensity rescaling, and forward
    XY matrices as the TIFF writer. XY coordinates use row vectors, regardless
    of the ZYX image array order. The optional buffer adds an output border.

    Preserve stitching metadata under each output image's record and add
    acquisition metadata and explicit source references. ``filename`` is None
    because the source section is held in Zarr rather than a stitched TIFF.
    The legacy source-coordinate label denotes the same stitched pixel frame.

    Return the aligned array and its path. An existing aligned array at the
    output path is replaced; source stores and their containing directories
    cannot be used as the destination. Process one complete section at a time.
    """
    required = {
        "store_path", "moving_section_number", "moving_z_index",
        "moving_ROI", "cumulative_ROI", "down_scale",
    }
    if not isinstance(alignment_df, pd.DataFrame) or alignment_df.empty:
        raise ValueError("alignment_df must be a nonempty pandas DataFrame")
    missing = required - set(alignment_df.columns)
    if missing:
        raise ValueError(f"alignment_df is missing required columns: {missing}")
    if not isinstance(buffer_pixels, int) or buffer_pixels < 0:
        raise ValueError("buffer_pixels must be a nonnegative integer")
    if not 0 <= percentile_low < percentile_high <= 100:
        raise ValueError("Percentiles must satisfy 0 <= low < high <= 100")

    # Validate the source references and calculate the canvas before writing.
    records = []
    bounds = []
    source_path = store.path.resolve()
    for position, (_, row) in enumerate(alignment_df.iterrows()):
        if Path(row["store_path"]).resolve() != source_path:
            raise ValueError(f"Source store mismatch at row {position}")
        number = row["moving_section_number"]
        record = dict(store.sections[f"S_{number}"].attrs)
        if record.get("status") != "complete" or record["z_index"] != row["moving_z_index"]:
            raise ValueError(f"Source section S_{number} is incomplete or its Z index changed")
        down_scale = row["down_scale"]
        if not np.isfinite(down_scale) or down_scale < 1 or int(down_scale) != down_scale:
            raise ValueError(f"Invalid down_scale at row {position}: {down_scale}")
        factor = 1 if use_down_sample else int(down_scale)
        roi = row["cumulative_ROI"]
        if not isinstance(roi, ROI) or not isinstance(row["moving_ROI"], ROI):
            raise ValueError(f"Missing or invalid ROI at row {position}")
        bounds.append((roi.x0 * factor, roi.x1 * factor, roi.y0 * factor, roi.y1 * factor))
        records.append(record)

    x0_min = int(min(b[0] for b in bounds) - buffer_pixels)
    x1_max = int(max(b[1] for b in bounds) + buffer_pixels)
    y0_min = int(min(b[2] for b in bounds) - buffer_pixels)
    y1_max = int(max(b[3] for b in bounds) + buffer_pixels)
    shape = (len(records), y1_max - y0_min, x1_max - x0_min)
    if min(shape) <= 0:
        raise ValueError(f"Invalid aligned canvas shape: {shape}")

    zarr_path = Path(output_path)
    resolved_output = zarr_path.resolve()
    if resolved_output.is_relative_to(source_path) or source_path.is_relative_to(resolved_output):
        raise ValueError("Aligned output must be separate from the master stitched store")
    if zarr_path.exists():
        if not (zarr_path / ".zarray").is_file():
            raise ValueError(f"Existing output is not a Zarr array: {zarr_path}")
        rm_tree(zarr_path)
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    aligned = zarr.open_array(
        str(zarr_path), mode="w", shape=shape, chunks=(1, 1024, 1024),
        dtype=store.group["images"].dtype, zarr_format=2,
    )
    metadata = {
        "axes": "ZYX", "coordinate_convention": "row_vectors",
        "source_coordinates": "original_stitched_tiff_pixels",
        "target_coordinates": "aligned_zarr_pixels",
        "use_down_sample": bool(use_down_sample),
        "canvas_origin": [x0_min, y0_min], "images": {},
    }
    print(f"Creating aligned Zarr at {zarr_path}, shape {shape}")

    for position, ((_, row), record) in enumerate(zip(alignment_df.iterrows(), records)):
        number = row["moving_section_number"]
        image = store.read_section(number)
        down_scale = int(row["down_scale"])
        if use_down_sample:
            image = downscale_local_mean(image, (down_scale, down_scale)).astype(image.dtype)
        rescaled, _, _ = rescale_image_intensity(
            image, percentile_low=percentile_low, percentile_high=percentile_high,
        )
        factor = 1 if use_down_sample else down_scale
        source_roi = row["moving_ROI"]
        target_roi = source_roi if position == 0 else row["cumulative_ROI"]
        ix0, ix1 = int(source_roi.x0 * factor), int(source_roi.x1 * factor)
        iy0, iy1 = int(source_roi.y0 * factor), int(source_roi.y1 * factor)
        tx0, tx1 = int(target_roi.x0 * factor - x0_min), int(target_roi.x1 * factor - x0_min)
        ty0, ty1 = int(target_roi.y0 * factor - y0_min), int(target_roi.y1 * factor - y0_min)
        if not (0 <= tx0 < tx1 <= shape[2] and 0 <= ty0 < ty1 <= shape[1]):
            raise ValueError(f"Destination ROI for S_{number} is outside the aligned canvas")
        if not (0 <= ix0 < ix1 <= image.shape[1] and 0 <= iy0 < iy1 <= image.shape[0]):
            raise ValueError(f"Source ROI for S_{number} is outside its original section shape")
        if (tx1 - tx0, ty1 - ty0) != (ix1 - ix0, iy1 - iy0):
            raise ValueError(f"Source and destination ROI sizes differ for S_{number}")
        aligned[position, ty0:ty1, tx0:tx1] = rescaled[iy0:iy1, ix0:ix1]

        matrix = np.eye(4)
        scale = 1.0 / down_scale if use_down_sample else 1.0
        matrix[0, 0] = matrix[1, 1] = scale
        matrix[3, 0], matrix[3, 1] = tx0 - ix0, ty0 - iy0
        metadata["images"][str(position)] = {
            "filename": None,
            "source_type": "zarr-store",
            "source_store": str(store.path),
            "source_array": "images",
            "source_z_index": int(record["z_index"]),
            "atlas_section_number": int(number),
            "source_folder": record.get("source_folder"),
            "stitched_to_aligned": matrix.tolist(),
            "down_scale": down_scale,
            "stitching_metadata": deepcopy(record.get("stitching_metadata")),
            "acquisition_metadata": deepcopy(record["acquisition_metadata"]),
            "transforms": deepcopy(record.get("transforms")),
            "tile_table_csv": record.get("tile_table_csv"),
        }
        print(f"Done for output z={position}, S_{number}")

    aligned.attrs["AtlasMetadata"] = metadata
    with zarr_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=4)
    return aligned, zarr_path
