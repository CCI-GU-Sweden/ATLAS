"""Consecutive-section alignment using a master stitched Zarr store."""

from pathlib import Path

import numpy as np
import pandas as pd
from skimage.transform import downscale_local_mean

from .utils import get_translation


def initialize_zarr_alignment_df(store, section_order, down_scale=10):
    """Initialize pairwise alignment rows for ordered stitched Zarr sections.

    Each row identifies the moving section and its preceding reference by
    ATLAS section number and stored Z index. The first section refers to
    itself. Store paths and section identifiers remain usable after the
    DataFrame is saved; no open Zarr handle or image data is stored in it.
    """
    rows = []
    for index, moving in enumerate(section_order):
        reference = section_order[index - 1] if index else moving
        rows.append({
            "store_path": str(store.path),
            "moving_section_number": moving["atlas_section_number"],
            "reference_section_number": reference["atlas_section_number"],
            "moving_z_index": moving["z_index"],
            "reference_z_index": reference["z_index"],
            "moving_ROI": None,
            "reference_ROI": None,
            "current_shift": np.array([0., 0.]),
            "cumulative_ROI": None,
            "cumulative_shift": np.array([0., 0.]),
            "down_scale": down_scale,
        })
    return pd.DataFrame(rows)


def pairwise_zarr_alignment(alignment_df, store):
    """Calculate pairwise translations using sections in a stitched Zarr store.

    Each section is read at its original stitched shape. Downsampling and
    translation estimation use the same steps as the TIFF alignment workflow.
    The input DataFrame is updated with ROIs and shifts and returned.
    """
    required = {
        "store_path", "moving_section_number", "reference_section_number",
        "moving_z_index", "reference_z_index", "down_scale",
        "moving_ROI", "reference_ROI", "current_shift",
    }
    if not isinstance(alignment_df, pd.DataFrame):
        raise TypeError("alignment_df must be a pandas DataFrame")
    missing = required - set(alignment_df.columns)
    if missing:
        raise ValueError(f"alignment_df is missing required columns: {missing}")

    alignment_df["current_shift"] = alignment_df["current_shift"].astype(object)
    for index, row in alignment_df.iterrows():
        if Path(row["store_path"]).resolve() != store.path.resolve():
            raise ValueError(f"Store path does not match the alignment DataFrame at row {index}")
        down_scale = row["down_scale"]
        if not isinstance(down_scale, (int, float)) or down_scale <= 0:
            raise ValueError(f"Invalid down_scale at row {index}: {down_scale}")

        reference_number = row["reference_section_number"]
        moving_number = row["moving_section_number"]
        for prefix, number in (("reference", reference_number), ("moving", moving_number)):
            record = store.sections[f"S_{number}"].attrs
            if record["z_index"] != row[f"{prefix}_z_index"]:
                raise ValueError(f"Stored Z index changed for S_{number} at row {index}")

        reference_image = store.read_section(reference_number)
        moving_image = store.read_section(moving_number)
        reference_ds = downscale_local_mean(reference_image, (down_scale, down_scale)).astype(reference_image.dtype)
        moving_ds = downscale_local_mean(moving_image, (down_scale, down_scale)).astype(moving_image.dtype)
        reference_roi, moving_roi, shift = get_translation(reference_ds, moving_ds)

        alignment_df.at[index, "reference_ROI"] = reference_roi
        alignment_df.at[index, "moving_ROI"] = moving_roi
        alignment_df.at[index, "current_shift"] = shift
        print(f"S_{reference_number} → S_{moving_number}: current shift {shift}")

    return alignment_df
