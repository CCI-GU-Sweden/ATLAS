"""Helpers for interpreting axis metadata on aligned ATLAS Zarr arrays."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


def normalize_aligned_array_to_zyx(
    data: Any, atlas_metadata: dict | None = None
) -> tuple[Any, dict]:
    """Return aligned array data in ZYX order using its declared axes.

    ATLAS aligned stores declare their axis order in ``AtlasMetadata.axes``.
    Legacy stores use YXZ; current stores use ZYX. No shape-based axis
    inference is safe because the Z dimension may be larger than either XY
    dimension.
    """
    if getattr(data, "ndim", None) != 3:
        raise ValueError(
            f"Expected a 3D aligned Zarr array, got shape {getattr(data, 'shape', None)}."
        )

    if atlas_metadata is None:
        attrs = getattr(data, "attrs", None)
        atlas_metadata = attrs.get("AtlasMetadata") if attrs is not None else None

    if not isinstance(atlas_metadata, dict):
        raise ValueError(
            "Aligned Zarr array is missing AtlasMetadata with an axes declaration."
        )

    axes = atlas_metadata.get("axes")
    if axes not in ("ZYX", "YXZ"):
        raise ValueError(
            f"Expected AtlasMetadata axes 'ZYX' or 'YXZ', got {axes!r}."
        )

    normalized_metadata = deepcopy(atlas_metadata)
    normalized_metadata["axes"] = "ZYX"
    if axes == "ZYX":
        return data, normalized_metadata

    if isinstance(data, np.ndarray):
        return np.transpose(data, (2, 0, 1)), normalized_metadata

    import dask.array as da

    if not isinstance(data, da.Array):
        data = da.from_array(data, chunks=getattr(data, "chunks", "auto"))
    return data.transpose(2, 0, 1), normalized_metadata


def write_atlas_metadata(zarr_path, atlas_metadata: dict) -> None:
    """Write normalized alignment metadata into an existing Zarr array."""
    import zarr

    target = zarr.open_array(str(zarr_path), mode="r+")
    target.attrs["AtlasMetadata"] = atlas_metadata


def write_atlas_metadata_sidecar(zarr_path, atlas_metadata: dict) -> None:
    """Write the alignment metadata sidecar beside an aligned Zarr array."""
    sidecar_path = Path(zarr_path).with_suffix(".json")
    sidecar_path.write_text(
        json.dumps(atlas_metadata, indent=4), encoding="utf-8"
    )
