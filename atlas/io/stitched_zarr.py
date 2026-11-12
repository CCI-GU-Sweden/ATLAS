"""Sequential storage of stitched sections in a resizable Zarr array."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc


class StitchedZarrStore:
    """Store stitched sections and their metadata in one local Zarr group.

    Images use ``(z, y, x)`` order, lossless Blosc/LZ4 compression, and a zero
    fill value. Z indices describe
    storage order; ATLAS section numbers identify sections independently of
    their physical or acquisition order. Each section has its own metadata
    group under ``sections/S_<number>``.

    This first version supports one sequential writer at a time. Do not write
    or resize the same store concurrently from multiple threads or processes.
    """

    def __init__(self, path, *, chunk_yx=(512, 512)):
        self.path = Path(path)
        if len(chunk_yx) != 2 or any(type(n) is not int or n <= 0 for n in chunk_yx):
            raise ValueError("chunk_yx must contain two positive integers")
        self.chunk_yx = tuple(chunk_yx)
        self.group = zarr.open_group(str(self.path), mode="a", zarr_format=2)
        if "atlas_stitched_store_version" not in self.group.attrs:
            if list(self.group.keys()) or dict(self.group.attrs):
                raise ValueError(f"Not an empty or compatible stitched store: {self.path}")
            self.group.attrs.update({
                "atlas_stitched_store_version": 1,
                "axes": ["z", "y", "x"],
            })
        elif (self.group.attrs["atlas_stitched_store_version"] != 1
              or self.group.attrs.get("axes") != ["z", "y", "x"]):
            raise ValueError(f"Unsupported stitched store format: {self.path}")
        self.sections = self.group.require_group("sections")

    @classmethod
    def open(cls, path, *, chunk_yx=(512, 512)):
        """Open an existing stitched store, or create an empty one."""
        return cls(path, chunk_yx=chunk_yx)

    def write_section(
        self, *, atlas_section_number, image, acquisition_metadata,
        stitching_metadata=None, transforms=None, source_folder=None,
        tile_dataframe=None,
    ):
        """Write one finished section and return its Z index.

        ``image`` must have the orientation intended for the saved TIFF. Its
        full extent is retained; other sections are padded with zeros as the
        shared canvas grows. Metadata must contain JSON-compatible values.
        If provided, ``tile_dataframe`` is stored as CSV text with no index,
        matching the current companion CSV representation.

        To retrieve the tile table for ATLAS section 3::

            from io import StringIO
            import pandas as pd

            record = store.sections["S_3"].attrs
            tile_df = pd.read_csv(StringIO(record["tile_table_csv"]))

        A pending entry is created before writing and marked complete only
        after the image is written. A pending section can be retried with the
        same shape and source folder. Completed section numbers are rejected
        to prevent accidental replacement. This method does not synchronize
        concurrent writers.
        """
        if type(atlas_section_number) is not int or atlas_section_number < 0:
            raise ValueError("atlas_section_number must be a nonnegative integer")
        if not isinstance(image, np.ndarray) or image.ndim != 2 or not all(image.shape):
            raise ValueError("image must be a nonempty 2D NumPy array")
        if not isinstance(acquisition_metadata, dict):
            raise TypeError("acquisition_metadata must be a dictionary")
        if tile_dataframe is not None and not isinstance(tile_dataframe, pd.DataFrame):
            raise TypeError("tile_dataframe must be a pandas DataFrame or None")
        recorded_number = acquisition_metadata.get("atlas_section_number")
        if recorded_number is not None and recorded_number != atlas_section_number:
            raise ValueError("Section number does not match acquisition metadata")

        source_folder = str(source_folder) if source_folder is not None else None
        metadata = {
            "atlas_section_number": atlas_section_number,
            "shape_yx": list(image.shape),
            "source_folder": source_folder,
            "acquisition_metadata": acquisition_metadata,
            "stitching_metadata": stitching_metadata,
            "transforms": transforms,
            "tile_table_csv": (
                tile_dataframe.to_csv(index=False) if tile_dataframe is not None else None
            ),
        }
        # Check before changing the store, including nested metadata.
        json.dumps(metadata, allow_nan=False)

        if "images" not in self.group:
            images = self.group.create_array(
                "images", shape=(0, 0, 0), chunks=(1, *self.chunk_yx),
                dtype=image.dtype, fill_value=0,
                compressor=Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE),
            )
        else:
            images = self.group["images"]
            if images.dtype != image.dtype:
                raise ValueError("Section dtype does not match the store")

        key = f"S_{atlas_section_number}"
        if key in self.sections:
            section = self.sections[key]
            record = dict(section.attrs)
            if record.get("status") == "complete":
                raise ValueError(f"Section {atlas_section_number} is already complete")
            if record.get("shape_yx") != list(image.shape) or record.get("source_folder") != source_folder:
                raise ValueError("Retry must use the same section shape and source folder")
            z_index = record["z_index"]
        else:
            indices = [self.sections[name].attrs["z_index"] for name in self.sections.group_keys()]
            z_index = max(indices, default=-1) + 1
            section = self.sections.create_group(key)

        metadata.update({"z_index": z_index, "status": "pending"})
        section.attrs.update(metadata)
        height, width = image.shape
        z_size, y_size, x_size = images.shape
        images.resize((max(z_size, z_index + 1), max(y_size, height), max(x_size, width)))
        images[z_index, :height, :width] = image
        section.attrs["status"] = "complete"
        return z_index

    def read_section(self, atlas_section_number):
        """Return a completed section without shared-canvas padding."""
        record = self.sections[f"S_{atlas_section_number}"].attrs
        if record.get("status") != "complete":
            raise ValueError(f"Section {atlas_section_number} is not complete")
        height, width = record["shape_yx"]
        return self.group["images"][record["z_index"], :height, :width]
