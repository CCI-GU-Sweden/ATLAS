"""Storage of stitched sections in a resizable Zarr array."""

import json
from pathlib import Path
from threading import Condition, Lock, RLock

import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc

from .utils import parse_shorthand_order


class StitchedZarrStore:
    """Store stitched sections and their metadata in one local Zarr group.

    Images use ``(z, y, x)`` order, lossless Blosc/LZ4 compression, and a zero
    fill value. Z indices describe
    storage order; ATLAS section numbers identify sections independently of
    their physical or acquisition order. Each section has its own metadata
    group under ``sections/S_<number>``.

    Section writes are serialized per store instance. Use
    :meth:`parallel_writer` to coordinate section writes from multiple
    stitching threads and reserve stable Z indices before those threads run.
    This coordination is for threads in one process; it does not coordinate
    multiple processes or multiple store instances opened on the same path.
    """

    def __init__(self, path, *, chunk_yx=(512, 512)):
        self.path = Path(path)
        self._write_lock = RLock()
        self._parallel_writer_lock = Lock()
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

    def parallel_writer(self, section_order):
        """Return a context manager for coordinated writes from worker threads.

        ``section_order`` is an ordered iterable of ATLAS section numbers. Its
        order assigns stable Z indices even when worker threads finish in a
        different order. The writer serializes changes to the shared Zarr
        store while leaving stitching work free to run concurrently.

        Example::

            with store.parallel_writer([2, 7, 12]) as writer:
                # Worker threads may call writer.write_section(...) in any order.
                ...

        Before normal exit, each section must be written or explicitly marked
        failed with ``writer.mark_failed(number, error)``. Existing completed
        sections keep their Z indices and can be skipped with
        ``writer.is_complete(number)`` when resuming a batch. This API is
        thread-safe within this store instance, not across processes.
        """
        return _ParallelSectionWriter(self, section_order)

    def write_section(
        self, *, atlas_section_number, image, acquisition_metadata,
        stitching_metadata=None, transforms=None, source_folder=None,
        tile_dataframe=None, z_index=None,
    ):
        """Write one finished section and return its Z index.

        ``image`` must have the orientation intended for the saved TIFF. Its
        full extent is retained; other sections are padded with zeros as the
        shared canvas grows. Metadata must contain JSON-compatible values.
        If provided, ``tile_dataframe`` is stored as CSV text with no index,
        matching the current companion CSV representation.
        ``z_index`` may be supplied by :meth:`parallel_writer` to preserve a
        preassigned order. When omitted, the section is appended as before.

        To retrieve the tile table for ATLAS section 3::

            from io import StringIO
            import pandas as pd

            record = store.sections["S_3"].attrs
            tile_df = pd.read_csv(StringIO(record["tile_table_csv"]))

        A pending entry is created before writing and marked complete only
        after the image is written. A pending section can be retried with the
        same shape and source folder. Completed section numbers are rejected
        to prevent accidental replacement. Writes are serialized for this
        store instance.
        """
        with self._write_lock:
            return self._write_section(
                atlas_section_number=atlas_section_number,
                image=image,
                acquisition_metadata=acquisition_metadata,
                stitching_metadata=stitching_metadata,
                transforms=transforms,
                source_folder=source_folder,
                tile_dataframe=tile_dataframe,
                z_index=z_index,
            )

    def _write_section(
        self, *, atlas_section_number, image, acquisition_metadata,
        stitching_metadata, transforms, source_folder, tile_dataframe, z_index,
    ):
        if type(atlas_section_number) is not int or atlas_section_number < 0:
            raise ValueError("atlas_section_number must be a nonnegative integer")
        if z_index is not None and (type(z_index) is not int or z_index < 0):
            raise ValueError("z_index must be a nonnegative integer or None")
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
            if z_index is not None and record.get("z_index") != z_index:
                raise ValueError("Requested Z index does not match the pending section")
            if record.get("status") != "reserved" and (
                record.get("shape_yx") != list(image.shape)
                or record.get("source_folder") != source_folder
            ):
                raise ValueError("Retry must use the same section shape and source folder")
            z_index = record["z_index"]
        else:
            indices = [
                self.sections[name].attrs["z_index"]
                for name in self.sections.group_keys()
            ]
            if z_index is None:
                z_index = max(indices, default=-1) + 1
            elif z_index in indices:
                raise ValueError(f"Z index {z_index} is already assigned to another section")
            section = self.sections.create_group(key)

        metadata.update({"z_index": z_index, "status": "pending"})
        if "last_error" in section.attrs:
            del section.attrs["last_error"]
        section.attrs.update(metadata)
        height, width = image.shape
        z_size, y_size, x_size = images.shape
        images.resize((max(z_size, z_index + 1), max(y_size, height), max(x_size, width)))
        images[z_index, :height, :width] = image
        section.attrs["status"] = "complete"
        return z_index

    def validate_section_pixel_sizes(self, tolerance_percent=1.0):
        """Check that each pixel-size source is consistent across completed sections.

        The stored acquisition ``Pixel size`` and tile-TIFF X/Y sizes are checked
        separately because they come from different metadata fields. Values are
        converted to microns, then compared with the median for that field.
        Missing or incomplete metadata, or a deviation above the tolerance, raises
        a clear error. Returns a summary in microns, including the first section
        in numeric ATLAS section order and the maximum deviation from the median.
        """
        section_records = sorted(
            (dict(self.sections[key].attrs) for key in self.sections.group_keys()
             if self.sections[key].attrs.get("status") == "complete"),
            key=lambda record: record["atlas_section_number"],
        )
        if not section_records:
            raise ValueError("No completed sections available for pixel-size validation")

        if not np.isfinite(tolerance_percent) or tolerance_percent < 0:
            raise ValueError("tolerance_percent must be a finite, nonnegative number")

        unit_to_microns = {
            "nm": 0.001,
            "µm": 1.0,
            "μm": 1.0,
            "um": 1.0,
            "mm": 1000.0,
            "m": 1_000_000.0,
        }
        fields = [
            ("Acquisition Pixel size", "Pixel size", "Pixel size unit", True),
            ("Tile TIFF X", "Pixel size tile TIFF X", "Pixel size tile TIFF X unit", False),
            ("Tile TIFF Y", "Pixel size tile TIFF Y", "Pixel size tile TIFF Y unit", False),
        ]
        summary = {}

        for label, value_key, unit_key, required in fields:
            measurements = []
            missing = []
            for record in section_records:
                section_number = record["atlas_section_number"]
                metadata = record.get("acquisition_metadata", {})
                value = metadata.get(value_key)
                unit = metadata.get(unit_key)
                if value is None and unit is None:
                    missing.append(section_number)
                    continue
                if value is None or unit is None:
                    raise ValueError(f"Incomplete {label} pixel-size metadata for S_{section_number}")

                value = float(value)
                unit = str(unit).strip().lower()
                if unit not in unit_to_microns or not np.isfinite(value) or value <= 0:
                    raise ValueError(f"Invalid {label} pixel size for S_{section_number}: {value} {unit}")
                measurements.append((section_number, value * unit_to_microns[unit]))

            if not measurements and not required:
                print(f"No {label} pixel-size metadata found; skipping this check.")
                continue
            if missing:
                raise ValueError(f"{label} pixel-size metadata is missing for sections {missing}")

            values = np.array([value for _, value in measurements], dtype=float)
            median_um = float(np.median(values))
            deviations = np.abs(values - median_um) / median_um * 100
            max_deviation = float(deviations.max())
            print(
                f"{label}: median={median_um:.12g} µm, "
                f"maximum deviation={max_deviation:.4g}% "
                f"(allowed {tolerance_percent:g}%)"
            )
            if max_deviation > tolerance_percent:
                outliers = [
                    f"S_{number}={value:.12g} µm"
                    for (number, value), deviation in zip(measurements, deviations)
                    if deviation > tolerance_percent
                ]
                raise ValueError(
                    f"{label} pixel sizes vary by more than {tolerance_percent:g}%: "
                    + ", ".join(outliers)
                )
            summary[label] = {
                "first_um": measurements[0][1],
                "median_um": median_um,
                "max_deviation_percent": max_deviation,
            }

        return summary


    def ordered_sections(self, order_path=None):
        """Return completed sections in alignment order with their Z indices.

        Sections default to numeric ATLAS section order. If ``order_path``
        points to an existing shorthand order file, its section numbers define
        the order and may select a subset. Each returned dictionary contains
        ``atlas_section_number`` and ``z_index``. Gaps are reported but allowed;
        requested missing or incomplete sections raise ``ValueError``.
        """
        sections = {}
        for key in self.sections.group_keys():
            record = self.sections[key].attrs
            if record.get("status") == "complete":
                number = record["atlas_section_number"]
                sections[number] = {
                    "atlas_section_number": number,
                    "z_index": record["z_index"],
                }

        if not sections:
            raise ValueError("No completed sections found in the stitched Zarr store")

        if order_path is not None and Path(order_path).is_file():
            order = parse_shorthand_order(order_path)
            missing = [number for number in order if number not in sections]
            if missing:
                raise ValueError(f"Sections in {order_path} are not complete or not present: {missing}")
            if not order:
                raise ValueError(f"Section order is empty: {order_path}")
        else:
            order = sorted(sections)
            if order_path is not None:
                print(f"No correct order file found at {order_path}; using numeric S order.")

        sorted_numbers = sorted(set(order))
        gaps = [(left, right) for left, right in zip(sorted_numbers, sorted_numbers[1:])
                if right - left > 1]
        if gaps:
            print(f"Warning: gaps in selected ATLAS section numbers: {gaps}")

        return [sections[number] for number in order]

    def read_section(self, atlas_section_number):
        """Return a completed section without shared-canvas padding."""
        record = self.sections[f"S_{atlas_section_number}"].attrs
        if record.get("status") != "complete":
            raise ValueError(f"Section {atlas_section_number} is not complete")
        height, width = record["shape_yx"]
        return self.group["images"][record["z_index"], :height, :width]


class _ParallelSectionWriter:
    """Reserve stable section indices and serialize writes for one store."""

    def __init__(self, store, section_order):
        self.store = store
        self.section_order = list(section_order)
        if any(type(number) is not int or number < 0 for number in self.section_order):
            raise ValueError("section_order must contain nonnegative integers")
        if len(self.section_order) != len(set(self.section_order)):
            raise ValueError("section_order must not contain duplicates")
        if not self.section_order:
            raise ValueError("section_order must contain at least one section")

        self._entered = False
        self._written = set()
        self._writing = set()
        self._failed = {}
        self._submission_lock = Condition()
        self.z_indices = {}

    def __enter__(self):
        if self._entered:
            raise RuntimeError("This parallel writer is already active")
        if not self.store._parallel_writer_lock.acquire(blocking=False):
            raise RuntimeError("A parallel writer is already active for this store")

        try:
            with self.store._write_lock:
                records = {
                    name: dict(self.store.sections[name].attrs)
                    for name in self.store.sections.group_keys()
                }
                occupied = {record["z_index"] for record in records.values()}
                next_index = max(occupied, default=-1) + 1
                for number in self.section_order:
                    key = f"S_{number}"
                    if key in records:
                        record = records[key]
                        self.z_indices[number] = record["z_index"]
                        if record.get("status") == "complete":
                            self._written.add(number)
                    else:
                        while next_index in occupied:
                            next_index += 1
                        self.z_indices[number] = next_index
                        occupied.add(next_index)
                        section = self.store.sections.create_group(key)
                        section.attrs.update({
                            "atlas_section_number": number,
                            "z_index": next_index,
                            "status": "reserved",
                        })
                        next_index += 1
            self._entered = True
            return self
        except Exception:
            self.store._parallel_writer_lock.release()
            raise

    def write_section(
        self, *, atlas_section_number, image, acquisition_metadata,
        stitching_metadata=None, transforms=None, source_folder=None,
        tile_dataframe=None,
    ):
        """Write one reserved section from a worker thread."""
        if not self._entered:
            raise RuntimeError("Use parallel_writer inside a with statement")
        if atlas_section_number not in self.z_indices:
            raise ValueError(f"Section {atlas_section_number} was not reserved")

        with self._submission_lock:
            if (atlas_section_number in self._written
                    or atlas_section_number in self._writing
                    or atlas_section_number in self._failed):
                raise ValueError(f"Section {atlas_section_number} was already submitted")
            self._writing.add(atlas_section_number)
        try:
            z_index = self.store.write_section(
                atlas_section_number=atlas_section_number,
                image=image,
                acquisition_metadata=acquisition_metadata,
                stitching_metadata=stitching_metadata,
                transforms=transforms,
                source_folder=source_folder,
                tile_dataframe=tile_dataframe,
                z_index=self.z_indices[atlas_section_number],
            )
        except Exception:
            with self._submission_lock:
                self._writing.remove(atlas_section_number)
                self._submission_lock.notify_all()
            raise

        with self._submission_lock:
            self._writing.remove(atlas_section_number)
            self._written.add(atlas_section_number)
            self._submission_lock.notify_all()
        return z_index

    def is_complete(self, atlas_section_number):
        """Return whether this section is complete in the store."""
        if not self._entered:
            raise RuntimeError("Use parallel_writer inside a with statement")
        if atlas_section_number not in self.z_indices:
            raise ValueError(f"Section {atlas_section_number} was not reserved")
        with self._submission_lock:
            return (
                atlas_section_number in self._written
                and atlas_section_number not in self._writing
            )

    def mark_failed(self, atlas_section_number, error):
        """Record a worker failure while leaving its reserved Z index retryable."""
        if not self._entered:
            raise RuntimeError("Use parallel_writer inside a with statement")
        if atlas_section_number not in self.z_indices:
            raise ValueError(f"Section {atlas_section_number} was not reserved")

        message = str(error)
        with self._submission_lock:
            if atlas_section_number in self._written or atlas_section_number in self._writing:
                raise ValueError(f"Section {atlas_section_number} has already been written")
            self._failed[atlas_section_number] = message
        with self.store._write_lock:
            self.store.sections[f"S_{atlas_section_number}"].attrs["last_error"] = message

    @property
    def failed_sections(self):
        """Return a copy of section numbers and errors reported by workers."""
        with self._submission_lock:
            return dict(self._failed)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            with self._submission_lock:
                while self._writing:
                    self._submission_lock.wait()
            if exc_type is None:
                missing = [
                    number for number in self.section_order
                    if number not in self._written and number not in self._failed
                ]
                if missing:
                    raise RuntimeError(f"Reserved sections were not written: {missing}")
        finally:
            self._entered = False
            self.store._parallel_writer_lock.release()
