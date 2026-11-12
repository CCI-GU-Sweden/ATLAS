"""Experiment: stitch several S folders concurrently and store them in Zarr.

This mirrors the Napari plugin's per-section ``ThreadPoolExecutor`` workflow.
Each worker stitches one section, then acquires a lock before updating the
shared :class:`StitchedZarrStore`. Stitching can therefore overlap, while the
current store's resize and metadata operations remain sequential.

The output is written under a new temporary directory so this experiment does
not modify either source dataset or an existing Zarr store.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from tempfile import mkdtemp
from threading import current_thread
from time import perf_counter

import numpy as np
import tifffile

from atlas.io import (
    StitchedZarrStore,
    get_metadata_for_stitched_tif,
    get_s_number_from_mif,
    extract_s_number,
)
from atlas.io.fibics_metadata import find_slice_valid_path
from atlas.stitching import stitch_ATLAS_tiles


# Small representative samples from both datasets. Edit these paths and S
# numbers as needed; the experiment keeps its output separate under /tmp.
DATASETS = [
    {
        "name": "section_set_6",
        "path": Path("/home/xcamra/projects/data4test/Section Set 6"),
        "section_numbers": [1, 2, 3],
    },
    {
        "name": "zeinab_roi",
        "path": Path("/home/xcamra/projects/data4test/ZEINAB-ROI-w1-20nm-bsd"),
        "section_numbers": [2, 3, 4],
    },
]

MAX_WORKERS = 2
BUFFER_IN_MICRONS = 1
MAX_SHIFT_PIXELS = 200


@dataclass
class StitchResult:
    """Image and metadata returned by one section's stitching worker."""

    section_number: int
    image: np.ndarray
    acquisition_metadata: dict
    stitching_metadata: dict | None
    transforms: dict
    tile_dataframe: object
    source_folder: Path
    stitch_seconds: float


def _find_mif_file(series_folder: Path) -> Path:
    """Return the section's MIF file, requiring exactly one match."""
    mif_files = [
        path for path in series_folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".ve-mif"
    ]
    if len(mif_files) != 1:
        raise ValueError(
            f"Expected one .ve-mif file in {series_folder}, found {len(mif_files)}"
        )
    return mif_files[0]


def _stitch_section(series_folder: Path) -> StitchResult:
    """Run the same per-section stitching call used by the import notebook."""
    mif_file = _find_mif_file(series_folder)
    section_number = get_s_number_from_mif(mif_file)
    acquisition_metadata = get_metadata_for_stitched_tif(mif_file)

    started = perf_counter()
    stitched_image, tile_dataframe, transform_dict, stitching_metadata = (
        stitch_ATLAS_tiles(
            mif_file,
            buffer_microns=BUFFER_IN_MICRONS,
            max_shift_pixels=MAX_SHIFT_PIXELS,
        )
    )
    stitch_seconds = perf_counter() - started

    transforms = {key: value.tolist() for key, value in transform_dict.items()}
    return StitchResult(
        section_number=section_number,
        image=np.flipud(stitched_image),
        acquisition_metadata=acquisition_metadata,
        stitching_metadata=stitching_metadata,
        transforms=transforms,
        tile_dataframe=tile_dataframe,
        source_folder=series_folder,
        stitch_seconds=stitch_seconds,
    )


def _image_checksum(image: np.ndarray) -> str:
    """Return a checksum for verifying that the stored section is unchanged."""
    return sha256(np.ascontiguousarray(image).view(np.uint8)).hexdigest()


def _stitch_and_store(series_folder, writer):
    """Stitch one section and submit it to the shared Zarr writer."""
    started = perf_counter()
    result = _stitch_section(series_folder)
    write_started = perf_counter()
    z_index = writer.write_section(
        atlas_section_number=result.section_number,
        image=result.image,
        acquisition_metadata=result.acquisition_metadata,
        stitching_metadata=result.stitching_metadata,
        transforms=result.transforms,
        source_folder=result.source_folder,
        tile_dataframe=result.tile_dataframe,
    )
    write_seconds = perf_counter() - write_started
    elapsed = perf_counter() - started
    print(
        f"{current_thread().name}: S_{result.section_number} stitched in "
        f"{result.stitch_seconds:.1f}s; stored at z={z_index}; "
        f"Zarr write {write_seconds:.2f}s; total {elapsed:.1f}s",
        flush=True,
    )
    return {
        "section_number": result.section_number,
        "z_index": z_index,
        "shape": result.image.shape,
        "checksum": _image_checksum(result.image),
        "stitch_seconds": result.stitch_seconds,
    }


def _select_series(dataset_path: Path, section_numbers: list[int]) -> list[Path]:
    """Validate only the requested sections and return them in S-number order."""
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_path}")

    by_number = {
        extract_s_number(folder): folder
        for folder in dataset_path.iterdir()
        if folder.is_dir() and extract_s_number(folder) is not None
    }
    selected = []
    for number in section_numbers:
        folder = by_number.get(number)
        if folder is None:
            raise ValueError(f"S_{number} was not found in {dataset_path}")
        valid_folder, reason = find_slice_valid_path(folder)
        if valid_folder is None:
            raise ValueError(f"S_{number} is invalid: {reason}")
        selected.append(valid_folder)
    return selected


def run_experiment():
    """Stitch selected sections concurrently, then check their stored images."""
    output_root = Path(mkdtemp(prefix="atlas_parallel_stitching_"))
    print(f"Zarr experiment outputs: {output_root}", flush=True)
    print(f"Workers: {MAX_WORKERS}; Zarr updates use parallel_writer", flush=True)

    for dataset in DATASETS:
        name = dataset["name"]
        series_folders = _select_series(dataset["path"], dataset["section_numbers"])
        store = StitchedZarrStore.open(output_root / f"{name}.zarr")
        section_order = [
            get_s_number_from_mif(_find_mif_file(folder)) for folder in series_folders
        ]

        print(f"\n{name}: {len(series_folders)} sections", flush=True)
        results = []
        started = perf_counter()
        with store.parallel_writer(section_order) as writer:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(_stitch_and_store, folder, writer): folder
                    for folder in series_folders
                }
                for future in as_completed(futures):
                    folder = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as error:
                        raise RuntimeError(f"Failed while processing {folder}") from error

        # Read every section back and compare with the sequential TIFF outputs.
        for result in results:
            section_number = result["section_number"]
            image = store.read_section(section_number)
            if image.shape != result["shape"]:
                raise AssertionError(f"Unexpected shape for S_{section_number}")
            if _image_checksum(image) != result["checksum"]:
                raise AssertionError(f"Stored pixels differ for S_{section_number}")

            reference_tiff = dataset["path"] / f"stitched_image_S_{section_number}.tiff"
            reference_csv = dataset["path"] / f"phaseCC_stitching_S_{section_number}.csv"
            reference_metadata = dataset["path"] / f"stitched_image_S_{section_number}_metadata.json"
            reference_transforms = dataset["path"] / f"transforms_S_{section_number}.json"
            np.testing.assert_array_equal(tifffile.imread(reference_tiff), image)

            record = dict(store.sections[f"S_{section_number}"].attrs)
            if record["z_index"] != result["z_index"]:
                raise AssertionError(f"Unexpected Z index for S_{section_number}")
            with tifffile.TiffFile(reference_tiff) as tif:
                reference_acquisition = tif.shaped_metadata[0]["AcquisitionMetadata"]
            if record["acquisition_metadata"] != reference_acquisition:
                raise AssertionError(f"Acquisition metadata differs for S_{section_number}")
            if record["tile_table_csv"] != reference_csv.read_text():
                raise AssertionError(f"Tile table differs for S_{section_number}")

            if record["stitching_metadata"] != json.loads(reference_metadata.read_text()):
                raise AssertionError(f"Stitching metadata differs for S_{section_number}")
            if record["transforms"] != json.loads(reference_transforms.read_text()):
                raise AssertionError(f"Transforms differ for S_{section_number}")

        elapsed = perf_counter() - started
        print(
            f"Verified {len(results)} sections for {name} in {elapsed:.1f}s; "
            f"Zarr: {store.path}"
        )

    print(f"\nExperiment complete. Inspect or remove the outputs at: {output_root}", flush=True)


if __name__ == "__main__":
    run_experiment()
