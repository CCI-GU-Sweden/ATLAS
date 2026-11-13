# atlas-cci-align

Utilities for processing SEM image data exported from ATLAS/FIBICS projects.

ATLAS stitches the tiles in each acquired section, then estimates translations
between consecutive sections to produce a roughly aligned 3D stack. The current
notebook workflow stores stitched sections and their metadata in a master Zarr,
creates an aligned Zarr, and can export the result as CZI.

The master-Zarr workflow is the preferred path forward and is still under active
development. The existing TIFF-based alignment functions remain available during
the transition, with the intention of phasing them out once downstream workflows
have moved to Zarr. The Napari plugin still uses the TIFF-based path and will be
updated separately.

## Install

```bash
pip install atlas-cci-align
```

To work from a local checkout, run these commands from the repository root:

```bash
conda env create --file environment.yml
conda activate atlas-env
```

The environment file also installs this checkout in editable mode, so code
changes are available without reinstalling. Keep the checkout in place.

The distribution name is `atlas-cci-align`, while the Python package is imported
as:

```python
import atlas
```

## Notebook workflow

From a local checkout, open the notebooks with the `atlas-env` Jupyter kernel.
If the kernel is not listed, register it once from the activated environment:

```bash
python -m ipykernel install --user --name atlas-env --display-name "Python ATLAS-env"
```

1. **Stitch sections:** open [ATLAS-raw-import.ipynb](ATLAS-raw-import.ipynb),
   set `series_folder` to the section set containing the `S_...` folders,
   and run the notebook. It writes `stitched_sections.zarr` beside those
   folders. Completed sections already in the store are skipped.
2. **Align sections:** open [ATLAS-z-align-zarr.ipynb](ATLAS-z-align-zarr.ipynb),
   set the same `series_folder` and the axial spacing in microns, then run the
   pairwise alignment and create a downsampled preview. Inspect the preview
   before applying optional manual landmark corrections and exporting the
   full-resolution aligned Zarr or CZI.
3. **Fine alignment:** create a WEBKNOSSOS dataset using the existing helpers
   and use the project's fine-alignment workflow as needed.

The raw import uses cross-correlation to refine overlapping tiles, crops to the
valid stitched region, and stores the oriented result. The master store keeps
one shared `images` array in **ZYX** order. Each `sections/S_<number>` record
contains the ATLAS section number, its separate Z index, original shape,
acquisition and stitching metadata, transforms, source folder, and tile table.
S numbers can be missing and need not match Z indices. Alignment uses numeric
S order by default, or `correct_order.txt` in the section-set folder if present.

The aligned Zarr is a separate ZYX array. Its `AtlasMetadata` includes the
stitching and alignment transforms, acquisition metadata, and references back to
the master store. The backend CZI exporter carries this metadata as a custom
attribute. The aligned writer replaces an existing aligned array at its chosen
output path; the master store is protected from being used as that destination.

Other notebooks:

- [ATLAS-stage-import.ipynb](ATLAS-stage-import.ipynb): stage-only TIFF
  stitching, without cross-correlation or valid-area cropping.
- [ATLAS-z-align.ipynb](ATLAS-z-align.ipynb): the earlier TIFF-based alignment
  workflow.
- [ATLAS-raw-import-parallel.ipynb](ATLAS-raw-import-parallel.ipynb): an
  experimental parallel stitching and TIFF comparison notebook.

## Using the stitched Zarr from Python

```python
from pathlib import Path
from atlas.io import StitchedZarrStore

series_folder = Path("/path/to/section-set")
store = StitchedZarrStore.open(series_folder / "stitched_sections.zarr")

# S numbers identify sections independently of their position in the array.
section = store.read_section(2)  # Original stitched shape, without padding.
record = dict(store.sections["S_2"].attrs)
print(record["z_index"], record["acquisition_metadata"])
print(record["acquisition_metadata"]["atlas_source_folder_name"])

# Respect correct_order.txt if available; otherwise use numeric S order.
section_order = store.ordered_sections(order_path=series_folder / "correct_order.txt")
```

For alignment, `atlas.alignment` provides `initialize_zarr_alignment_df`,
`pairwise_zarr_alignment`, and `calculate_cumulative_shifts`.
`atlas.io.apply_alignment_zarr` writes a separate aligned array. The Zarr
notebook shows these calls and the manual correction workflow in context.
The master and aligned arrays use Zarr format 2; they are not OME-Zarr
pyramids. OME-Zarr compatibility and Zarr v3 are tracked separately in #28.

## Stage-only stitching

For metadata-based placement without cross-correlation or valid-area cropping:

```python
from atlas.io import get_metadata_for_stitched_tif
from atlas.stitching import stage_stitching, save_stitched_image

image, tiles, transforms = stage_stitching(mif_file)
save_stitched_image(
    output_tif_path,
    image,
    get_metadata_for_stitched_tif(mif_file),
    compression="deflate",
)
```

This retains full tiles and zero-filled gaps, using stage positions corrected
for scan rotation. Later tiles in acquisition order overwrite earlier tiles
in overlaps. Deflate compression is lossless. The existing
`stitch_ATLAS_tiles` workflow and default uncompressed saving are unchanged.

Use `ATLAS-stage-import.ipynb` to process a series. It writes
`stage_stitched_S_<number>.tiff`, `stage_stitching_S_<number>.csv`, and
`stage_stitched_transforms_S_<number>.json` next to the section folders,
skipping TIFFs that already exist. The CSV records tile metadata and nominal
placement; the JSON records zero registration corrections. The notebook
defaults to no added buffer, matching the function; raw import uses 1 µm
to allow for registration shifts.
Local examples and comparison results under `private-data/` are Git-ignored.

## Transfer an aligned WEBKNOSSOS region

`transfer_wk_annotation_region.py` downloads the user bounding box from an
annotation on the fine-aligned dataset, then optionally uploads the image data
using a different organization's token. Configure the inputs at the top of the
script. The default test magnification is `32-32-8`; `DO_UPLOAD = False` downloads
only. Tokens can be entered at the top or supplied as `WK_SOURCE_TOKEN` and
`WK_TARGET_TOKEN` environment variables. The destination organization is taken
from the target token; an optional expected organization ID can be specified.
The uploaded name defaults to `<source dataset name>_mag<MAG>`;
set `TARGET_DATASET_NAME` to override it.

The script lists bounding boxes and requires an index when there are several.
It selects image/color layers by default, preserves voxel size and source
coordinates, and stores the subset as compressed Zarr v3 under `private-data/`.
Only the requested magnification is downloaded. With `BUILD_PYRAMID = True`
(the default), the script then creates compressed coarser levels locally using
anisotropic downsampling, up to `PYRAMID_COARSEST_MAG` (default `64`). This
also repairs an existing completed download that has only one resolution.
Use `MAG = "1"` to transfer full-resolution data with a viewing pyramid.
Generation always starts at the downloaded magnification: a `16-16-4` subset
gets only coarser levels, never a synthesized mag 1. The exact levels follow
the voxel-size-aware anisotropic sampling scheme.
Coarse voxels can extend to the next magnification-aligned boundary. No
annotation contents are transferred.

Completed downloads and pyramids are recorded locally and reused for upload
retries. Upload records include the magnifications, so adding a pyramid
triggers a new upload even if an earlier single-resolution upload was recorded.
This creates a new destination dataset; the previous upload is retained.
Interrupted pyramid builds are regenerated from the downloaded resolution.
A partial download or a changed selection requires a new output directory.
For server-side display transformations, use a materialized aligned dataset
before transferring the selected region.

The SDK and compatible IPython/Pydantic versions are included in
`environment.yml`. Run with:

```bash
conda activate atlas-env
python transfer_wk_annotation_region.py
```
