# atlas-cci-align

Utilities for processing SEM image data exported from ATLAS/FIBICS projects.

The package currently focuses on turning an ATLAS project into stitched 2D
planes, then roughly aligning those planes into a 3D stack. It provides helpers
to:

- read ATLAS mosaic metadata (`.mif`) and per-tile TIFF metadata;
- reconstruct one stitched 2D image per acquisition plane;
- estimate rough pairwise z-alignment between stitched planes;
- apply the cumulative alignment into a Zarr array;
- export the aligned Zarr stack back to a CZI file.

The stitching code uses tile positions, scan rotation, acquisition order, image
size, and pixel size from the ATLAS/FIBICS metadata. ATLAS records when tile
acquisition is finished, so dynamic/incremental processing should be possible in
principle.

## Status

This is early packaging work. The rough stitching and z-alignment utilities are
being moved from notebooks/scripts into an installable package.

## Install

```bash
pip install atlas-cci-align
```

The distribution name is `atlas-cci-align`, while the Python package is imported
as:

```python
import atlas
```

## Minimal Workflow

At a high level, the intended workflow is:

1. Parse an ATLAS mosaic project and build a tile dataframe.
2. Stitch tiles into one 2D plane.
3. Repeat for the planes in the project.
4. Run rough z-alignment across the stitched planes.
5. Save the aligned stack as Zarr.
6. Optionally export the Zarr stack as CZI.

The public API is still being shaped, so consult the modules directly for now:

- `atlas.stitching`
- `atlas.alignment`
- `atlas.io`
- `atlas.image_analysis`

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
adds a configurable 1 µm buffer; the function defaults to no added buffer.
Local examples and comparison results under `private-data/` are Git-ignored.
