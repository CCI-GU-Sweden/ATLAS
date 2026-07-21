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
principle, but this package does not yet provide a controlled or well-tested
dynamic workflow.

## Status

This is early packaging work. The rough stitching and z-alignment utilities are
being moved from notebooks/scripts into an installable package.

Fine z-alignment is not implemented yet, but it is planned.

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
