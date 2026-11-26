# Master and aligned outputs

The recommended workflow keeps stitched source data separate from alignment
products. Run [Raw import](https://github.com/CCI-GU-Sweden/ATLAS/blob/main/backend/examples/ATLAS-raw-import.ipynb)
first, then [Z alignment from Zarr](https://github.com/CCI-GU-Sweden/ATLAS/blob/main/backend/examples/ATLAS-z-align-zarr.ipynb).
Both notebooks write their outputs next to the section folders.

## Master stitched store

Raw import creates `stitched_sections.zarr`, a Zarr format 2 group with a
shared `images` array in ZYX order and a `sections/S_<number>` record for each
section. The common canvas can be larger than an individual section; use
`store.read_section(number)` to read the original unpadded image. Records keep
the separate Z index, original shape, source folder, acquisition and stitching
metadata, tile transforms, and tile table. Completed records are skipped on a
rerun, so an interrupted import can continue. Keep this store as the source
for future alignment runs.

## Alignment results and preview

The Zarr alignment notebook checks section pixel sizes, reads the order from
`correct_order.txt` when present (otherwise numeric S order), and aligns
consecutive sections. It saves the alignment table at
`alignment_results/z_alignment_results.pkl`. Pairwise translations are
calculated on downsampled sections; the example sets `down_scale = 10`.

It writes `aligned_preview.zarr` as a downsampled ZYX array. Inspect this
preview before exporting the full-resolution output. The notebook also
provides an optional manual landmark correction cycle: points are entered in
the current preview's `(Z, Y, X)` pixel coordinates, then the preview and
alignment table are regenerated. Reload the updated preview before making
another correction.

## Full aligned output and CZI

After review, the notebook reads the latest saved alignment table and writes
`aligned_full_resolution.zarr`. This output is a separate Zarr array in ZYX
order, with an `AtlasMetadata` attribute containing source references,
acquisition and stitching metadata, transforms, and the alignment transforms.
The writer replaces an existing aligned array at the chosen output path, so
choose a dedicated output path and do not point it inside the master store.

CZI export is optional. The notebook includes a `zarr_array_to_czi` call for
exporting either preview or full-resolution data; set the in-plane and axial
pixel sizes correctly before using it. The full-resolution CZI call is
commented out by default. For the notebook parameters and workflow sequence,
see the [quickstart](quickstart.md); for code examples and other tools, see
[Examples](examples.md) and the [backend README](https://github.com/CCI-GU-Sweden/ATLAS/blob/main/backend/README.md).
