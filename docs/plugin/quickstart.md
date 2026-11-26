# Quickstart: stitch and align in napari

This quickstart takes a section set of raw ATLAS tiles through stitching and
an alignment preview. Install the plugin first using the
[plugin installation guide](install.md).

## 1. Open the section set

Launch napari in the `atlas-napari` Conda environment and open **Plugins →
Atlas CCI → AtlasCCI**. Click **Browse...** and select the section-set folder:
the directory containing the `S_...` section folders directly. Do not select
one section folder or its higher-level session directory.

Each section folder needs a completed `.ve-mif` acquisition metadata file and
the TIFF tiles named by that metadata. The plugin lists valid folders it finds.
For example:

```text
Section Set 6/                  ← select this folder
├── S_012_1314243284/
│   ├── MosaicInfo_....ve-mif
│   └── Tile_r1-c1_....tif
└── S_013_1234567890/
    ├── MosaicInfo_....ve-mif
    ├── Tile_r1-c1_....tif
    └── Tile_r2-c1_....tif
```

The names are examples; each `S_` folder may contain one or several tiles.
The acquisition must be marked complete, and the metadata tile list must match
the TIFF files present. See [backend input-data guidance](../backend/input-data.md)
for the input conventions and section ordering.

## 2. Stitch sections

Select one or more sections in the plugin tree, then click **Stitch Selected
Series**. Use **Stitch ALL Series** to process every valid listed section.
Stitched images and metadata are written to `stitched_sections.zarr` in the
selected section-set folder. The number of concurrent sections is set in
**Options...**; more threads use more memory. Select a completed section and
click **Display Stitched Image** to inspect it in napari.

## 3. Calculate and inspect initial alignment

Enter the axial pixel size and select its unit. Click **Initial Z Alignment**
to calculate translations between completed sections and create a downsampled
preview. The plugin uses ascending ATLAS section-number order unless a
`correct_order.txt` file specifies a subset or different order. The file uses
comma-separated section numbers without the `S_` prefix; `...` expands an
inclusive range, for example `1,...,5,7,9,8`.

Scroll through the preview planes to inspect the alignment. The downsampling
factor is available in **Options...**. For the backend notebook workflow and
its outputs, see the [backend quickstart](../backend/quickstart.md).
