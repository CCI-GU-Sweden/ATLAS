# Quickstart: stitch and align a series

This workflow takes raw ATLAS section folders through a stitched master Zarr
and an alignment preview. Run it in the repository's `atlas-env` environment;
see [installation and Jupyter setup](install.md). The notebooks and source
files are tracked in the repository and linked below.

## 1. Stitch raw tiles into the master Zarr

Open [ATLAS raw import](https://github.com/CCI-GU-Sweden/ATLAS/blob/main/backend/examples/ATLAS-raw-import.ipynb).
Set `series_folder` near the top to the section-set directory containing the
`S_...` folders, then run the notebook. It finds section folders with `.ve-mif`
files, stitches their tiles with overlap refinement, and writes completed
sections to `stitched_sections.zarr` beside those folders. Sections already
marked complete in the store are skipped on later runs.

## 2. Align the sections and inspect a preview

Open [Z alignment from Zarr](https://github.com/CCI-GU-Sweden/ATLAS/blob/main/backend/examples/ATLAS-z-align-zarr.ipynb).
Set `series_folder` to the same section-set directory and set
`AXIAL_PIXEL_MICRONS` to the section spacing in microns. The notebook reads the
master store, determines section order, estimates pairwise translations, and
saves alignment results under `alignment_results/`. It then creates a
downsampled `aligned_preview.zarr`; inspect that preview before proceeding.

If the section set contains `correct_order.txt`, the notebook uses it to set
the order. Otherwise it uses numeric ATLAS section order. ATLAS section
numbers and Z indices are separate values.

## 3. Export after review

After reviewing the preview, you can use the notebook's optional manual
landmark correction section, regenerate and inspect the preview, and write the
full-resolution result to `aligned_full_resolution.zarr`. The notebook also
shows an optional CZI export. The aligned output is separate from the master
store; keep the master store as the source for future alignment runs.
