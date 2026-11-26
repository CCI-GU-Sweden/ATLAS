# ATLAS

ATLAS provides Python tools for processing SEM image data exported from
ATLAS/FIBICS projects. The backend stitches image tiles within each acquired
section and estimates translations between consecutive sections to create an
aligned 3D stack.

The recommended workflow stores stitched sections and their metadata in a
master Zarr, then writes alignment results to a separate Zarr. You can inspect
a downsampled preview before exporting the full-resolution result. The master
Zarr workflow is under active development; TIFF-based workflows remain
available during the transition.

- [Install the backend and set up Jupyter](backend/install.md)
- [Run the raw import and Zarr alignment workflow](backend/quickstart.md)
- [Prepare input data](backend/input-data.md) and understand the
  [workflow outputs](backend/outputs.md)
- [Browse notebooks and scripts](backend/examples.md)

The Python distribution is `atlas-cci-align`; import it as `atlas`.

For a graphical workflow in napari, see the [plugin installation guide](plugin/install.md)
and [plugin quickstart](plugin/quickstart.md).
