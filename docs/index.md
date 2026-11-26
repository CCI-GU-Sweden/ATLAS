# ATLAS

ATLAS helps process SEM image data exported from ATLAS/FIBICS projects. It
stitches tiles within each acquired section and aligns sections to create a
3D stack. Choose the Python backend for scripts and notebooks, or use the
napari plugin for a graphical workflow.

## Backend

The backend provides Python tools and example notebooks for stitching,
alignment and export. Its distribution is `atlas-cci-align`; import it as
`atlas`.

- [Install the backend and set up Jupyter](backend/install.md)
- [Start with the backend quickstart](backend/quickstart.md)
- [Explore the examples](backend/examples.md)

## Plugin

The napari plugin provides a graphical interface for stitching sections,
reviewing alignment and producing aligned output. It also supports completed
sections from file watcher output.

- [Install the plugin](plugin/install.md)
- [Start with the plugin quickstart](plugin/quickstart.md)
- [Explore the plugin workflow](plugin/workflow.md)

## Data and outputs

The recommended workflow stores stitched sections and metadata in a master
Zarr, then creates separate alignment results and a downsampled preview for
review before full-resolution export. The master Zarr workflow is under active
development; TIFF-based workflows remain available during the transition.
See [input data](backend/input-data.md) and [outputs](backend/outputs.md) for
details.
