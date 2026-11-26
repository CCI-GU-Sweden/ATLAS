# Backend examples

Run notebooks using the repository's `atlas-env` environment and select its
**Python (ATLAS)** kernel. See [installation and Jupyter setup](install.md).
Configure each example's input paths before running it. Links below open the
tracked notebooks and scripts on GitHub.

| Example | What it does |
| --- | --- |
| [Raw import](https://github.com/CCI-GU-Sweden/ATLAS/blob/main/backend/examples/ATLAS-raw-import.ipynb) | Stitch raw ATLAS tiles with overlap refinement and save completed sections to `stitched_sections.zarr`. |
| [Z alignment from Zarr](https://github.com/CCI-GU-Sweden/ATLAS/blob/main/backend/examples/ATLAS-z-align-zarr.ipynb) | Align sections from the master Zarr, inspect a preview, and export aligned output to Zarr or CZI. |
| [Stage-only import](https://github.com/CCI-GU-Sweden/ATLAS/blob/main/backend/examples/ATLAS-stage-import.ipynb) | Place raw tiles using stage positions and save stitched TIFFs without overlap refinement. |
| [Parallel raw import](https://github.com/CCI-GU-Sweden/ATLAS/blob/main/backend/examples/ATLAS-raw-import-parallel.ipynb) | Experimental parallel stitching and TIFF comparison workflow. |
| [Z alignment from TIFF](https://github.com/CCI-GU-Sweden/ATLAS/blob/main/backend/examples/ATLAS-z-align.ipynb) | Earlier alignment workflow using stitched TIFF files. |
| [Fine alignment](https://github.com/CCI-GU-Sweden/ATLAS/blob/main/backend/examples/ATLAS-fine-alignment.ipynb) | Explore finer registration of previously stitched sections. |
| [OME companion metadata](https://github.com/CCI-GU-Sweden/ATLAS/blob/main/backend/examples/ATLAS-ome-companion.ipynb) | Generate OME-XML companion metadata from ATLAS TIFF tiles. Requires `ome-types` in the active environment. |
| [Create a WEBKNOSSOS dataset from CZI](https://github.com/CCI-GU-Sweden/ATLAS/blob/main/backend/examples/create_wk_set_from_czi.py) | Convert a CZI to a local WEBKNOSSOS dataset and build a downsampling pyramid. |
| [Upload a WEBKNOSSOS dataset](https://github.com/CCI-GU-Sweden/ATLAS/blob/main/backend/examples/upload_wk_set.py) | Upload a local WEBKNOSSOS dataset. Configure the input and provide `WK_TARGET_TOKEN`. |
| [Transfer a WEBKNOSSOS annotation region](https://github.com/CCI-GU-Sweden/ATLAS/blob/main/backend/examples/transfer_wk_annotation_region.py) | Download an image region defined by an annotation bounding box and optionally upload it. |

For the recommended raw tile to master Zarr to alignment preview workflow, see
the [quickstart](quickstart.md). For additional backend details, see the
[backend README](https://github.com/CCI-GU-Sweden/ATLAS/blob/main/backend/README.md).
