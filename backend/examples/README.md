# ATLAS backend examples

Start with **Raw import** and then **Z alignment from Zarr** for the standard
stitching and alignment workflow. Each notebook expects you to set the input
paths near the top before running it.

## Set up Jupyter

From the ATLAS repository root, create and activate the Conda environment.

```bash
conda env create --file environment.yml
conda activate atlas-env
```

The environment file installs the backend in editable mode and includes
JupyterLab and `ipykernel`. A plain `pip install atlas-cci-align` only installs
the backend. If you are reusing an existing `atlas-env` without JupyterLab or
`ipykernel`, install them into the active environment.

```bash
conda install -c conda-forge jupyterlab ipykernel
```

If your existing environment points to the backend before it moved into
`backend/`, refresh its editable install from the repository root.

```bash
python -m pip install -e ./backend
```

With `atlas-env` still active, register its Python as a Jupyter kernel, then
start JupyterLab.

```bash
python -m ipykernel install --user --name atlas-env --display-name "Python (ATLAS)"
jupyter lab
```

In JupyterLab, open a notebook under `backend/examples/` and select **Python
(ATLAS)** in the kernel selector. If your editor opens notebooks directly,
select the same kernel there.

## Choose an example

| Example | What it does |
| --- | --- |
| [Raw import](ATLAS-raw-import.ipynb) | Stitch raw ATLAS tiles with overlap refinement and save the completed sections to `stitched_sections.zarr`. Start here for new acquisitions. |
| [Z alignment from Zarr](ATLAS-z-align-zarr.ipynb) | Align stitched sections from the master Zarr, inspect the preview, and export the aligned result to Zarr or CZI. |
| [Stage-only import](ATLAS-stage-import.ipynb) | Place raw tiles using stage positions and save stitched TIFFs. Use this when you want placement without overlap refinement. |
| [Parallel raw import](ATLAS-raw-import-parallel.ipynb) | Try experimental parallel stitching and compare its results with the TIFF workflow. |
| [Z alignment from TIFF](ATLAS-z-align.ipynb) | Align a series from existing stitched TIFFs using the earlier workflow. |
| [Fine alignment](ATLAS-fine-alignment.ipynb) | Explore finer registration of previously stitched sections. |
| [OME companion metadata](ATLAS-ome-companion.ipynb) | Generate OME-XML companion metadata from ATLAS TIFF tiles. Install the extra dependency with `python -m pip install ome-types` in `atlas-env` before running this notebook. |
| [Create a WEBKNOSSOS dataset from CZI](create_wk_set_from_czi.py) | Convert a CZI to a local WEBKNOSSOS dataset and build a downsampling pyramid. To upload, set `WK_TARGET_TOKEN` and call `main(do_upload=True)`. |
| [Upload a WEBKNOSSOS dataset](upload_wk_set.py) | Upload an existing local WEBKNOSSOS dataset. Set the input directory and provide the token through `WK_TARGET_TOKEN`. |
| [Transfer a WEBKNOSSOS annotation region](transfer_wk_annotation_region.py) | Download an image region defined by a bounding box and optionally upload it to another organization. Set `DO_UPLOAD = False` for download only. |

Run standalone scripts with `atlas-env` active and configure their inputs
before running. See the [backend README](../README.md) for the workflows and
their outputs.
