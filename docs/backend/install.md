# Install and set up Jupyter

The backend supports Python 3.11 and 3.12.

## Install the published package

For using the Python package without the repository notebooks, create a fresh
Conda environment and install the published distribution:

```bash
conda create -n atlas -c conda-forge python=3.11 pip
conda activate atlas
python -m pip install atlas-cci-align
```

The distribution is named `atlas-cci-align`, and its import name is `atlas`.

## Set up the repository environment

For running notebooks or editing the backend, clone the
[ATLAS repository](https://github.com/CCI-GU-Sweden/ATLAS) and run these
commands from its top-level directory, where `environment.yml` is located:

```bash
conda env create --file environment.yml
conda activate atlas-env
```

This environment includes JupyterLab and `ipykernel`, and installs the backend
in editable mode. Keep the checkout in place while using this environment.
If you already have `atlas-env` but it lacks JupyterLab or `ipykernel`, install
them with:

```bash
conda install -c conda-forge jupyterlab ipykernel
```

Register and start the environment's Jupyter kernel:

```bash
python -m ipykernel install --user --name atlas-env --display-name "Python (ATLAS)"
jupyter lab
```

In JupyterLab, open a notebook from `backend/examples/` and select **Python
(ATLAS)** as its kernel.
