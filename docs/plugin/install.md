# Install the napari plugin

The napari plugin provides a graphical workflow for stitching and aligning
FIBICS ATLAS section data. It requires Python 3.11 or 3.12, napari, and a Qt
backend. We recommend a fresh Conda environment to keep the GUI dependencies
separate from other Python installations.

## Recommended installation

Install [Miniforge](https://github.com/conda-forge/miniforge) or another Conda
distribution, then run:

```bash
conda create -n atlas-napari -c conda-forge python=3.11 napari "pyqt6>=6.7" pip
conda activate atlas-napari
python -m pip install napari-atlas-cci
napari
```

In napari, open **Plugins → Atlas CCI → AtlasCCI**. Later, activate the same
environment and run `napari` to start it again. The plugin installs the ATLAS
backend as a dependency. The master Zarr workflow requires `atlas-cci-align`
1.3.19 or later.

The plugin checkout may contain changes newer than the published release. A plain
`pip install napari-atlas-cci` installs the latest stable package available on
the package index, which may not include the current watcher workflow or other
recent changes. If you need features from this checkout, follow the editable
source installation steps in [`plugin/README.md`](https://github.com/CCI-GU-Sweden/ATLAS/blob/main/plugin/README.md),
or install a release candidate only when one is published.

## Qt troubleshooting

The plugin needs a compatible Qt backend. PyQt6 (Qt 6.7 or later) is the
recommended choice. If napari reports an incompatible Qt binding, explicitly
select PyQt6 when launching:

Linux, WSL, or macOS:

```bash
QT_API=pyqt6 napari
```

Windows Command Prompt:

```bat
set QT_API=pyqt6
napari
```

Windows PowerShell:

```powershell
$env:QT_API = "pyqt6"
napari
```

For Linux `xcb` errors or other GUI setup issues, see the
[napari installation guide](https://napari.org/stable/tutorials/fundamentals/installation.html).

## WebKnossos compatibility

The plugin uses `webknossos` for dataset creation and upload. It requires
version 5.1 or later within the 5.x series. Version 5.1.0 has been tested
with the plugin in a napari environment.

## Update

Close napari, activate the environment where the plugin is installed, and
update the plugin and backend together:

```bash
conda activate atlas-napari
python -m pip install --upgrade napari-atlas-cci atlas-cci-align
```

Published packages may lag behind changes on GitHub. For a source installation
and development setup, see the installation instructions in
[`plugin/README.md`](https://github.com/CCI-GU-Sweden/ATLAS/blob/main/plugin/README.md).

For the Python backend and its notebook environment, see
[backend installation](../backend/install.md).
