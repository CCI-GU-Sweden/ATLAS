# ATLAS

[![Backend version on PyPI](https://img.shields.io/pypi/v/atlas-cci-align?label=Backend%20PyPI)](https://pypi.org/project/atlas-cci-align/)
[![Plugin version on PyPI](https://img.shields.io/pypi/v/napari-atlas-cci?label=Plugin%20PyPI)](https://pypi.org/project/napari-atlas-cci/)
[![Python 3.11 and 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](backend/README.md#install)
[![Tests](https://github.com/CCI-GU-Sweden/ATLAS/actions/workflows/CI.yaml/badge.svg?branch=main)](https://github.com/CCI-GU-Sweden/ATLAS/actions/workflows/CI.yaml)
[![Docs](https://img.shields.io/badge/docs-ATLAS-blue)](https://cci-gu-sweden.github.io/ATLAS/)
[![Documentation build](https://github.com/CCI-GU-Sweden/ATLAS/actions/workflows/docs-build.yml/badge.svg?branch=main)](https://github.com/CCI-GU-Sweden/ATLAS/actions/workflows/docs-build.yml)
[![MIT license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

ATLAS processes scanning electron microscopy images exported by ATLAS/FIBICS.
It stitches acquisition tiles, aligns serial sections in Zarr, and exports the
aligned stack for viewing and further analysis.

- [Backend installation and usage](backend/README.md) for `atlas-cci-align`,
  imported in Python as `atlas`.
- [Backend examples](backend/examples/README.md) for notebooks, standalone
  scripts, and Jupyter setup.
- [Backend changelog](backend/CHANGELOG.md) for release changes.
- [Napari plugin](plugin/README.md) for the interactive workflow.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and release instructions
and [LICENSE](LICENSE) for the project's MIT license.

## Contributors in alphabetic order

- **[Bosio, Noemi](https://orcid.org/0000-0001-6294-4152)**
- **[Camacho, Rafael](https://orcid.org/0000-0003-2325-6407)**
- **[Fernandez-Rodriguez, Julia](https://orcid.org/0000-0003-4522-0966)**
- **[Gupta, Ketan](https://orcid.org/0009-0000-9875-7324)**
- **[Leclerc, Simon](https://orcid.org/0000-0003-1611-7572)**

Contributions to ATLAS include experiments, scientific discussions, microscopy
expertise, and software development, and are not limited to repository commits.
We also thank current and past Centre for Cellular Imaging colleagues for their
shared expertise, discussions, and laboratory input.
