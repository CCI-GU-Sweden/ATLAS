# atlas-cci-align changelog

## 1.4.0

- Move the backend and Python examples into `backend/`. The published
  distribution name and `atlas` import stay the same.
- Keep backend version discovery independent of Napari plugin tags.
- Adopt explicit release tags and publish reviewed build artifacts.
- Declare and include the MIT license in distributions, and exclude notebooks,
  standalone scripts, experiments, and CI workflows from the source archive.

## 1.4.0rc1 — 2026-11-25

- Use explicit release tags to build and verify distributions before publishing.
- Declare the MIT license with SPDX metadata and ship the license in both
  distributions.
- Exclude notebooks, standalone scripts, experiments, and CI workflows
  from the backend source distribution.

Earlier releases are available in the
[ATLAS tag history](https://github.com/CCI-GU-Sweden/ATLAS/tags).
