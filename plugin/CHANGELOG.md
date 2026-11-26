# Changelog

Notable changes are recorded here. Entries for version 1.1.9 and earlier were
reconstructed from Git history and summarize the main changes rather than
every fix. Dates are the dates recorded on the Git tags.

## 1.2.0rc1

The plugin is now developed in the ATLAS repository under `plugin/`.
Its version is independent of the backend's Git tags. This release candidate
is intended for testing before the stable 1.2.0 release.

### Added

- View completed stitched sections from watcher output and use watcher projects
  in the alignment and correction workflow.
- Stitch and align sections through a master Zarr store, including manual
  correction and full-resolution output.

### Changed

- Read export calibration from the master Zarr store.
- Improve installation and workflow documentation.
- Run CI tests in a conda environment with PyQt6.

### Fixed

- Bound memory use during OME-Zarr export and improve WebKnossos uploads.
- Handle explicit axis order in aligned Zarr output.

## 1.1.9 - 2026-11-11

- Fix compatibility with stitching backend return values and include stitching
  metadata in TIFF files for alignment.

## 1.1.8 - 2026-10-22

- Add support for downloading WebKnossos data and exporting it to tiled CZI.

## 1.1.7 - 2026-10-22

- Add multiple-point correction options for alignment.

## 1.1.6 - 2026-10-22

- Expose percentile options and save settings locally.

## 1.1.5 - 2026-10-21

- Update the WebKnossos integration and address temporary folder issues.

## 1.1.4 - 2026-08-12

- Replace invalid characters in WebKnossos dataset names with underscores.

## 1.1.3 - 2026-08-11

- Add explicit dataset naming and validate names before WebKnossos upload.

## 1.1.2 - 2026-08-11

- Adjust the interface for napari 0.8.

## 1.1.1 - 2026-08-11

- Update Zarr and OME-Zarr handling and fix CZI export and upload issues.
- Validate Python 3.12 compatibility.

## 1.1.0 - 2026-08-04

- Fix multiscale handling in the widget.

## 1.0.3 - 2026-08-03

- Update the `atlas-cci-align` dependency.

## 1.0.2 - 2026-08-03

- Fix multiscale reading when creating OME-Zarr output.

## 1.0.1 - 2026-08-03

- Update Python and dependencies, and run CI with Python 3.11.

## 1.0.0 - 2026-07-24

- Prepare the package for pip installation.

## 0.0.1 - 2026-07-24

- Initial tagged version, with stitching, alignment, correction, CZI export and
  WebKnossos upload functionality.
