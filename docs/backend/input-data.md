# Prepare the input data

The raw import notebook expects one section-set directory containing the ATLAS
section folders. Each section folder must be named `S_...` and contain a
completed `.ve-mif` acquisition file plus the TIFF tiles listed in that file.
The importer checks that the acquisition status is `completed`, that the tile
count matches the TIFF count, and that every TIFF named in the metadata exists.
Folders that fail these checks are reported and skipped.

```text
section-set/
├── S_1/
│   ├── acquisition.ve-mif
│   ├── tile_....tif
│   └── ...
├── S_2/
│   ├── acquisition.ve-mif
│   └── ...
└── correct_order.txt       # optional
```

Use the actual `.ve-mif` and TIFF names from your acquisition; the names above
are illustrative. Set `series_folder` in [Raw import](https://github.com/CCI-GU-Sweden/ATLAS/blob/main/backend/examples/ATLAS-raw-import.ipynb)
to this directory. The notebook discovers section folders by the `S_` prefix,
orders them numerically for initial storage, then stitches each valid folder.
By default it uses a 1 µm border for registration shifts and a 200 pixel
maximum tile shift. The stitched image is vertically flipped before storage to
match the orientation expected by this workflow.

## Section order

ATLAS section numbers (`S_1`, `S_2`, and so on) identify sections. The Z index
is the section's position in the master Zarr array. These are separate values:
missing section numbers are allowed, and `correct_order.txt` can define a
different alignment order or select a subset. It is a comma-separated sequence;
`...` expands an inclusive range:

```text
1,...,5,7,9,8,6
```

Without this file, alignment uses numeric section-number order. Every listed
section must have completed import. See [Outputs and alignment](outputs.md)
for how order and Z indices are used.

## Pixel calibration

The Z alignment notebook checks the recorded acquisition pixel size for every
completed section and stops if values are missing or inconsistent beyond its
1% default tolerance. It uses the first section's acquisition pixel size as
the in-plane value. Set `AXIAL_PIXEL_MICRONS` in that notebook to the physical
spacing between sections; the example value of 70 µm is only a starting
example and should be replaced with the value for the dataset. The alignment
workflow records these values when exporting CZI. Review the acquisition
metadata and calibration before interpreting physical dimensions.

For environment setup and the notebook sequence, see the
[quickstart](quickstart.md) and [installation guide](install.md).
