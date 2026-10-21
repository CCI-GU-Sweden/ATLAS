import numpy as np
import pandas as pd
import pytest
import tifffile
import xmltodict
from shapely.geometry import box

import atlas.stitching as stitching


@pytest.mark.parametrize("buffer_microns", [0, 1])
def test_stage_stitching_from_metadata(tmp_path, buffer_microns):
    """Exercise TIFF metadata, scan rotation and acquisition order together."""
    tiles = []
    images = []
    for index in range(2):
        image = np.arange(20, dtype=np.uint16).reshape(4, 5) + index * 100
        filename = f"tile_{index}.tif"
        fibics_xml = xmltodict.unparse({
            "Fibics": {"Scan": {"Ux": 1}, "Image": {"Width": 5, "Height": 4}}
        })
        tifffile.imwrite(
            tmp_path / filename,
            image,
            extratags=[(51023, "s", 0, fibics_xml, False)],
        )
        images.append(image)
        tiles.append({
            "Filename": filename,
            "StageX": -7 * index,
            "StageY": 2 * index,
            "StartTime": f"2026-10-01T00:00:0{index}+00:00",
        })
    mif_path = tmp_path / "section.ve-mif"
    mif_path.write_text(xmltodict.unparse({
        "MosaicInfo": {
            "ReferenceInfo": {"Beam": {"ScanRot": 90}},
            "Tiles": {"Tile": tiles[::-1]},
        }
    }))

    image, tile_df, _ = stitching.stage_stitching(
        mif_path, buffer_microns=buffer_microns
    )
    # A -90 degree coordinate rotation maps (-7, 2) to (2, 7).
    b = buffer_microns
    expected = np.zeros((11 + b, 7 + b), dtype=np.uint16)
    expected[b:b + 4, b:b + 5] = np.flipud(images[0])
    expected[7 + b:11 + b, 2 + b:7 + b] = np.flipud(images[1])
    np.testing.assert_array_equal(image, expected)
    assert tile_df["Filename"].tolist() == ["tile_0.tif", "tile_1.tif"]


@pytest.mark.parametrize(
    "positions", [[(0, 0)], [(0, 0), (7, 3)], [(0, 0), (2, 0)]]
)
def test_stage_stitching_preserves_full_tiles(tmp_path, monkeypatch, positions):
    """Preserve invalid borders, gaps and overwrite order without matching."""
    mif_path = tmp_path / "section.ve-mif"
    mif_path.touch()
    tiles = []
    rows = []
    height, width = 4, 5
    for index, (x, y) in enumerate(positions):
        tile = np.zeros((height, width), dtype=np.uint16)
        tile[1:3, 1:4] = np.arange(6).reshape(2, 3) + index * 10 + 1
        tile[1, 1] = np.iinfo(tile.dtype).max
        filename = f"tile_{index}.tif"
        tifffile.imwrite(tmp_path / filename, tile)
        tiles.append(tile)
        rows.append(dict(
            Filename=filename, ImageWidth=width, ImageHeight=height,
            X0_pix=x, Y0_pix=y, geometry=box(x, y, x + width, y + height),
        ))

    # Metadata parsing is shared with the existing stitching workflow.
    # Use nonconsecutive labels to exercise acquisition-order index reset.
    df = pd.DataFrame(rows, index=np.arange(len(rows)) + 10)
    monkeypatch.setattr(stitching, "get_tiles_dataframe", lambda *a, **kw: df)

    def unexpected_matching(*args, **kwargs):
        pytest.fail("Stage stitching must not run image registration")

    monkeypatch.setattr(stitching, "phase_cross_correlation", unexpected_matching)
    monkeypatch.setattr(stitching, "match_tiles", unexpected_matching)
    monkeypatch.setattr(stitching, "minimum_spanning_tree", unexpected_matching)

    image, tile_df, transforms = stitching.stage_stitching(mif_path)
    expected = np.zeros(
        (max(y for x, y in positions) + height,
         max(x for x, y in positions) + width), dtype=np.uint16,
    )
    for tile, (x, y) in zip(tiles, positions):
        expected[y:y + height, x:x + width] = np.flipud(tile)
    np.testing.assert_array_equal(image, expected)
    assert list(tile_df.index) == list(range(len(positions)))
    for shift in transforms.values():
        np.testing.assert_array_equal(shift, [0, 0])


@pytest.mark.parametrize("compression", [None, "deflate"])
def test_stitched_tiff_round_trip(tmp_path, compression):
    image = np.zeros((256, 384), dtype=np.uint16)
    image[10:20, 30:40] = np.arange(100).reshape(10, 10)
    image[10, 30] = 65535
    metadata = {"Pixel size": 0.02, "Pixel size unit": "µm"}
    path = tmp_path / "stitched.tiff"
    stitching.save_stitched_image(path, image, metadata, compression=compression)
    restored, restored_metadata = stitching.load_stitched_image(path)
    np.testing.assert_array_equal(restored, np.flipud(image))
    assert restored_metadata == metadata
    with tifffile.TiffFile(path) as tif:
        assert tif.pages[0].compression.name == (
            "DEFLATE" if compression else "NONE"
        )
    if compression:
        assert path.stat().st_size < image.nbytes // 10
