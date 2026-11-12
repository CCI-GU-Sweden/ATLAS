import pytest
import xmltodict
import numpy as np
import tifffile

from atlas.io import get_metadata_for_stitched_tif
from atlas.io.fibics_metadata import get_fibics_tif_scan_pixel_size, get_fibics_mif_pixel_size


def test_source_pixel_sizes_preserve_units_and_vector_directions(tmp_path):
    mif = tmp_path / "section.ve-mif"
    mif.write_text(xmltodict.unparse({"MosaicInfo": {"PixelSize": {"@unit": "nm", "#text": "10.269"}}}))
    tif = tmp_path / "tile.tif"
    xml = xmltodict.unparse({"Fibics": {"Scan": {"Ux": "0.006", "Uy": "0.008", "Vx": "0", "Vy": "-0.012"}}})
    tifffile.imwrite(tif, np.zeros((2, 2), dtype=np.uint8), extratags=[(51023, "s", 0, xml, False)])
    assert get_fibics_mif_pixel_size(mif) == {"pixel_size": 10.269, "unit": "nm"}
    scan = get_fibics_tif_scan_pixel_size(tif)
    assert scan["u_vector"] == [.006, .008]
    assert scan["v_vector"] == [0, -.012]
    assert scan["pixel_size_x"] == pytest.approx(.01)
    assert scan["pixel_size_y"] == pytest.approx(.012)
    assert scan["unit"] == "µm"


@pytest.mark.parametrize(
    "detector,items,expected_gain",
    [
        ("SE2", {"@name": "Collector", "#text": "300 V"}, None),
        ("Sense BSD", {"@name": "BSD Gain", "#text": "High"}, "High"),
        (
            "Sense BSD",
            [
                {"@name": "BSD Q1", "#text": "Normal"},
                {"@name": "BSD Gain", "#text": "Very High"},
            ],
            "Very High",
        ),
    ],
)
def test_stitched_metadata_single_and_multiple_detector_items(
    tmp_path, detector, items, expected_gain
):
    """Single detector settings must parse like repeated XML items."""
    tile = {"Filename": "tile.tif", "WD": "0.006", "Brightness": "48", "Contrast": "61"}
    xml = xmltodict.unparse({"Fibics": {"Scan": {"Ux": "0.01", "Uy": "0", "Vx": "0", "Vy": "-0.012"}}})
    tifffile.imwrite(tmp_path / "tile.tif", np.ones((2, 2), dtype=np.uint8), extratags=[(51023, "s", 0, xml, False)])
    xml_other = xmltodict.unparse({"Fibics": {"Scan": {"Ux": "0.02", "Uy": "0", "Vx": "0", "Vy": "-0.02"}}})
    tifffile.imwrite(tmp_path / "tile_other.tif", np.ones((2, 2), dtype=np.uint8), extratags=[(51023, "s", 0, xml_other, False)])
    mosaic = {
        "Date": "2026-10-01T12:00:00",
        "DateFinished": "2026-10-01T12:01:00",
        "SetupInfo": {
            "SetupFilename": "test",
            "Beam": {
                "Aperture": "2 kV 300 pA", "AccV": "2000", "BeamI": "300"
            },
            "Signal": {"Detector": detector, "DetectorInfo": {"item": items}},
        },
        "TileInfo": {
            "NumTilesX": "2", "NumTilesY": "2",
            "AutoTune": {"AutoFocusFrequency": "3", "AutoStigFrequency": "4"},
        },
        "Tiles": {"Tile": [tile.copy(), {**tile, "Filename": "tile_other.tif"}, tile.copy(), tile.copy()]},
        "PixelSize": {"@unit": "nm", "#text": "10"},
        "DwellTime": {"@unit": "µs", "#text": "6.4"},
        "LineAveraging": "1",
    }
    path = tmp_path / "section.ve-mif"
    path.write_text(xmltodict.unparse({"MosaicInfo": mosaic}))

    result = get_metadata_for_stitched_tif(path)

    assert result["Detector"] == detector
    assert result["BSD Gain"] == expected_gain
    assert result["Tile arrangement"] == "R2C2"
    assert result["Working distance"] == 6
    assert result["Brightness"] == 48
    assert result["Contrast"] == 61
    assert result["Pixel size"] == 10
    assert result["Pixel size unit"] == "nm"
    assert result["Pixel size MIF"] == 10
    assert result["Pixel size MIF unit"] == "nm"
    assert result["Pixel size tile TIFF X"] == pytest.approx(.01)
    assert result["Pixel size tile TIFF Y"] == pytest.approx(.012)
    assert result["Pixel size tile TIFF X unit"] == result["Pixel size tile TIFF Y unit"] == "µm"
    assert len(result["Pixel size tile TIFF per tile"]) == 4
    assert result["Pixel size tile TIFF per tile"][1]["pixel_size_x"] == pytest.approx(.02)
