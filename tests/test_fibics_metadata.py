import pytest
import xmltodict

from atlas.io import get_metadata_for_stitched_tif


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
    tile = {"WD": "0.006", "Brightness": "48", "Contrast": "61"}
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
        "Tiles": {"Tile": [tile.copy() for _ in range(4)]},
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
