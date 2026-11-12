import pytest
import numpy as np
import tifffile

from atlas.io.fibics_metadata import get_stitched_pixel_size_microns


@pytest.mark.parametrize("value,unit,expected", [(15, "nm", .015), (.015, "µm", .015), (.015, "um", .015), (15e-9, "m", .015)])
def test_stitched_tiff_pixel_size_units(tmp_path, value, unit, expected):
    path = tmp_path / "section.tiff"
    tifffile.imwrite(path, np.ones((3, 4), dtype=np.uint8), metadata={
        "axes": "YX", "AcquisitionMetadata": {"Pixel size": value, "Pixel size unit": unit}
    })
    assert get_stitched_pixel_size_microns(path) == pytest.approx(expected)


def test_stitched_tiff_pixel_size_missing(tmp_path):
    path = tmp_path / "section.tiff"
    tifffile.imwrite(path, np.ones((3, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="pixel-size metadata"):
        get_stitched_pixel_size_microns(path)
