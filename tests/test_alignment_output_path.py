import json

import numpy as np
import pandas as pd
import pytest
import tifffile
from skimage.transform import downscale_local_mean

from atlas.alignment import ROI
from atlas.io import apply_alignment
from atlas.image_analysis import rescale_image_intensity


def test_custom_alignment_output_matches_default_and_preserves_it(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    image_path = source / "stitched_image_S_1.tiff"
    image = np.arange(64, dtype=np.uint16).reshape(8, 8)
    tifffile.imwrite(image_path, image, metadata={"axes": "YX", "stitching_metadata": {"tile": 1}})
    roi = ROI(x0=0, x1=8, y0=0, y1=8)
    frame = pd.DataFrame([{
        "moving_path": image_path, "reference_path": image_path,
        "cumulative_ROI": roi, "moving_ROI": roi, "down_scale": 1,
    }])
    kwargs = dict(buffer_pixels=0, percentile_low=1, percentile_high=99, use_down_sample=True)

    baseline, baseline_path = apply_alignment(frame, **kwargs)
    baseline_pixels = baseline[:]
    baseline_metadata = dict(baseline.attrs)
    baseline_json = json.loads(baseline_path.with_suffix(".json").read_text())

    output_dir = tmp_path / "zyx-validation"
    output_dir.mkdir()
    requested_path = output_dir / "aligned.zarr"
    result, result_path = apply_alignment(frame, output_path=requested_path, **kwargs)

    assert baseline_path == source / "source.zarr"
    assert result_path == requested_path
    np.testing.assert_array_equal(result[:], baseline_pixels)
    np.testing.assert_array_equal(baseline[:], baseline_pixels)
    assert dict(result.attrs) == baseline_metadata == dict(baseline.attrs)
    assert json.loads(result_path.with_suffix(".json").read_text()) == baseline_json
    assert baseline_metadata["AtlasMetadata"]["axes"] == "ZYX"
    rerun, _ = apply_alignment(frame, output_path=requested_path, **kwargs)
    np.testing.assert_array_equal(rerun[:], baseline_pixels)
    np.testing.assert_array_equal(baseline[:], baseline_pixels)


def test_custom_output_cannot_replace_input_or_unrelated_directory(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    image_path = source / "section.tiff"
    tifffile.imwrite(image_path, np.arange(64, dtype=np.uint16).reshape(8, 8),
                     metadata={"axes": "YX", "stitching_metadata": {}})
    roi = ROI(x0=0, x1=8, y0=0, y1=8)
    frame = pd.DataFrame([{"moving_path": image_path, "reference_path": image_path,
                           "cumulative_ROI": roi, "moving_ROI": roi, "down_scale": 1}])
    unrelated = tmp_path / "other"
    unrelated.mkdir()
    sentinel = unrelated / "important.txt"
    sentinel.write_text("keep")

    with pytest.raises(ValueError, match="contains an input TIFF"):
        apply_alignment(frame, output_path=source)
    assert image_path.is_file()
    with pytest.raises(ValueError, match="not a Zarr array"):
        apply_alignment(frame, output_path=unrelated)
    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize("downsample", [True, False])
def test_zyx_planes_keep_canvas_and_shift_placement(tmp_path, downsample):
    images = [
        np.arange(48, dtype=np.uint16).reshape(6, 8),
        np.arange(24, dtype=np.uint16).reshape(4, 6) * 2,
    ]
    paths = []
    for index, image in enumerate(images):
        path = tmp_path / f"section_{index}.tiff"
        tifffile.imwrite(path, image, metadata={"axes": "YX", "stitching_metadata": {"section": index}})
        paths.append(path)

    first_roi = ROI(x0=0, x1=4, y0=0, y1=3)
    second_roi = ROI(x0=0, x1=3, y0=0, y1=2)
    shifted_roi = ROI(x0=-1, x1=2, y0=1, y1=3)
    frame = pd.DataFrame([
        {"moving_path": paths[0], "reference_path": paths[0], "moving_ROI": first_roi,
         "cumulative_ROI": first_roi, "down_scale": 2},
        {"moving_path": paths[1], "reference_path": paths[0], "moving_ROI": second_roi,
         "cumulative_ROI": shifted_roi, "down_scale": 2},
    ])
    result, _ = apply_alignment(frame, buffer_pixels=0, percentile_low=1,
                                percentile_high=99, use_down_sample=downsample,
                                output_path=tmp_path / "result.zarr")

    factor = 1 if downsample else 2
    expected_yxz = np.zeros((3 * factor, 5 * factor, 2), dtype=np.uint16)
    for index, image in enumerate(images):
        if downsample:
            image = downscale_local_mean(image, (2, 2)).astype(image.dtype)
        rescaled, _, _ = rescale_image_intensity(image, percentile_low=1, percentile_high=99)
        if index == 0:
            expected_yxz[0:3*factor, factor:5*factor, 0] = rescaled
        else:
            expected_yxz[factor:3*factor, 0:3*factor, 1] = rescaled

    assert result.shape == (2, 3 * factor, 5 * factor)
    assert result.chunks == (1, 1024, 1024)
    assert result.dtype == np.dtype("uint16")
    for index in range(2):
        np.testing.assert_array_equal(result[index, :, :], expected_yxz[:, :, index])
    metadata = result.attrs["AtlasMetadata"]
    assert metadata["axes"] == "ZYX"
    assert metadata["canvas_origin"] == [-factor, 0]
    assert metadata["images"]["1"]["stitching_metadata"] == {"section": 1}
