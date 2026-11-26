"""Regression checks for alignment from a stitched Zarr store."""

import numpy as np
import pandas as pd
import pytest
import tifffile

from atlas.alignment import (
    calculate_cumulative_shifts,
    correct_z_alignment_from_points,
    initialize_zarr_alignment_df,
    pairwise_zarr_alignment,
)
from atlas.alignment.utils import align_image_pair
from atlas.io import StitchedZarrStore, apply_alignment, apply_alignment_zarr


@pytest.fixture
def stitched_inputs(tmp_path):
    store = StitchedZarrStore.open(tmp_path / "stitched.zarr", chunk_yx=(32, 32))
    rng = np.random.default_rng(12)
    first = np.zeros((96, 110), dtype=np.uint16)
    first[12:80, 15:90] = rng.integers(100, 30000, (68, 75), dtype=np.uint16)
    images = {4: first, 6: np.roll(first, (4, -6), axis=(0, 1))}
    paths = {}
    for number in (6, 4):  # Storage order deliberately differs from S order.
        stitching = {"tiles": {"0": {"tile_to_stitched": np.eye(4).tolist()}}}
        store.write_section(
            atlas_section_number=number, image=images[number],
            acquisition_metadata={"atlas_section_number": number, "Pixel size": 10, "Pixel size unit": "nm"},
            stitching_metadata=stitching, source_folder=f"S_{number}", transforms={},
        )
        paths[number] = tmp_path / f"stitched_S_{number}.tiff"
        tifffile.imwrite(paths[number], images[number], metadata={"stitching_metadata": stitching})
    return store, paths


def aligned_dataframe(store):
    order = store.ordered_sections()
    frame = initialize_zarr_alignment_df(store, order, down_scale=2)
    return calculate_cumulative_shifts(pairwise_zarr_alignment(frame, store))


def test_pairwise_alignment_and_manual_correction(stitched_inputs, tmp_path):
    store, paths = stitched_inputs
    frame = aligned_dataframe(store)
    assert list(frame["moving_section_number"]) == [4, 6]
    assert list(frame["moving_z_index"]) == [1, 0]
    reference_roi, moving_roi, shift = align_image_pair(paths[4], paths[6], 2)
    assert frame.at[1, "reference_ROI"] == reference_roi
    assert frame.at[1, "moving_ROI"] == moving_roi
    np.testing.assert_array_equal(frame.at[1, "current_shift"], shift)

    for _ in range(2):
        prior = frame.at[1, "current_shift"].copy()
        frame = correct_z_alignment_from_points(frame, [[0, 10, 10]], [[1, 9, 8]])
        path = tmp_path / "alignment.pkl"
        frame.to_pickle(path)
        frame = pd.read_pickle(path)
        np.testing.assert_array_equal(frame.at[1, "current_shift"], prior + [1, 2])


@pytest.mark.parametrize("preview", [True, False])
def test_zarr_writer_matches_tiff_pixels_and_transforms(stitched_inputs, tmp_path, preview):
    store, paths = stitched_inputs
    frame = aligned_dataframe(store)
    frame["moving_path"] = [paths[4], paths[6]]
    frame["reference_path"] = [paths[4], paths[4]]
    baseline, _ = apply_alignment(frame, buffer_pixels=0, use_down_sample=preview,
                                  output_path=tmp_path / "baseline.zarr")
    result, _ = apply_alignment_zarr(frame, store, use_down_sample=preview,
                                     output_path=tmp_path / "aligned.zarr")
    np.testing.assert_array_equal(result[:], baseline[:])
    old = baseline.attrs["AtlasMetadata"]
    new = result.attrs["AtlasMetadata"]
    for key in ("axes", "coordinate_convention", "canvas_origin", "use_down_sample"):
        assert new[key] == old[key]
    for index, number in enumerate((4, 6)):
        previous = old["images"][str(index)]
        actual = new["images"][str(index)]
        for key in ("stitched_to_aligned", "stitching_metadata", "down_scale"):
            assert actual[key] == previous[key]
        assert actual["atlas_section_number"] == number
        assert actual["source_z_index"] == 1 - index
        assert actual["acquisition_metadata"]["atlas_section_number"] == number

    with pytest.raises(ValueError, match="separate"):
        apply_alignment_zarr(frame, store, output_path=store.path)


def test_section_order_and_pixel_size_validation(stitched_inputs, tmp_path, capsys):
    store, _ = stitched_inputs
    summary = store.validate_section_pixel_sizes()
    assert summary["Acquisition Pixel size"]["first_um"] == pytest.approx(0.01)
    assert [record["atlas_section_number"] for record in store.ordered_sections()] == [4, 6]
    assert "Warning: gaps" in capsys.readouterr().out

    order_file = tmp_path / "correct_order.txt"
    order_file.write_text("6,4", encoding="utf-8")
    assert [record["atlas_section_number"] for record in store.ordered_sections(order_file)] == [6, 4]
    order_file.write_text("5", encoding="utf-8")
    with pytest.raises(ValueError, match="not complete or not present"):
        store.ordered_sections(order_file)

    attrs = store.sections["S_6"].attrs
    metadata = dict(attrs["acquisition_metadata"])
    metadata["Pixel size"] = 12
    attrs["acquisition_metadata"] = metadata
    with pytest.raises(ValueError, match="vary by more than"):
        store.validate_section_pixel_sizes()
