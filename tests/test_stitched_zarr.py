import numpy as np
import pandas as pd
import pytest
import tifffile
import zarr

from atlas.io import StitchedZarrStore


def test_stitched_sections_grow_and_preserve_tiff_and_metadata(tmp_path):
    path = tmp_path / "stitched_sections.zarr"
    store = StitchedZarrStore.open(path, chunk_yx=(4, 4))
    sections = [
        (7, np.arange(15, dtype=np.uint16).reshape(3, 5)),
        (1, np.full((7, 2), 44, dtype=np.uint16)),
        (3, np.full((2, 9), 77, dtype=np.uint16)),
    ]
    for expected_index, (number, original) in enumerate(sections):
        # The TIFF and Zarr receive the same exported orientation.
        saved_image = np.flipud(original)
        acquisition = {"atlas_section_number": number, "Pixel size": 10.2, "Pixel size unit": "nm"}
        stitching = {"crop_origin": [2, 4], "tiles": {"0": {"origin_in_stitched": [0, 1]}}}
        transforms = {"t00": [0.0, 0.0]}
        tile_dataframe = pd.DataFrame({"Filename": ["tile.tif"], "PixelSizeMicron": [0.01], "stitching_shifts": ["[0. 0.]"]})
        csv_path = tmp_path / f"section_{number}.csv"
        tile_dataframe.to_csv(csv_path, index=False)
        tif_path = tmp_path / f"section_{number}.tiff"
        tifffile.imwrite(tif_path, saved_image, metadata={"axes": "YX", "AcquisitionMetadata": acquisition})
        index = store.write_section(
            atlas_section_number=number, image=saved_image,
            acquisition_metadata=acquisition, stitching_metadata=stitching,
            transforms=transforms, source_folder=tmp_path / f"S_{number:03}",
            tile_dataframe=tile_dataframe,
        )
        assert index == expected_index
        for previous_number, previous in sections[:expected_index + 1]:
            np.testing.assert_array_equal(store.read_section(previous_number), np.flipud(previous))

    reopened = StitchedZarrStore.open(path)
    assert reopened.group.attrs["axes"] == ["z", "y", "x"]
    assert reopened.group["images"].shape == (3, 7, 9)
    assert reopened.group["images"].chunks == (1, 4, 4)
    compressor = reopened.group["images"].metadata.compressor.get_config()
    assert compressor["id"] == "blosc"
    assert compressor["cname"] == "lz4"
    assert compressor["clevel"] == 5
    for number, original in sections:
        record = dict(reopened.sections[f"S_{number}"].attrs)
        assert record["atlas_section_number"] == number
        assert record["shape_yx"] == list(original.shape)
        assert record["status"] == "complete"
        assert record["acquisition_metadata"]["Pixel size"] == 10.2
        assert record["stitching_metadata"]["crop_origin"] == [2, 4]
        assert record["transforms"] == {"t00": [0.0, 0.0]}
        assert record["tile_table_csv"] == (tmp_path / f"section_{number}.csv").read_text()
        np.testing.assert_array_equal(reopened.read_section(number), tifffile.imread(tmp_path / f"section_{number}.tiff"))
    np.testing.assert_array_equal(reopened.group["images"][0, 3:, :], 0)
    np.testing.assert_array_equal(reopened.group["images"][0, :3, 5:], 0)


def test_interrupted_section_is_retried_at_same_index(tmp_path, monkeypatch):
    path = tmp_path / "master.zarr"
    store = StitchedZarrStore.open(path, chunk_yx=(4, 4))
    image = np.full((6, 6), 42, dtype=np.uint16)
    kwargs = dict(atlas_section_number=9, image=image, acquisition_metadata={}, source_folder="S_009")
    original = zarr.Array.__setitem__

    def interrupted(array, selection, value):
        if array.path.endswith("images"):
            raise OSError("simulated interrupted write")
        return original(array, selection, value)

    monkeypatch.setattr(zarr.Array, "__setitem__", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        store.write_section(**kwargs)
    assert store.sections["S_9"].attrs["status"] == "pending"
    with pytest.raises(ValueError, match="not complete"):
        store.read_section(9)
    monkeypatch.setattr(zarr.Array, "__setitem__", original)

    reopened = StitchedZarrStore.open(path)
    with pytest.raises(ValueError, match="same section shape"):
        reopened.write_section(**{**kwargs, "source_folder": "different"})
    assert reopened.write_section(**kwargs) == 0
    np.testing.assert_array_equal(reopened.read_section(9), image)
    with pytest.raises(ValueError, match="already complete"):
        reopened.write_section(**kwargs)


def test_invalid_inputs_do_not_reserve_section(tmp_path):
    store = StitchedZarrStore.open(tmp_path / "master.zarr")
    image = np.ones((2, 2), dtype=np.uint16)
    with pytest.raises(ValueError, match="does not match"):
        store.write_section(atlas_section_number=2, image=image, acquisition_metadata={"atlas_section_number": 3})
    with pytest.raises(TypeError):
        store.write_section(atlas_section_number=2, image=image, acquisition_metadata={"bad": object()})
    assert list(store.sections.group_keys()) == []
