import json

import numpy as np
import pytest
import zarr
from pylibCZIrw import czi

from atlas.io import zarr_array_to_czi


def _make_store(path, data, metadata):
    array = zarr.open_array(str(path), mode="w", shape=data.shape,
                            dtype=data.dtype, zarr_format=2)
    array[:] = data
    if metadata is not None:
        array.attrs["AtlasMetadata"] = metadata
    return path


def test_czi_export_matches_for_zyx_and_legacy_yxz(tmp_path):
    volume = np.arange(3 * 4 * 7, dtype=np.uint16).reshape(3, 4, 7)
    pixel_size = {"Value": 0.012, "Axial": 0.3, "Unit": "µm"}
    output_paths = []
    for axes, data in (("ZYX", volume), ("YXZ", volume.transpose(1, 2, 0))):
        folder = tmp_path / axes
        folder.mkdir()
        path = _make_store(folder / "aligned.zarr", data, {"axes": axes, "images": {}})
        output_paths.append(zarr_array_to_czi(path, pixel_size, "_test"))

    for axes, path in zip(("ZYX", "YXZ"), output_paths):
        with czi.open_czi(str(path)) as reader:
            for z in range(volume.shape[0]):
                np.testing.assert_array_equal(reader.read(plane={"Z": z})[:, :, 0], volume[z])
            metadata = reader.metadata
            distances = metadata["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]
            scales = {item["@Id"]: float(item["Value"]) for item in distances}
            assert scales == pytest.approx({"X": 0.012e-6, "Y": 0.012e-6, "Z": 0.3e-6})
            assert json.loads(reader.custom_attributes_metadata["AtlasMetadata"]) == {
                "axes": axes, "images": {}
            }


@pytest.mark.parametrize("metadata", [None, {}, {"axes": "XYZ"}])
def test_czi_export_rejects_missing_or_unsupported_axes(tmp_path, metadata):
    path = _make_store(tmp_path / "aligned.zarr", np.zeros((2, 4, 7), dtype=np.uint16), metadata)
    with pytest.raises(ValueError, match="axes"):
        zarr_array_to_czi(path, {"Value": 0.01, "Axial": 0.3, "Unit": "µm"}, "_test")
    assert not list(tmp_path.glob("*.czi"))
