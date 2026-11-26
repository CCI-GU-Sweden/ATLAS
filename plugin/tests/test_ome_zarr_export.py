from types import SimpleNamespace

import dask.array as da
import numpy as np
import zarr

from napari_atlas_cci import AtlasCCIWidget


def test_ome_zarr_export_reads_lazy_chunks(tmp_path):
    image = np.arange(3 * 64 * 80, dtype=np.uint16).reshape(3, 64, 80)
    data = da.from_array(image, chunks=(1, 32, 40))
    widget = SimpleNamespace(
        _replace_directory_from_temp=lambda temp, final: temp.rename(final)
    )
    path = tmp_path / "aligned.ome.zarr"
    metadata = {"axes": "ZYX", "images": {"0": {"source": "section.tif"}}}
    success, message, payload = AtlasCCIWidget._write_ome_zarr_pyramid_from_array(
        widget, data, path, "aligned", metadata
    )
    assert success, message
    group = zarr.open_group(str(path), mode="r")
    np.testing.assert_array_equal(group[payload["dataset_paths"][0]][:], image)
    assert len(payload["dataset_paths"]) == 5
    assert payload["p_low"] < payload["p_high"]
    assert group.attrs["AtlasMetadata"] == metadata
