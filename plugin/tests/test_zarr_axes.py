import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from napari_atlas_cci import AtlasCCIWidget
from napari_atlas_cci._zarr_axes import normalize_aligned_array_to_zyx


@pytest.mark.parametrize("axes", ["ZYX", "YXZ"])
def test_normalize_uses_declared_axes_with_non_square_xy_and_deep_z(axes):
    expected = np.arange(5 * 3 * 2, dtype=np.uint16).reshape(5, 3, 2)
    stored = expected if axes == "ZYX" else expected.transpose(1, 2, 0)
    metadata = {"axes": axes, "images": {"0": {"source": "section.tif"}}}

    normalized, normalized_metadata = normalize_aligned_array_to_zyx(
        stored, metadata
    )

    assert normalized.shape == (5, 3, 2)
    np.testing.assert_array_equal(normalized, expected)
    assert normalized_metadata["axes"] == "ZYX"
    assert normalized_metadata["images"] == metadata["images"]
    assert metadata["axes"] == axes


@pytest.mark.parametrize("metadata", [None, {}, {"axes": "XYZ"}])
def test_normalize_rejects_missing_or_unsupported_axes(metadata):
    with pytest.raises(ValueError, match="axes"):
        normalize_aligned_array_to_zyx(np.zeros((5, 3, 2)), metadata)


@pytest.mark.parametrize("axes", ["ZYX", "YXZ"])
def test_full_resolution_writer_uses_backend_output_path(
    tmp_path, monkeypatch, axes
):
    import dask.array as da
    import atlas.io

    root = tmp_path / "sample"
    results = root / "alignment_results"
    results.mkdir(parents=True)
    (results / "z_alignment_results.pkl").write_bytes(b"stub")
    image = np.arange(5 * 3 * 2, dtype=np.uint16).reshape(5, 3, 2)
    stored = image if axes == "ZYX" else image.transpose(1, 2, 0)
    metadata = {
        "axes": axes,
        "coordinate_convention": "row_vectors",
        "images": {"0": {"source": "section.tif"}},
    }
    normalized_metadata = {**metadata, "axes": "ZYX"}
    requested_paths = []

    def apply_alignment(
        frame,
        buffer_pixels,
        percentile_low,
        percentile_high,
        use_down_sample,
        *,
        output_path,
    ):
        requested_paths.append(Path(output_path))
        Path(output_path).mkdir()
        Path(output_path).with_suffix(".json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        class FakeZarrArray:
            ndim = 3
            shape = stored.shape
            dtype = stored.dtype
            chunks = stored.shape
            attrs = {"AtlasMetadata": metadata}

            def __getitem__(self, key):
                return stored[key]

        array = FakeZarrArray()
        return array, Path(output_path)

    monkeypatch.setattr(atlas.io, "apply_alignment", apply_alignment)
    monkeypatch.setattr(
        "pandas.read_pickle", lambda _: pd.DataFrame({"section": [0]})
    )
    zarr_copies = []

    def write_test_zarr(data, path, overwrite=True):
        zarr_copies.append((Path(path), data.compute()))
        Path(path).mkdir()

    monkeypatch.setattr(da, "to_zarr", write_test_zarr)
    monkeypatch.setattr(
        da, "from_zarr", lambda *args, **kwargs: da.from_array(image)
    )
    stored_metadata = []
    monkeypatch.setattr(
        "napari_atlas_cci._zarr_axes.write_atlas_metadata",
        lambda path, value: stored_metadata.append((Path(path), value)),
    )

    ome_inputs = []
    widget = SimpleNamespace(
        settings=SimpleNamespace(percentile_low=1.0, percentile_high=99.8),
        _aligned_zarr_path=lambda folder: folder / "sample_aligned.zarr",
        _aligned_ome_zarr_path=lambda folder: folder / "sample_aligned.ome.zarr",
        _legacy_alignment_zarr_path=lambda folder: folder / f"{folder.name}.zarr",
        _normalize_aligned_array_axes=normalize_aligned_array_to_zyx,
        _replace_directory_from_temp=MethodType(
            AtlasCCIWidget._replace_directory_from_temp, SimpleNamespace()
        ),
        _write_ome_zarr_pyramid_from_array=lambda data, path, name, atlas_metadata: (
            ome_inputs.append((data, path, name, atlas_metadata))
            or (True, "", {"ome_zarr_path": str(path)})
        ),
    )

    success, message, payload = AtlasCCIWidget._write_aligned_zarr_worker(
        widget, root
    )

    assert success, message
    assert payload["aligned_zarr_path"] == str(root / "sample_aligned.zarr")
    assert requested_paths == [root / "sample_aligned.zarr.tmp"]
    if axes == "ZYX":
        assert not zarr_copies
        assert not stored_metadata
    else:
        assert len(zarr_copies) == 1
        assert zarr_copies[0][0].name.endswith(".tmp.zyx")
        np.testing.assert_array_equal(zarr_copies[0][1], image)
        assert stored_metadata[0][1]["axes"] == "ZYX"
    assert (root / "sample_aligned.zarr").is_dir()
    assert not (root / "sample_aligned.zarr.json").exists()
    assert json.loads(
        (root / "sample_aligned.json").read_text(encoding="utf-8")
    ) == normalized_metadata
    np.testing.assert_array_equal(ome_inputs[0][0].compute(), image)
    assert ome_inputs[0][3] == normalized_metadata


def test_full_resolution_writer_falls_back_without_output_path(
    tmp_path, monkeypatch
):
    import dask.array as da
    import atlas.io

    root = tmp_path / "sample"
    results = root / "alignment_results"
    results.mkdir(parents=True)
    (results / "z_alignment_results.pkl").write_bytes(b"stub")
    image = np.arange(5 * 3 * 2, dtype=np.uint16).reshape(5, 3, 2)
    stored = image.transpose(1, 2, 0)
    metadata = {"axes": "YXZ", "images": {"0": {"source": "section.tif"}}}

    def apply_alignment(
        frame, buffer_pixels, percentile_low, percentile_high, use_down_sample
    ):
        class FakeZarrArray:
            ndim = 3
            shape = stored.shape
            dtype = stored.dtype
            chunks = stored.shape
            attrs = {"AtlasMetadata": metadata}

            def __getitem__(self, key):
                return stored[key]

        return FakeZarrArray(), root / "sample.zarr"

    monkeypatch.setattr(atlas.io, "apply_alignment", apply_alignment)
    monkeypatch.setattr(
        "pandas.read_pickle", lambda _: pd.DataFrame({"section": [0]})
    )
    writes = []

    def write_test_zarr(data, path, overwrite=True):
        writes.append((Path(path), data.compute()))
        Path(path).mkdir()

    monkeypatch.setattr(da, "to_zarr", write_test_zarr)
    monkeypatch.setattr(
        da, "from_zarr", lambda *args, **kwargs: da.from_array(image)
    )
    normalized_metadata = {**metadata, "axes": "ZYX"}
    stored_metadata = []
    monkeypatch.setattr(
        "napari_atlas_cci._zarr_axes.write_atlas_metadata",
        lambda path, value: stored_metadata.append((Path(path), value)),
    )

    widget = SimpleNamespace(
        settings=SimpleNamespace(percentile_low=1.0, percentile_high=99.8),
        _aligned_zarr_path=lambda folder: folder / "sample_aligned.zarr",
        _aligned_ome_zarr_path=lambda folder: folder / "sample_aligned.ome.zarr",
        _legacy_alignment_zarr_path=lambda folder: folder / f"{folder.name}.zarr",
        _normalize_aligned_array_axes=normalize_aligned_array_to_zyx,
        _replace_directory_from_temp=MethodType(
            AtlasCCIWidget._replace_directory_from_temp, SimpleNamespace()
        ),
        _write_ome_zarr_pyramid_from_array=lambda *args: (
            True, "", {"ome_zarr_path": str(args[1])}
        ),
    )

    success, message, _ = AtlasCCIWidget._write_aligned_zarr_worker(
        widget, root
    )

    assert success, message
    assert len(writes) == 1
    assert writes[0][0] == root / "sample_aligned.zarr.tmp"
    np.testing.assert_array_equal(writes[0][1], image)
    assert stored_metadata[0][1] == normalized_metadata
    assert json.loads(
        (root / "sample_aligned.json").read_text(encoding="utf-8")
    ) == normalized_metadata
