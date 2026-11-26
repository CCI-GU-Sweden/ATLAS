from types import SimpleNamespace

import numpy as np
import pandas as pd

from napari_atlas_cci import AtlasCCIWidget


def test_initial_alignment_uses_store_order_and_metadata(tmp_path, monkeypatch):
    import atlas.io
    import atlas.alignment

    (tmp_path / "stitched_sections.zarr").mkdir()
    order = [{"atlas_section_number": 6, "z_index": 0}, {"atlas_section_number": 2, "z_index": 1}]
    store = SimpleNamespace(
        validate_section_pixel_sizes=lambda: {"Acquisition Pixel size": {"first_um": 0.012}},
        ordered_sections=lambda order_path: order,
    )
    monkeypatch.setattr(atlas.io.StitchedZarrStore, "open", lambda path: store)
    calls = []
    def initialize(actual_store, actual_order, factor):
        calls.append((actual_store, actual_order, factor))
        return pd.DataFrame({"moving_section_number": [6, 2]})
    monkeypatch.setattr(atlas.alignment, "initialize_zarr_alignment_df", initialize)
    monkeypatch.setattr(atlas.alignment, "pairwise_zarr_alignment", lambda frame, store: frame)
    monkeypatch.setattr(atlas.alignment, "calculate_cumulative_shifts", lambda frame: frame)
    widget = SimpleNamespace(settings=SimpleNamespace(downsampling_factor=10))
    ok, _, sizes = AtlasCCIWidget._prepare_alignement_worker(widget, tmp_path, 70, "nm")
    assert ok and sizes["Value"] == 0.012
    assert calls == [(store, order, 10)]
    assert pd.read_pickle(tmp_path / "alignment_results/z_alignment_results.pkl")["moving_section_number"].tolist() == [6, 2]


def test_preview_uses_zarr_writer_and_passes_metadata(tmp_path, monkeypatch):
    import atlas.io

    (tmp_path / "stitched_sections.zarr").mkdir()
    results = tmp_path / "alignment_results"
    results.mkdir()
    pd.DataFrame({"store_path": [str(tmp_path / "stitched_sections.zarr")]}).to_pickle(results / "z_alignment_results.pkl")
    store = object()
    monkeypatch.setattr(atlas.io.StitchedZarrStore, "open", lambda path: store)
    metadata = {"axes": "ZYX", "images": {"0": {"atlas_section_number": 6}}}
    data = np.zeros((2, 3, 4), dtype=np.uint8)
    captured = {}
    def apply(frame, actual_store, **kwargs):
        captured.update(kwargs)
        assert actual_store is store
        return data, kwargs["output_path"]
    monkeypatch.setattr(atlas.io, "apply_alignment_zarr", apply)
    widget = SimpleNamespace(
        settings=SimpleNamespace(percentile_low=1, percentile_high=99),
        _normalize_aligned_array_axes=lambda array: (array, metadata),
        _preview_ome_zarr_path=lambda root: root / "preview.ome.zarr",
        _legacy_alignment_zarr_path=lambda root: root / f"{root.name}.zarr",
        _write_ome_zarr_pyramid_from_array=lambda array, path, name, meta: (True, "", {"metadata": meta}),
    )
    ok, _, payload = AtlasCCIWidget._apply_z_shifts_preview_worker(widget, tmp_path)
    assert ok and payload["metadata"] == metadata
    assert captured["output_path"] == tmp_path / "aligned_preview.zarr"
    assert captured["use_down_sample"] is True


def test_final_alignment_uses_latest_dataframe_and_direct_output(tmp_path, monkeypatch):
    import atlas.io
    import dask.array as da
    from types import MethodType
    from napari_atlas_cci._zarr_axes import normalize_aligned_array_to_zyx

    (tmp_path / "stitched_sections.zarr").mkdir()
    results = tmp_path / "alignment_results"
    results.mkdir()
    pd.DataFrame({"store_path": ["master"], "shift": [42]}).to_pickle(results / "z_alignment_results.pkl")
    metadata = {"axes": "ZYX", "images": {"0": {"atlas_section_number": 6}}}
    class Array:
        ndim = 3
        shape = (1, 3, 4)
        attrs = {"AtlasMetadata": metadata}
    store = object()
    monkeypatch.setattr(atlas.io.StitchedZarrStore, "open", lambda path: store)
    def apply(frame, actual_store, **kwargs):
        assert frame["shift"].iloc[0] == 42
        assert actual_store is store
        assert kwargs["use_down_sample"] is False
        kwargs["output_path"].mkdir()
        kwargs["output_path"].with_suffix(".json").write_text("temporary")
        return Array(), kwargs["output_path"]
    monkeypatch.setattr(atlas.io, "apply_alignment_zarr", apply)
    monkeypatch.setattr(da, "from_zarr", lambda path: da.zeros((1, 3, 4)))
    widget = SimpleNamespace(
        settings=SimpleNamespace(percentile_low=1, percentile_high=99),
        _aligned_zarr_path=lambda root: root / "final.zarr",
        _aligned_ome_zarr_path=lambda root: root / "final.ome.zarr",
        _legacy_alignment_zarr_path=lambda root: root / f"{root.name}.zarr",
        _normalize_aligned_array_axes=normalize_aligned_array_to_zyx,
        _replace_directory_from_temp=MethodType(AtlasCCIWidget._replace_directory_from_temp, SimpleNamespace()),
        _write_ome_zarr_pyramid_from_array=lambda data, path, name, meta: (True, "", {"metadata": meta}),
    )
    ok, message, payload = AtlasCCIWidget._write_aligned_zarr_worker(widget, tmp_path)
    assert ok, message
    assert payload["metadata"] == metadata
    assert (tmp_path / "final.zarr").is_dir()
    assert (tmp_path / "final.json").is_file()
    assert not (tmp_path / "final.zarr.json").exists()
    assert (tmp_path / "stitched_sections.zarr").is_dir()
