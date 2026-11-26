from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from napari_atlas_cci import AtlasCCIWidget


@pytest.mark.parametrize("fail", [False, True])
def test_correction_stages_preview_before_saving(tmp_path, monkeypatch, fail):
    import atlas.alignment

    results = tmp_path / "alignment_results"
    results.mkdir()
    alignment = results / "z_alignment_results.pkl"
    pd.DataFrame({"store_path": ["master"], "shift": [0]}).to_pickle(alignment)
    preview = tmp_path / "aligned_preview.zarr"
    preview.mkdir()
    (preview / "old").touch()
    ome = tmp_path / "preview.ome.zarr"
    ome.mkdir()
    (ome / "old").touch()
    (tmp_path / "aligned_preview.json").write_text("old")
    def correct(frame, fixed, moving):
        frame["shift"] += 1
        return frame
    monkeypatch.setattr(atlas.alignment, "correct_z_alignment_from_points", correct)
    def build(root, *, alignment_path, preview_path, ome_path):
        assert pd.read_pickle(alignment_path)["shift"].iloc[0] == 1
        assert pd.read_pickle(alignment)["shift"].iloc[0] == 0
        if fail:
            return False, "failed", None
        preview_path.mkdir()
        (preview_path / "new").touch()
        preview_path.with_suffix(".json").write_text("new")
        ome_path.mkdir()
        (ome_path / "new").touch()
        return True, "", {"ome_zarr_path": str(ome_path)}
    widget = SimpleNamespace(_preview_ome_zarr_path=lambda root: ome, _apply_z_shifts_preview_worker=build)
    points = np.array([[0, 1, 2]])
    ok, _, payload = AtlasCCIWidget._apply_correction_worker(widget, tmp_path, points, points)
    assert ok is not fail
    assert pd.read_pickle(alignment)["shift"].iloc[0] == (0 if fail else 1)
    assert (preview / ("old" if fail else "new")).exists()
    assert (ome / ("old" if fail else "new")).exists()
    assert pd.read_pickle(results / "z_alignment_results_before_manual_correction.pkl")["shift"].iloc[0] == 0
    if not fail:
        assert payload["ome_zarr_path"] == str(ome)
