from concurrent.futures import Future
from queue import SimpleQueue
from types import SimpleNamespace

import numpy as np
import pandas as pd

from napari_atlas_cci import AtlasCCIWidget


def test_master_store_batch_writes_sections_and_retries_failure(tmp_path, monkeypatch):
    import atlas.io
    import atlas.stitching

    paths = [tmp_path / f"S_{n}" for n in (1, 2, 3)]
    for path in paths:
        path.mkdir()
        (path / "tiles.ve-mif").touch()

    monkeypatch.setattr(
        atlas.io, "get_metadata_for_stitched_tif",
        lambda mif: {"atlas_section_number": int(mif.parent.name[2:])},
    )
    failed = {2}
    def stitch(mif, **kwargs):
        number = int(mif.parent.name[2:])
        if number in failed:
            raise ValueError("stitch failed")
        image = np.array([[number, 0], [0, number + 1]], dtype=np.uint8)
        return image, pd.DataFrame({"tile": [number]}), {"t": np.array([number])}, {"section": number}

    monkeypatch.setattr(atlas.stitching, "stitch_ATLAS_tiles", stitch)
    widget = SimpleNamespace(settings=SimpleNamespace(max_shift_pixels=200), _stitch_events=SimpleQueue())
    widget._process_one_series = lambda *args: AtlasCCIWidget._process_one_series(widget, *args)
    results = {path: Future() for path in paths}
    AtlasCCIWidget._run_stitching_batch(widget, tmp_path, paths, 2, results)

    assert [results[path].result()[0] for path in paths] == [True, False, True]
    store = atlas.io.StitchedZarrStore.open(tmp_path / "stitched_sections.zarr")
    assert [store.sections[f"S_{n}"].attrs["z_index"] for n in (1, 2, 3)] == [0, 1, 2]
    np.testing.assert_array_equal(store.read_section(1), [[0, 2], [1, 0]])
    assert store.sections["S_1"].attrs["stitching_metadata"] == {"section": 1}
    assert store.sections["S_2"].attrs["last_error"] == "stitch failed"

    failed.clear()
    retry = {paths[1]: Future()}
    AtlasCCIWidget._run_stitching_batch(widget, tmp_path, [paths[1]], 1, retry)
    assert retry[paths[1]].result() == (True, "")
    store = atlas.io.StitchedZarrStore.open(tmp_path / "stitched_sections.zarr")
    assert store.sections["S_2"].attrs["z_index"] == 1
    assert store.sections["S_2"].attrs["status"] == "complete"
