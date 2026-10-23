from contextlib import contextmanager
import json
from types import SimpleNamespace

from cluster_tools import get_executor
import numpy as np
import pytest

wk = pytest.importorskip("webknossos")

import transfer_wk_annotation_region as transfer


def test_box_selection_requires_choice_for_multiple_boxes():
    first = wk.BoundingBox((32, 64, 8), (64, 96, 16))
    second = wk.BoundingBox((0, 0, 0), (32, 32, 8))
    annotation = SimpleNamespace(user_bounding_boxes=[first, second])
    with pytest.raises(ValueError, match="BOUNDING_BOX_INDEX"):
        transfer.select_bounding_box(annotation)
    assert transfer.select_bounding_box(annotation, 1) == second
    with pytest.raises(ValueError, match="no user-defined"):
        transfer.select_bounding_box(SimpleNamespace(user_bounding_boxes=[]))


@pytest.mark.parametrize("name_override", [None, "custom_sample"])
@pytest.mark.parametrize("source_mag", ["16-16-4", "32-32-8"])
def test_transfer_download_reuse_and_target_context(tmp_path, monkeypatch, name_override, source_mag):
    """Exercise a real coarse local dataset, replacing only network calls."""
    bbox = wk.BoundingBox((32, 64, 8), (64, 96, 16))
    mag = wk.Mag(source_mag)
    annotation = SimpleNamespace(user_bounding_boxes=[bbox])
    calls = []
    active = []
    remote = SimpleNamespace(
        name="fine-aligned", dataset_id="source-id",
        layers={"color": SimpleNamespace(
            category=wk.COLOR_CATEGORY, mags={mag: None}, coordinate_transformations=()
        )},
    )

    @contextmanager
    def context(*, token, url):
        active.append(token)
        try:
            yield
        finally:
            active.pop()

    def download(**kwargs):
        assert active == ["source-test-token"]
        assert kwargs["bounding_box"] == bbox
        assert kwargs["mags"] == [mag]
        assert kwargs["layers"] == ["color"]
        calls.append("download")
        dataset = wk.Dataset(kwargs["path"], voxel_size=(4, 4, 30))
        layer = dataset.add_layer("color", wk.COLOR_CATEGORY, dtype="uint8")
        layer.bounding_box = bbox
        view = layer.add_mag(mag)
        shape = tuple(size // factor for size, factor in zip(bbox.size, mag.to_vec3_int()))
        view.write_cxyz(np.ones((1, *shape), dtype=np.uint8), absolute_offset=bbox.topleft)
        return dataset

    def upload(dataset, *, new_dataset_name):
        assert active == ["target-test-token"]
        assert new_dataset_name == (name_override or f"fine-aligned_mag{source_mag}")
        assert mag in dataset.get_layer("color").mags
        assert any(m.max_dim > mag.max_dim for m in dataset.get_layer("color").mags)
        assert wk.Mag(1) not in dataset.get_layer("color").mags
        for level in dataset.get_layer("color").mags:
            assert all(a >= b for a, b in zip(level.to_vec3_int(), mag.to_vec3_int()))
        assert tuple(dataset.voxel_size) == (4, 4, 30)
        calls.append("upload")
        return SimpleNamespace(url="https://example.org/datasets/copied")

    remote.download = download
    annotation.get_remote_base_dataset = lambda **kwargs: remote
    monkeypatch.setattr(transfer, "SOURCE_TOKEN", "source-test-token")
    monkeypatch.setattr(transfer, "MAG", source_mag)
    monkeypatch.setattr(transfer, "TARGET_DATASET_NAME", name_override)
    monkeypatch.setattr(transfer, "TARGET_TOKEN", "target-test-token")
    monkeypatch.setattr(transfer, "OUTPUT_DIR", tmp_path / "download")
    monkeypatch.setattr(transfer, "get_executor", lambda *args, **kwargs: get_executor("sequential"))
    monkeypatch.setattr(wk, "webknossos_context", context)
    monkeypatch.setattr(wk.Annotation, "download", lambda *args, **kwargs: annotation)
    monkeypatch.setattr(wk.User, "get_current_user", lambda: SimpleNamespace(organization_id="target-org"))
    monkeypatch.setattr(wk.Dataset, "upload", upload)

    transfer.main(do_upload=False)
    assert calls == ["download"]
    transfer.main(do_upload=True)
    assert calls == ["download", "upload"]
    # Simulate a download/upload made by the old script: only the requested
    # resolution, with no pyramid or uploaded-magnification completion record.
    path = tmp_path / "download"
    dataset = wk.Dataset.open(path)
    for other_mag in list(dataset.get_layer("color").mags):
        if other_mag != mag:
            dataset.get_layer("color").delete_mag(other_mag)
    marker = path / ".atlas-transfer.json"
    record = json.loads(marker.read_text())
    record.pop("pyramid")
    record["upload"].pop("magnifications")
    marker.write_text(json.dumps(record))
    transfer.main(do_upload=True)
    assert calls == ["download", "upload", "upload"]
    transfer.main(do_upload=True)
    assert calls == ["download", "upload", "upload"]
    monkeypatch.setattr(transfer, "MAG", "8-8-2")
    remote.layers["color"].mags[wk.Mag("8-8-2")] = None
    with pytest.raises(ValueError, match="different download"):
        transfer.main(do_upload=False)


def test_incomplete_download_is_not_reused(tmp_path):
    mag = wk.Mag("32-32-8")
    annotation = SimpleNamespace(user_bounding_boxes=[wk.BoundingBox((0, 0, 0), (64, 64, 16))])
    remote = SimpleNamespace(
        name="fine-aligned", dataset_id="source-id",
        layers={"color": SimpleNamespace(
            category=wk.COLOR_CATEGORY, mags={mag: None}, coordinate_transformations=()
        )},
    )
    with pytest.raises(ValueError, match="without a completed download"):
        transfer.download_subset(annotation, remote, tmp_path, mag)
    remote.layers["color"].coordinate_transformations = (object(),)
    with pytest.raises(ValueError, match="display transformations"):
        transfer.download_subset(annotation, remote, tmp_path / "new", mag)


def test_full_resolution_pyramid_and_interrupted_build(tmp_path, monkeypatch):
    """Repair partial coarse data from the intact full-resolution download."""
    monkeypatch.setattr(transfer, "PYRAMID_COARSEST_MAG", "4")
    monkeypatch.setattr(transfer, "get_executor", lambda *args, **kwargs: get_executor("sequential"))
    path = tmp_path / "dataset"
    dataset = wk.Dataset(path, voxel_size=(4, 4, 30))
    layer = dataset.add_layer("color", wk.COLOR_CATEGORY, dtype="uint8")
    data = np.full((1, 16, 16, 8), 100, dtype=np.uint8)
    data[:, 8:, :, :] = 200
    layer.bounding_box = wk.BoundingBox((0, 0, 0), (16, 16, 8))
    layer.add_mag(1).write_cxyz(data)
    marker = path / ".atlas-transfer.json"
    marker.write_text(json.dumps({"download": {"mag": "1"}}))
    # An earlier process could have created a target mag before failing.
    layer.add_mag("2-2-1")
    original_downsample = wk.Layer.downsample
    attempts = []

    def interrupted(*args, **kwargs):
        attempts.append(True)
        raise RuntimeError("interrupted build")

    monkeypatch.setattr(wk.Layer, "downsample", interrupted)
    with pytest.raises(RuntimeError, match="interrupted build"):
        transfer.ensure_pyramid(dataset, path)
    assert "pyramid" not in json.loads(marker.read_text())
    monkeypatch.setattr(wk.Layer, "downsample", original_downsample)
    transfer.ensure_pyramid(dataset, path)
    np.testing.assert_array_equal(layer.get_mag(1).read_cxyz(), data)
    coarse = layer.get_mag("4-4-1").read_cxyz()
    assert coarse.shape == (1, 4, 4, 8)
    assert np.all(coarse[:, :2] == 100)
    assert np.all(coarse[:, 2:] == 200)
    assert json.loads(marker.read_text())["pyramid"]["magnifications"] == transfer.magnification_signature(dataset)
    # Completed pyramids are reused without starting another downsampling job.
    monkeypatch.setattr(wk.Layer, "downsample", interrupted)
    transfer.ensure_pyramid(dataset, path)
    assert len(attempts) == 1
