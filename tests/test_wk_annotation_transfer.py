from contextlib import contextmanager
from types import SimpleNamespace

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
def test_transfer_download_reuse_and_target_context(tmp_path, monkeypatch, name_override):
    """Exercise a real coarse local dataset, replacing only network calls."""
    bbox = wk.BoundingBox((32, 64, 8), (64, 96, 16))
    mag = wk.Mag("32-32-8")
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
        view.write_cxyz(np.ones((1, 2, 3, 2), dtype=np.uint8), absolute_offset=bbox.topleft)
        return dataset

    def upload(dataset, *, new_dataset_name):
        assert active == ["target-test-token"]
        assert new_dataset_name == (name_override or "fine-aligned_mag32-32-8")
        assert list(dataset.get_layer("color").mags) == [mag]
        assert tuple(dataset.voxel_size) == (4, 4, 30)
        calls.append("upload")
        return SimpleNamespace(url="https://example.org/datasets/copied")

    remote.download = download
    annotation.get_remote_base_dataset = lambda **kwargs: remote
    monkeypatch.setattr(transfer, "SOURCE_TOKEN", "source-test-token")
    monkeypatch.setattr(transfer, "TARGET_DATASET_NAME", name_override)
    monkeypatch.setattr(transfer, "TARGET_TOKEN", "target-test-token")
    monkeypatch.setattr(transfer, "OUTPUT_DIR", tmp_path / "download")
    monkeypatch.setattr(wk, "webknossos_context", context)
    monkeypatch.setattr(wk.Annotation, "download", lambda *args, **kwargs: annotation)
    monkeypatch.setattr(wk.User, "get_current_user", lambda: SimpleNamespace(organization_id="target-org"))
    monkeypatch.setattr(wk.Dataset, "upload", upload)

    transfer.main(do_upload=False)
    assert calls == ["download"]
    transfer.main(do_upload=True)
    assert calls == ["download", "upload"]
    transfer.main(do_upload=True)
    assert calls == ["download", "upload"]
    monkeypatch.setattr(transfer, "MAG", "16-16-4")
    remote.layers["color"].mags[wk.Mag("16-16-4")] = None
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
