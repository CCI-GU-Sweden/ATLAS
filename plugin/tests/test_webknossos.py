"""Real Zarr conversion/pyramid tests; only server interaction is mocked."""

import json
import os
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from cluster_tools import SequentialExecutor
from webknossos import Dataset
from webknossos.dataset.layer.layer import Layer

from napari_atlas_cci._webknossos import prepare_dataset


@pytest.fixture
def project(tmp_path, monkeypatch):
    # Set this to a mapped network directory to repeat the same checks over SMB.
    root = os.environ.get("ATLAS_WEBKNOSSOS_TEST_ROOT")
    if root:
        import tempfile

        with tempfile.TemporaryDirectory(
            prefix="atlas-wk-test-", dir=root
        ) as folder:
            yield from _project(Path(folder), monkeypatch)
    else:
        yield from _project(tmp_path, monkeypatch)


def _project(root, monkeypatch):
    original = Layer.downsample
    calls = []

    def downsample(self, **kwargs):
        calls.append({"buffer_shape": kwargs.get("buffer_shape")})
        kwargs.pop("executor", None)
        with SequentialExecutor() as executor:
            return original(self, executor=executor, **kwargs)

    monkeypatch.setattr(Layer, "downsample", downsample)
    source = root / "aligned.zarr"
    data = np.arange(9 * 37 * 261, dtype=np.uint16).reshape(9, 37, 261)
    array = zarr.open_array(
        str(source),
        mode="w",
        shape=data.shape,
        chunks=(4, 16, 32),
        dtype=data.dtype,
    )
    array[:] = data
    array.attrs["AtlasMetadata"] = {"axes": "ZYX"}
    output = root / "webknossos" / "example"
    output.parent.mkdir()
    yield source, output, data, calls


def prepare(source, output, voxel_size=(4, 4, 40)):
    return prepare_dataset(source, output, "example", voxel_size)


def test_direct_conversion_orientation_metadata_and_reuse(project, capsys):
    source, output, data, calls = project
    dataset = prepare(source, output)
    progress = capsys.readouterr().out
    assert "Copying aligned data: 2/2 blocks (100%)" in progress
    assert "Creating WebKnossos resolution pyramid" in progress
    assert "ready for upload" in progress
    layer = Dataset.open(output).get_layer("color")
    np.testing.assert_array_equal(
        layer.get_mag(1).read()[0], data.transpose(2, 1, 0)
    )
    assert dataset.voxel_size == (4, 4, 40)
    assert tuple(layer.bounding_box.size) == data.shape[::-1]
    assert layer.dtype == data.dtype
    assert len(layer.mags) > 1
    assert max(max(mag.to_tuple()) for mag in layer.mags) == 32
    assert set(output.parent.iterdir()) == {output}
    assert json.loads((output / ".atlas_cci_ready").read_text())[
        "shape"
    ] == list(data.shape)
    prepare(source, output)
    assert len(calls) == 1


def test_source_chunk_edit_and_voxel_size_invalidate_reuse(project):
    source, output, data, calls = project
    prepare(source, output)
    old_directory_time = source.stat().st_mtime_ns
    array = zarr.open_array(str(source), mode="r+")
    array[0, 0, 0] = 50000
    os.utime(source, ns=(old_directory_time, old_directory_time))
    dataset = prepare(source, output)
    assert dataset.get_layer("color").get_mag(1).read()[0, 0, 0, 0] == 50000
    dataset = prepare(source, output, (8, 8, 40))
    assert dataset.voxel_size == (8, 8, 40)
    assert len(calls) == 3


def test_downsampling_limits_workers(project, monkeypatch):
    import cluster_tools

    calls = []

    def get_executor(environment, **kwargs):
        calls.append((environment, kwargs))
        return SequentialExecutor()

    monkeypatch.setattr(cluster_tools, "get_executor", get_executor)
    source, output, _, downsample_calls = project
    prepare(source, output)
    assert calls == [("multiprocessing", {"max_workers": 1})]
    assert downsample_calls == [{"buffer_shape": (128, 128, 128)}]


def test_interrupted_build_retries_without_ready_marker(project, monkeypatch):
    source, output, data, calls = project
    original = Layer.downsample

    def interrupt(*args, **kwargs):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(Layer, "downsample", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        prepare(source, output)
    assert not (output / ".atlas_cci_ready").exists()
    monkeypatch.setattr(Layer, "downsample", original)
    prepare(source, output)
    assert (output / ".atlas_cci_ready").exists()


def test_upload_failure_retry_uses_project_dataset(project, monkeypatch):
    source, output, data, calls = project
    widget_class = import_module("napari_atlas_cci").AtlasCCIWidget
    widget = widget_class.__new__(widget_class)
    uploads = []

    def upload(self, **kwargs):
        assert Path(str(self.path)) == output
        assert kwargs == {"new_dataset_name": "example"}
        uploads.append(self.path)
        if len(uploads) == 1:
            raise RuntimeError("connection lost")
        return SimpleNamespace(url="https://webknossos.org/datasets/test")

    monkeypatch.setattr(Dataset, "upload", upload)
    args = (source, output.parent.parent, "test-token", "example", (4, 4, 40))
    assert widget._upload_to_webknossos_worker(*args)[0] is False
    assert widget._upload_to_webknossos_worker(*args)[0] is True
    assert len(calls) == 1


def test_legacy_marker_and_partial_output_are_rebuilt(project):
    source, output, data, calls = project
    output.mkdir()
    (output / ".atlas_cci_ready").write_text("ready\n")
    (output / "incomplete").write_text("partial build")
    prepare(source, output)
    assert not (output / "incomplete").exists()
    assert len(calls) == 1


def test_real_hive_image_crop(project):
    root = os.environ.get("ATLAS_WEBKNOSSOS_TEST_ROOT")
    if not root:
        pytest.skip(
            "Set ATLAS_WEBKNOSSOS_TEST_ROOT to test the real aligned image"
        )
    source, output, _, calls = project
    aligned = zarr.open_array(
        str(Path(root) / "HHD-S6_aligned.zarr"), mode="r"
    )
    crop = np.asarray(aligned[:4, :128, :261])
    array = zarr.open_array(
        str(source),
        mode="w",
        shape=crop.shape,
        chunks=(1, 64, 64),
        dtype=crop.dtype,
    )
    array[:] = crop
    array.attrs["AtlasMetadata"] = {"axes": "ZYX"}
    dataset = prepare(source, output)
    actual = dataset.get_layer("color").get_mag(1).read()[0]
    np.testing.assert_array_equal(actual, crop.transpose(2, 1, 0))


@pytest.mark.parametrize("delete_local, upload_fails", [(True, False), (False, False), (True, True)])
def test_upload_cleanup_only_after_success(tmp_path, monkeypatch, delete_local, upload_fails):
    from contextlib import nullcontext
    import webknossos
    import napari_atlas_cci._webknossos as wk_helper

    widget_class = import_module("napari_atlas_cci").AtlasCCIWidget
    widget = widget_class.__new__(widget_class)
    local = tmp_path / "webknossos" / "example"
    source = tmp_path / "aligned.zarr"
    source.mkdir()

    class FakeDataset:
        def upload(self, **kwargs):
            if upload_fails:
                raise RuntimeError("upload failed")
            return SimpleNamespace(url="https://webknossos.org/datasets/test")

    def prepare_dataset(*args):
        local.mkdir()
        (local / "data").write_text("derived")
        return FakeDataset()

    monkeypatch.setattr(wk_helper, "prepare_dataset", prepare_dataset)
    monkeypatch.setattr(webknossos, "webknossos_context", lambda **kwargs: nullcontext())
    success, message, url = widget._upload_to_webknossos_worker(
        source, tmp_path, "token", "example", (10, 10, 70), delete_local
    )
    assert success is not upload_fails
    assert local.exists() is (upload_fails or not delete_local)
    assert local.parent.exists() is (upload_fails or not delete_local)
    assert source.exists()
    if not upload_fails:
        assert url == "https://webknossos.org/datasets/test"


def test_upload_cleanup_failure_reports_success(tmp_path, monkeypatch):
    from contextlib import nullcontext
    import shutil
    import webknossos
    import napari_atlas_cci._webknossos as wk_helper

    widget_class = import_module("napari_atlas_cci").AtlasCCIWidget
    widget = widget_class.__new__(widget_class)
    fake_dataset = SimpleNamespace(upload=lambda **kwargs: SimpleNamespace(url="https://webknossos.org/datasets/test"))
    monkeypatch.setattr(wk_helper, "prepare_dataset", lambda *args: fake_dataset)
    monkeypatch.setattr(webknossos, "webknossos_context", lambda **kwargs: nullcontext())
    def fail_cleanup(path):
        raise PermissionError("access denied")
    monkeypatch.setattr(shutil, "rmtree", fail_cleanup)
    success, message, url = widget._upload_to_webknossos_worker(
        tmp_path / "aligned.zarr", tmp_path, "token", "example", (10, 10, 70), True
    )
    assert success
    assert "Could not delete" in message
    assert url == "https://webknossos.org/datasets/test"


def test_upload_cleanup_preserves_other_datasets(tmp_path, monkeypatch):
    from contextlib import nullcontext
    import webknossos
    import napari_atlas_cci._webknossos as wk_helper

    widget_class = import_module("napari_atlas_cci").AtlasCCIWidget
    widget = widget_class.__new__(widget_class)
    parent = tmp_path / "webknossos"
    other = parent / "other"
    other.mkdir(parents=True)
    local = parent / "example"
    local.mkdir()
    fake = SimpleNamespace(upload=lambda **kwargs: SimpleNamespace(url="https://webknossos.org/datasets/test"))
    monkeypatch.setattr(wk_helper, "prepare_dataset", lambda *args: fake)
    monkeypatch.setattr(webknossos, "webknossos_context", lambda **kwargs: nullcontext())
    success, _, _ = widget._upload_to_webknossos_worker(
        tmp_path / "aligned.zarr", tmp_path, "token", "example", (10, 10, 70), True
    )
    assert success
    assert not local.exists()
    assert other.is_dir()
