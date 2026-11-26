from concurrent.futures import Future
from contextlib import nullcontext
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
from pylibCZIrw import czi
from qtpy.QtWidgets import QApplication
from webknossos import BoundingBox, Dataset

from napari_atlas_cci import AtlasCCIWidget, _settings
from napari_atlas_cci._czi import write_czi_plane
from napari_atlas_cci._settings import PluginSettings
from napari_atlas_cci.gui import DownloadDataDialog


@pytest.fixture(scope="session")
def app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def widget(app, tmp_path, monkeypatch):
    monkeypatch.setattr(
        _settings, "settings_path", lambda: tmp_path / "preferences.json"
    )
    monkeypatch.setattr(
        AtlasCCIWidget, "refresh_local_folder_tree", lambda self: None
    )
    widget = AtlasCCIWidget(SimpleNamespace())
    widget.main_directory = tmp_path
    yield widget
    widget.close()
    widget._export_timer.stop()
    widget._export_executor.shutdown()
    widget._zalign_executor.shutdown()


def test_tiles_are_bounded_yxc_and_handle_edges():
    image = np.arange(1031 * 7, dtype=np.uint16).reshape(7, 1031)
    tiles = []
    writer = SimpleNamespace(write=lambda **kwargs: tiles.append(kwargs))
    write_czi_plane(writer, image, 2, channel=1)
    assert len(tiles) == 2
    restored = np.zeros_like(image)
    for tile in tiles:
        data = tile["data"]
        assert data.shape[2] == 1
        assert data.nbytes < 10_000_000
        assert tile["plane"] == {"Z": 2, "C": 1}
        x, y = tile["location"]
        restored[y : y + data.shape[0], x : x + data.shape[1]] = data[..., 0]
    np.testing.assert_array_equal(restored, image)


def test_failed_tile_write_is_reported():
    with pytest.raises(OSError, match="CZI write failed"):
        write_czi_plane(
            SimpleNamespace(write=lambda **kwargs: False), np.zeros((2, 3)), 0
        )


@pytest.mark.parametrize("compression", [None, "zstd0:ExplicitLevel=3"])
@pytest.mark.parametrize(
    "dataset_reference",
    ["dataset-id", "https://webknossos.org/datasets/dataset-id/view"],
)
def test_download_round_trip_real_wk_reader_and_czi(
    widget, tmp_path, monkeypatch, compression, dataset_reference
):
    dataset = Dataset(tmp_path / "wk", voxel_size=(4, 6, 40))
    layer = dataset.add_layer(
        "color",
        category="color",
        dtype="uint16",
        num_channels=2,
        data_format="zarr3",
    )
    box = BoundingBox((13, 17, 4), (1031, 7, 2))
    layer.bounding_box = box
    mag = layer.add_mag(1, chunk_shape=(32, 32, 32), shard_shape=(32, 32, 32))
    data = np.arange(2 * 1031 * 7 * 2, dtype=np.uint16).reshape(2, 1031, 7, 2)
    mag.write(data, absolute_bounding_box=box, allow_unaligned=True)
    opened = []
    context = []

    def open_remote(**kwargs):
        opened.append(kwargs)
        return dataset

    monkeypatch.setattr(Dataset, "open_remote", open_remote)
    monkeypatch.setattr(
        "webknossos.webknossos_context",
        lambda **kwargs: context.append(kwargs) or nullcontext(),
    )
    original = mag.get_buffered_slice_reader
    reads = []

    def slices(**kwargs):
        reads.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(mag, "get_buffered_slice_reader", slices)
    output = tmp_path / "download.czi"
    success, message, payload = widget._download_from_webknossos_worker(
        "secret", dataset_reference, output, compression
    )
    assert success, message
    assert payload == output
    assert reads == [{"buffer_size": 1, "dimension": "z"}]
    assert context == [{"token": "secret", "url": "https://webknossos.org"}]
    assert opened[0]["read_only"] is True
    assert (
        opened[0].get("dataset_id", opened[0].get("dataset_name_or_url"))
        == dataset_reference
    )
    with czi.open_czi(str(output)) as reader:
        for z in range(2):
            for channel in range(2):
                np.testing.assert_array_equal(
                    reader.read(plane={"Z": z, "C": channel})[..., 0],
                    data[channel, :, :, z].T,
                )
        distances = reader.metadata["ImageDocument"]["Metadata"]["Scaling"][
            "Items"
        ]["Distance"]
        scales = {entry["@Id"]: float(entry["Value"]) for entry in distances}
        assert scales == pytest.approx({"X": 4e-9, "Y": 6e-9, "Z": 40e-9})


def test_local_export_streams_without_loading_volume(
    widget, tmp_path, monkeypatch
):
    planes = np.arange(2 * 1 * 1031, dtype=np.uint16).reshape(2, 1, 1031)
    accesses = []

    class Source:
        ndim = 3
        shape = planes.shape
        attrs = {"AtlasMetadata": {"axes": "ZYX"}}

        def __array__(self, *args, **kwargs):
            raise AssertionError("Do not load the entire volume")

        def __getitem__(self, key):
            accesses.append(key[0])
            return planes[key]

    monkeypatch.setattr("zarr.open_array", lambda *args, **kwargs: Source())
    output = tmp_path / "local.czi"
    assert widget._export_to_czi_worker(
        tmp_path / "unused.zarr", output, 0.004, 0.04, None
    )[0]
    assert accesses == [0, 1]
    with czi.open_czi(str(output)) as reader:
        for z in range(2):
            np.testing.assert_array_equal(
                reader.read(plane={"Z": z})[..., 0], planes[z]
            )


def test_download_refuses_to_overwrite(widget, tmp_path):
    output = tmp_path / "existing.czi"
    output.write_bytes(b"preserve this")
    success, message, _ = widget._download_from_webknossos_worker(
        "secret", "id", output, None
    )
    assert not success
    assert "already exists" in message
    assert output.read_bytes() == b"preserve this"


def test_download_network_error_reports_failure_without_creating_czi(
    widget, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "webknossos.webknossos_context", lambda **kwargs: nullcontext()
    )

    def fail(**kwargs):
        raise RuntimeError("Access denied")

    monkeypatch.setattr(Dataset, "open_remote", fail)
    output = tmp_path / "failed.czi"
    success, message, payload = widget._download_from_webknossos_worker(
        "secret", "id", output, None
    )
    assert not success
    assert "Access denied" in message
    assert payload is None
    assert not output.exists()


def test_download_failure_restores_button(widget, tmp_path, monkeypatch):
    future = Future()
    monkeypatch.setattr(
        widget._export_executor, "submit", lambda *args: future
    )
    widget._download_from_webknossos(
        {
            "wkn_token": "token",
            "dataset_id": "id",
            "output_path": tmp_path / "failed.czi",
        }
    )
    errors = []
    monkeypatch.setattr(
        import_module("napari_atlas_cci.Atlas_CCI_Widget"),
        "show_error",
        errors.append,
    )
    future.set_result((False, "Download failed", None))
    widget._poll_export_future()
    assert errors == ["Download failed"]
    assert widget.download_wkn_button.isEnabled()


def test_download_dialog_fields_validation_and_cancel(app, tmp_path):
    dialog = DownloadDataDialog(location=tmp_path)
    emitted = []
    dialog.download_requested.connect(emitted.append)
    dialog._request_download()
    assert not emitted
    assert not dialog.validation_error.isHidden()
    dialog.wkn_token_input.setText("token")
    dialog.dataset_id_input.setText("dataset-id")
    dialog.output_name_input.setText("../bad.czi")
    dialog._request_download()
    assert not emitted
    dialog.output_name_input.setText("aligned")
    assert dialog.values()["output_path"] == tmp_path / "aligned.czi"
    dialog.reject()
    assert not emitted
    dialog._request_download()
    assert emitted[0]["dataset_id"] == "dataset-id"


@pytest.mark.parametrize(
    "enabled,expected", [(True, "zstd0:ExplicitLevel=7"), (False, None)]
)
def test_download_reuses_executor_compression_and_completion(
    widget, monkeypatch, tmp_path, enabled, expected
):
    widget.settings = PluginSettings.from_dict(
        {"compression_enabled": enabled, "compression_level": 7}
    )
    calls = []
    future = Future()
    monkeypatch.setattr(
        widget._export_executor,
        "submit",
        lambda *args: calls.append(args) or future,
    )
    path = tmp_path / "download.czi"
    widget._download_from_webknossos(
        {"wkn_token": "token", "dataset_id": "id", "output_path": path}
    )
    assert calls[0][-1] == expected
    assert not widget.download_wkn_button.isEnabled()
    assert "Download" in widget.download_wkn_button.text()
    messages = []
    monkeypatch.setattr(
        import_module("napari_atlas_cci.Atlas_CCI_Widget"),
        "show_info",
        messages.append,
    )
    future.set_result((True, "done", path))
    widget._poll_export_future()
    assert widget.download_wkn_button.isEnabled()
    assert str(path) in messages[0]
