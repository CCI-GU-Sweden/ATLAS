from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from napari.components import LayerList
from napari.layers import Image, Points
from qtpy.QtWidgets import QApplication

from napari_atlas_cci import AtlasCCIWidget


@pytest.fixture
def widget(monkeypatch):
    app = QApplication.instance() or QApplication([])
    images = []
    viewer = SimpleNamespace(
        layers=LayerList(), add_image=lambda image, **kw: images.append((image, kw))
    )
    with monkeypatch.context() as startup:
        startup.setattr(AtlasCCIWidget, "refresh_local_folder_tree", lambda self: None)
        w = AtlasCCIWidget(viewer)
    yield w, images
    w.close()
    w._zalign_executor.shutdown()
    w._export_executor.shutdown()


def make_store(path):
    group = zarr.open_group(str(path / 'stitched_sections.zarr'), mode='w', zarr_format=2)
    group.attrs.update({'atlas_stitched_store_version': 1, 'axes': ['z', 'y', 'x']})
    group.create_array('images', data=np.arange(24, dtype=np.uint8).reshape(2, 3, 4))
    sections = group.create_group('sections')
    sections.create_group('S_4').attrs.update({'status': 'complete', 'atlas_section_number': 4,
        'source_folder': r'C:\\acquisition\\S_4', 'z_index': 0, 'shape_yx': [2, 3]})
    sections.create_group('S_9').attrs.update({'status': 'writing', 'atlas_section_number': 9,
        'z_index': 1, 'shape_yx': [3, 4]})
    return group


def test_watcher_display_and_refresh(widget, tmp_path):
    w, images = widget
    group = make_store(tmp_path)
    w.path_edit.setText(str(tmp_path))
    AtlasCCIWidget.refresh_local_folder_tree(w)
    root = w.local_folder_tree.topLevelItem(0)
    assert root.childCount() == 1
    assert '1 completed' in root.text(0)
    assert not w.stitching_all_btn.isEnabled()
    assert w.zalign_zshift_button.isEnabled()
    w._set_zalign_buttons_busy()
    assert not w.zalign_zshift_button.isEnabled()
    w._reset_zalign_buttons_idle()
    assert w.zalign_zshift_button.isEnabled()
    assert not w.zalign_preview_button.isEnabled()
    root.child(0).setSelected(True)
    assert w.display_btn.isEnabled()
    w.display_stitched_image()
    np.testing.assert_array_equal(images[0][0], [[0, 1, 2], [4, 5, 6]])
    group['sections']['S_9'].attrs['status'] = 'complete'
    AtlasCCIWidget.refresh_local_folder_tree(w)
    assert w.local_folder_tree.topLevelItem(0).childCount() == 2
    w._preview_ome_zarr_path(tmp_path).mkdir()
    w._reset_zalign_buttons_idle()
    assert w.zalign_preview_button.isEnabled()


def test_raw_folder_takes_precedence_even_when_invalid(widget, tmp_path):
    w, _ = widget
    make_store(tmp_path)
    (tmp_path / 'S_invalid').mkdir()
    w.path_edit.setText(str(tmp_path))
    AtlasCCIWidget.refresh_local_folder_tree(w)
    assert not w._watcher_mode
    assert w.local_folder_tree.topLevelItem(0).childCount() == 0
    assert w.stitching_all_btn.isEnabled()


def test_watcher_initial_alignment_requires_a_completed_section(widget, tmp_path):
    w, _ = widget
    group = make_store(tmp_path)
    group["sections"]["S_4"].attrs["status"] = "writing"
    w.path_edit.setText(str(tmp_path))
    w.refresh_local_folder_tree()
    assert w._watcher_mode
    assert not w.zalign_zshift_button.isEnabled()


def test_watcher_alignment_import_requires_consent_and_preserves_order(widget, tmp_path, monkeypatch):
    import pandas as pd
    from atlas.alignment.utils import ROI
    from qtpy.QtWidgets import QMessageBox

    w, _ = widget
    group = make_store(tmp_path)
    group["sections"]["S_9"].attrs["status"] = "complete"
    roi = ROI(0, 1, 0, 1)
    frame = pd.DataFrame({"store_path": ["old", "old"], "moving_section_number": [9, 4],
                          "moving_z_index": [1, 0], "moving_ROI": [roi, roi],
                          "cumulative_ROI": [roi, roi], "down_scale": [1, 1],
                          "current_shift": [np.zeros(2), np.ones(2)]}, index=[8, 3])
    frame.to_pickle(tmp_path / "z_alignment_results.pkl")
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    w.path_edit.setText(str(tmp_path))
    w.refresh_local_folder_tree()
    target = tmp_path / "alignment_results" / "z_alignment_results.pkl"
    assert not target.exists()
    w.refresh_local_folder_tree(allow_import=True)
    result = pd.read_pickle(target)
    assert result["moving_section_number"].tolist() == [9, 4]
    assert result.index.tolist() == [0, 1]
    assert not w.correction_Z_btn.isEnabled()
    assert not w.correction_apply_btn.isEnabled()
    assert w.zalign_final_button.isEnabled()
    assert not w.export_czi_button.isEnabled()
    assert not w.upload_wkn_button.isEnabled()
    assert not w.zalign_view_button.isEnabled()
    w._aligned_zarr_path(tmp_path).mkdir()
    w._aligned_ome_zarr_path(tmp_path).mkdir()
    w._reset_zalign_buttons_idle()
    assert w.export_czi_button.isEnabled()
    assert w.upload_wkn_button.isEnabled()
    assert w.zalign_view_button.isEnabled()
    assert not w.stitching_all_btn.isEnabled()
    assert result["store_path"].tolist() == [str((tmp_path / "stitched_sections.zarr").resolve())] * 2
    original = target.read_bytes()
    w.refresh_local_folder_tree(allow_import=True)
    assert target.read_bytes() == original


def test_watcher_invalid_import_keeps_sections_accessible(widget, tmp_path, monkeypatch):
    import pandas as pd
    from qtpy.QtWidgets import QMessageBox
    import napari_atlas_cci.Atlas_CCI_Widget as module

    w, _ = widget
    make_store(tmp_path)
    pd.DataFrame({"moving_section_number": [4]}).to_pickle(tmp_path / "z_alignment_results.pkl")
    errors = []
    monkeypatch.setattr(module, "show_error", errors.append)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    w.path_edit.setText(str(tmp_path))
    w.refresh_local_folder_tree(allow_import=True)
    assert errors and "Missing alignment columns" in errors[-1]
    assert w.local_folder_tree.topLevelItem(0).childCount() == 1
    assert not (tmp_path / "alignment_results" / "z_alignment_results.pkl").exists()


@pytest.mark.parametrize("watcher", [True, False])
def test_preview_renders_saved_alignment_then_opens_existing(widget, tmp_path, monkeypatch, watcher):
    from concurrent.futures import Future

    w, _ = widget
    if watcher:
        make_store(tmp_path)
    w.path_edit.setText(str(tmp_path))
    w.refresh_local_folder_tree()
    target = tmp_path / "alignment_results" / "z_alignment_results.pkl"
    target.parent.mkdir()
    target.write_bytes(b"saved alignment")
    w.update_series_status_indicators(tmp_path)
    assert w.zalign_preview_button.isEnabled()
    assert w.zalign_preview_button.text() == "Render and Preview"
    submitted = []
    future = Future()
    monkeypatch.setattr(w._zalign_executor, "submit", lambda fn, root: submitted.append((fn, root)) or future)
    finalized = []
    monkeypatch.setattr(w, "_finalize_apply_z_shifts", lambda payload: finalized.append(payload) or (True, ""))
    w.display_initial_zalign_preview()
    assert submitted[0][0] == w._apply_z_shifts_preview_worker
    assert submitted[0][1] == tmp_path
    assert not w.zalign_preview_button.isEnabled()
    w._preview_ome_zarr_path(tmp_path).mkdir()
    future.set_result((True, "", {"ome_zarr_path": "preview"}))
    w._poll_zalign_future()
    assert finalized == [{"ome_zarr_path": "preview"}]
    assert w.zalign_preview_button.text() == "Preview"
    displayed = []
    monkeypatch.setattr(w, "_display_zalign_ome_zarr", lambda use_downsample: displayed.append(use_downsample))
    w.display_initial_zalign_preview()
    assert displayed == [True]
    assert len(submitted) == 1


def test_preview_render_failure_is_reported(widget, tmp_path, monkeypatch):
    from concurrent.futures import Future
    import napari_atlas_cci.Atlas_CCI_Widget as module

    w, _ = widget
    make_store(tmp_path)
    w.path_edit.setText(str(tmp_path))
    w.refresh_local_folder_tree()
    target = tmp_path / "alignment_results" / "z_alignment_results.pkl"
    target.parent.mkdir()
    target.write_bytes(b"invalid")
    future = Future()
    monkeypatch.setattr(w._zalign_executor, "submit", lambda *args: future)
    errors = []
    monkeypatch.setattr(module, "show_error", errors.append)
    w.display_initial_zalign_preview()
    future.set_exception(ValueError("invalid alignment"))
    w._poll_zalign_future()
    assert errors and "invalid alignment" in errors[-1]
    assert w.zalign_preview_button.isEnabled()
    assert w.zalign_preview_button.text() == "Render and Preview"


def test_preview_and_stitching_do_not_run_together(widget, tmp_path, monkeypatch):
    import napari_atlas_cci.Atlas_CCI_Widget as module
    w, _ = widget
    make_store(tmp_path)
    w.path_edit.setText(str(tmp_path))
    w.refresh_local_folder_tree()
    target = tmp_path / "alignment_results" / "z_alignment_results.pkl"
    target.parent.mkdir()
    target.write_bytes(b"saved")
    errors = []
    monkeypatch.setattr(module, "show_error", errors.append)
    w._is_processing = True
    w._update_preview_control(tmp_path)
    assert not w.zalign_preview_button.isEnabled()
    w.display_initial_zalign_preview()
    assert "stitching" in errors[-1]
    w._preview_ome_zarr_path(tmp_path).mkdir()
    w._update_preview_control(tmp_path)
    assert w.zalign_preview_button.isEnabled()
    w._is_processing = False
    w._zalign_active_action = "preview"
    w._start_processing_queue([tmp_path / "S_1"])
    assert "preview rendering" in errors[-1]
    assert not w._is_processing


@pytest.mark.parametrize("watcher", [True, False])
def test_correction_requires_the_project_preview_layer(widget, tmp_path, watcher):
    w, _ = widget
    if watcher:
        make_store(tmp_path)
    results = tmp_path / "alignment_results"
    results.mkdir()
    (results / "z_alignment_results.pkl").touch()
    w._preview_ome_zarr_path(tmp_path).mkdir()
    w.path_edit.setText(str(tmp_path))
    w.refresh_local_folder_tree()
    assert w._watcher_mode is watcher

    def assert_correction_enabled(expected):
        assert w.correction_Z_btn.isEnabled() == expected
        assert w.correction_apply_btn.isEnabled() == expected

    assert_correction_enabled(False)
    preview_name = f"{tmp_path.name}_downsample"
    wrong_layer_type = Points(ndim=3, name=preview_name)
    w.viewer.layers.append(wrong_layer_type)
    assert_correction_enabled(False)
    w.viewer.layers.remove(wrong_layer_type)
    image = Image(np.zeros((2, 3, 4)), name="Different project preview")
    w.viewer.layers.append(image)
    assert_correction_enabled(False)
    image.name = preview_name
    assert_correction_enabled(True)
    image.name = "Renamed preview"
    assert_correction_enabled(False)
    image.name = preview_name
    assert_correction_enabled(True)
    w.viewer.layers.remove(image)
    assert_correction_enabled(False)
