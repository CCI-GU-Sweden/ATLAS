import json
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from qtpy.QtWidgets import QApplication

from napari_atlas_cci import AtlasCCIWidget
from napari_atlas_cci import _settings
from napari_atlas_cci._settings import (
    PluginSettings,
    load_settings,
    save_settings,
)
from napari_atlas_cci.gui import OptionsDialog


@pytest.fixture(scope="module")
def app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def preferences(tmp_path, monkeypatch):
    path = tmp_path / "config" / "settings.json"
    monkeypatch.setattr(_settings, "settings_path", lambda: path)
    return path


@pytest.fixture
def widget(app, preferences, monkeypatch):
    monkeypatch.setattr(
        AtlasCCIWidget, "refresh_local_folder_tree", lambda self: None
    )
    widget = AtlasCCIWidget(SimpleNamespace())
    yield widget
    widget.close()
    widget._zalign_executor.shutdown()
    widget._export_executor.shutdown()


def test_preferences_round_trip_precision_units_and_token_exclusion(
    preferences,
):
    settings = PluginSettings.from_dict(
        {
            "percentile_low": 1.23456789,
            "percentile_high": 99.987654321,
            "axial_pixel_size": 35.125,
            "axial_pixel_unit": "nm",
            "compression_enabled": False,
            "wkn_token": "never-save-this",
        }
    )
    save_settings(settings)
    assert load_settings() == settings
    text = preferences.read_text(encoding="utf-8")
    assert "never-save-this" not in text
    assert json.loads(text)["schema_version"] == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"percentile_low": -1},
        {"percentile_high": 101},
        {"percentile_low": 99.95},
        {"percentile_low": float("nan")},
        {"percentile_high": float("inf")},
        {"downsampling_factor": 0},
        {"thread_count": True},
        {"compression_enabled": "false"},
        {"axial_pixel_unit": "mm"},
    ],
)
def test_invalid_preferences_rejected(changes):
    with pytest.raises(ValueError):
        PluginSettings.from_dict(changes)


@pytest.mark.parametrize(
    "bad_document",
    [
        '{"schema_version": 999, "settings": {}}',
        '{"schema_version": 1, "settings": {"percentile_low": 100}}',
        "{invalid JSON",
    ],
)
def test_missing_and_bad_file_defaults(
    widget, preferences, monkeypatch, bad_document
):
    assert load_settings().percentile_low == 1.0
    assert widget.axial_pixel_unit_dropdown.currentText() == "µm"
    preferences.parent.mkdir()
    preferences.write_text(bad_document)
    messages = []
    module = import_module("napari_atlas_cci.Atlas_CCI_Widget")
    monkeypatch.setattr(module, "show_error", messages.append)
    other = AtlasCCIWidget(SimpleNamespace())
    assert other.settings == PluginSettings.from_dict({})
    assert "using defaults" in messages[0]
    assert preferences.read_text() == bad_document
    other.close()
    other._zalign_executor.shutdown()
    other._export_executor.shutdown()


def test_dialog_apply_reopen_cancel_and_restart(
    widget, preferences, monkeypatch
):
    def accept(dialog):
        assert dialog.values()["percentile_low"] == 1.0
        dialog.percentile_low_input.setText("2.125")
        dialog.percentile_high_input.setText("98.875")
        dialog._accept_options()
        return 1

    monkeypatch.setattr(widget, "_exec_dialog", accept)
    widget.open_options_dialog()
    assert widget.settings.percentile_low == 2.125
    before = preferences.read_bytes()

    def cancel(dialog):
        assert dialog.values()["percentile_low"] == 2.125
        dialog.percentile_low_input.setText("3.5")
        dialog.reject()
        return 0

    monkeypatch.setattr(widget, "_exec_dialog", cancel)
    widget.open_options_dialog()
    assert widget.settings.percentile_low == 2.125
    assert preferences.read_bytes() == before
    other = AtlasCCIWidget(SimpleNamespace())
    assert other.settings == widget.settings
    other.close()
    other._zalign_executor.shutdown()
    other._export_executor.shutdown()


@pytest.mark.parametrize(
    "low,high", [("90", "80"), ("", "99"), ("nan", "99"), ("-1", "99")]
)
def test_invalid_dialog_keeps_open_without_applying(app, low, high):
    dialog = OptionsDialog()
    applied = []
    dialog.options_applied.connect(applied.append)
    dialog.percentile_low_input.setText(low)
    dialog.percentile_high_input.setText(high)
    dialog._accept_options()
    assert not applied
    assert not dialog.validation_error.isHidden()
    assert "Invalid options" in dialog.validation_error.text()
    dialog.close()


def test_axial_size_and_units_restore(widget, preferences):
    widget.axial_pixel_size_input.setText("35.125")
    widget.axial_pixel_unit_dropdown.setCurrentText("nm")
    widget.axial_pixel_size_input.editingFinished.emit()
    assert load_settings().axial_pixel_size == 35.125
    other = AtlasCCIWidget(SimpleNamespace())
    assert float(other.axial_pixel_size_input.text()) == 35.125
    assert other.axial_pixel_unit_dropdown.currentText() == "nm"
    other.close()
    other._zalign_executor.shutdown()
    other._export_executor.shutdown()


@pytest.mark.parametrize(
    "action,downsample",
    [("preview", True), ("final", False), ("correction", True)],
)
def test_percentiles_reach_alignment_for_all_outputs(
    widget, tmp_path, monkeypatch, action, downsample
):
    widget.settings = PluginSettings.from_dict(
        {"percentile_low": 3.25, "percentile_high": 97.625}
    )
    results = tmp_path / "alignment_results"
    results.mkdir()
    pd.DataFrame({"z": [0]}).to_pickle(results / "z_alignment_results.pkl")
    called = []

    class AlignmentReached(Exception):
        pass

    def apply_alignment(frame, **kwargs):
        called.append(kwargs)
        raise AlignmentReached

    monkeypatch.setattr("atlas.io.apply_alignment", apply_alignment)
    monkeypatch.setattr(
        "atlas.alignment.correct_z_alignment_from_points",
        lambda frame, *args: frame,
    )
    with pytest.raises(AlignmentReached):
        if action == "correction":
            widget._apply_correction_worker(
                tmp_path, np.zeros((1, 3)), np.zeros((1, 3))
            )
        elif action == "preview":
            widget._apply_z_shifts_preview_worker(tmp_path)
        else:
            widget._write_aligned_zarr_worker(tmp_path)
    assert called[0]["percentile_low"] == 3.25
    assert called[0]["percentile_high"] == 97.625
    assert called[0]["use_down_sample"] is downsample


def test_upload_xy_pixel_size_is_read_and_forwarded(widget, tmp_path, monkeypatch):
    import pandas as pd

    widget.main_directory = tmp_path
    widget._final_zarr_path().mkdir()
    pd.DataFrame({"PixelSizeMicron": [0.01000112]}).to_csv(
        tmp_path / "phaseCC_stitching_S_1.csv", index=False
    )
    captured = []
    monkeypatch.setattr(widget, "_exec_dialog", lambda dialog: captured.append(dialog.values()))
    widget.upload_to_webknossos()
    assert captured[0]["xy_pixel_size"] == pytest.approx(10.00112)
    assert captured[0]["delete_local_dataset"] is True

    submitted = []
    monkeypatch.setattr(widget._export_executor, "submit", lambda *args: submitted.append(args))
    monkeypatch.setattr(widget, "_set_export_buttons_busy", lambda: None)
    widget._upload_to_webknossos({"xy_pixel_size": 12.5, "z_pixel_size": 40, "dataset_name": "test", "wkn_token": "token"})
    assert submitted[0][-2:] == ((12.5, 12.5, 40.0), True)
    widget._upload_to_webknossos({"xy_pixel_size": 12.5, "z_pixel_size": 40, "dataset_name": "test", "wkn_token": "token", "delete_local_dataset": False})
    assert submitted[1][-1] is False


def test_upload_requires_manual_xy_when_metadata_missing(widget, tmp_path, monkeypatch):
    widget.main_directory = tmp_path
    widget._final_zarr_path().mkdir()
    captured = []
    monkeypatch.setattr(widget, "_exec_dialog", lambda dialog: captured.append(dialog.xy_pixel_size_input.text()))
    widget.upload_to_webknossos()
    assert captured == [""]


def test_upload_dialog_shows_actual_editable_dataset_name(widget):
    from napari_atlas_cci.gui import UploadDataDialog

    dialog = UploadDataDialog(widget)
    dialog.set_values({"xy_pixel_size": 10, "z_pixel_size": 70, "dataset_name": "Section Set 6"})
    assert dialog.upload_name_preview.text() == "Section_Set_6"
    dialog.dataset_name_input.setText("My experiment 1")
    assert dialog.upload_name_preview.text() == "My_experiment_1"
    assert dialog.values()["dataset_name"] == "My experiment 1"
    dialog.close()


def test_master_store_calibration_reaches_both_exports(widget, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from atlas.io import StitchedZarrStore

    widget.main_directory = tmp_path
    (tmp_path / "stitched_sections.zarr").mkdir()
    widget._final_zarr_path().mkdir()
    store = SimpleNamespace(validate_section_pixel_sizes=lambda: {
        "Acquisition Pixel size": {"first_um": 0.0125}
    })
    monkeypatch.setattr(StitchedZarrStore, "open", lambda path: store)
    widget.pixel_size = {}  # Reopened project: no session calibration.
    captured = []
    monkeypatch.setattr(widget, "_exec_dialog", lambda dialog: captured.append(dialog.values()))
    widget.upload_to_webknossos()
    assert captured[0]["xy_pixel_size"] == pytest.approx(12.5)

    submitted = []
    monkeypatch.setattr(widget._export_executor, "submit", lambda *args: submitted.append(args))
    monkeypatch.setattr(widget, "_set_export_buttons_busy", lambda: None)
    widget.export_to_czi()
    assert submitted[0][3] == pytest.approx(0.0125)
