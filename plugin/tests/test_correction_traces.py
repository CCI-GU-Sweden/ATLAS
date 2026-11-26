from concurrent.futures import Future
import re
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from napari.layers import Points
from qtpy.QtWidgets import QApplication

from napari_atlas_cci import AtlasCCIWidget, _settings


def trace(z):
    return np.column_stack(
        (z, np.asarray(z) * 2 + 10, np.asarray(z) * 3 + 5)
    ).astype(float)


@pytest.mark.parametrize(
    "slices,pairs",
    [
        ([2, 3, 4], [(2, 3), (3, 4)]),
        ([4, 3, 2], [(4, 3), (3, 2)]),
        ([2, 3, 4, 8, 9, 10], [(2, 3), (3, 4), (8, 9), (9, 10)]),
        ([10, 9, 8, 4, 3, 2], [(10, 9), (9, 8), (4, 3), (3, 2)]),
        ([2, 3, 4, 10, 9, 8], [(2, 3), (3, 4), (10, 9), (9, 8)]),
    ],
)
def test_adjacent_trace_pairs_preserve_direction_and_gaps(slices, pairs):
    points = trace(slices)
    fixed, moving = AtlasCCIWidget._trace_correction_pairs(points)
    assert list(zip(fixed[:, 0], moving[:, 0])) == pairs
    for f, m in zip(fixed, moving):
        np.testing.assert_array_equal(f, points[points[:, 0] == f[0]][0])
        np.testing.assert_array_equal(m, points[points[:, 0] == m[0]][0])


@pytest.mark.parametrize(
    "slices,message",
    [
        ([2], "at least two"),
        ([2, 3, 3, 4], "duplicates on slice(s): 3"),
        ([2, 3, 8], "isolated"),
        ([2, 4], "isolated"),
        ([2, 3, 5, 4], "Z order"),
        ([2, 8, 3, 9], "Z order"),
        ([2.5, 3.5], "integer slice"),
        ([-1, 0], "nonnegative"),
        ([float("nan"), 3], "finite"),
    ],
)
def test_invalid_traces_rejected(slices, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        AtlasCCIWidget._trace_correction_pairs(trace(slices))


def test_trace_requires_zyx_coordinates():
    with pytest.raises(ValueError, match="coordinates"):
        AtlasCCIWidget._trace_correction_pairs(np.zeros((2, 2)))


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

    class Layers(list):
        selection = SimpleNamespace(active=None)

    layers = Layers()

    def add_points(**kwargs):
        layer = Points(**kwargs)
        layers.append(layer)
        return layer

    viewer = SimpleNamespace(
        layers=layers, dims=SimpleNamespace(ndim=3), add_points=add_points
    )
    widget = AtlasCCIWidget(viewer)
    widget.main_directory = tmp_path
    monkeypatch.setattr(widget, "_has_z_alignment_results", lambda root: True)
    monkeypatch.setattr(
        widget, "_remove_zalign_image_layer", lambda **kwargs: None
    )
    monkeypatch.setattr(widget, "_set_zalign_buttons_busy", lambda: None)
    calls = []

    def submit(*args):
        calls.append(args)
        return Future()

    monkeypatch.setattr(widget._zalign_executor, "submit", submit)
    widget.submitted = calls
    yield widget
    widget.close()
    widget._zalign_executor.shutdown()
    widget._export_executor.shutdown()


@pytest.mark.parametrize("slices", [[2, 3, 4, 8, 9, 10], [10, 9, 8, 4, 3, 2]])
def test_trace_ui_and_real_atlas_correction(widget, monkeypatch, slices):
    from atlas.alignment.utils import ROI

    widget.create_correction_points_layer()
    assert [layer.name for layer in widget.viewer.layers] == [
        "Reference Points"
    ]
    widget.viewer.layers[0].data = trace(slices)
    widget.apply_correction()
    assert len(widget.submitted) == 1
    worker, root, fixed, moving = widget.submitted[0]
    assert len(fixed) == 4
    np.testing.assert_array_equal(moving[:, 0] - fixed[:, 0], np.ones(4))
    assert not widget.viewer.layers

    results = root / "alignment_results"
    results.mkdir()
    frame = pd.DataFrame(
        {
            "current_shift": [np.zeros(2) for _ in range(11)],
            "moving_ROI": [ROI(0, 100, 0, 100) for _ in range(11)],
        }
    )
    path = results / "z_alignment_results.pkl"
    frame.to_pickle(path)
    monkeypatch.setattr(
        widget, "_apply_z_shifts_preview_worker", lambda root: (True, "", {})
    )
    assert worker(root, fixed, moving)[0]
    corrected = pd.read_pickle(path)
    for slices in ([2, 3, 4], [8, 9, 10]):
        points = trace(slices)
        aligned = np.array(
            [
                point[1:] + corrected.at[int(point[0]), "cumulative_shift"]
                for point in points
            ]
        )
        np.testing.assert_array_equal(
            aligned, np.repeat(aligned[:1], 3, axis=0)
        )
    for gap_slice in (5, 6, 7, 8):
        np.testing.assert_array_equal(
            corrected.at[gap_slice, "current_shift"], [0, 0]
        )
    assert (
        results / "z_alignment_results_before_manual_correction.pkl"
    ).exists()


def test_invalid_trace_preserves_layers_and_does_not_submit(
    widget, monkeypatch
):
    errors = []
    monkeypatch.setattr(
        import_module("napari_atlas_cci.Atlas_CCI_Widget"),
        "show_error",
        errors.append,
    )
    widget.create_correction_points_layer()
    layer = widget.viewer.layers[0]
    layer.data = trace([2, 3, 3, 4])
    widget.apply_correction()
    assert not widget.submitted
    assert widget.viewer.layers[0] is layer
    assert "duplicates" in errors[0]


def test_explicit_pairs_workflow_retained(widget):
    widget.correction_mode_dropdown.setCurrentIndex(1)
    widget.create_correction_points_layer()
    assert [layer.name for layer in widget.viewer.layers] == [
        "Reference Points",
        "Adjustable Points",
    ]
    fixed, moving = widget.viewer.layers
    fixed.data, moving.data = trace([2, 8]), trace([3, 9])
    widget.apply_correction()
    assert len(widget.submitted) == 1
    np.testing.assert_array_equal(widget.submitted[0][2], trace([2, 8]))
    np.testing.assert_array_equal(widget.submitted[0][3], trace([3, 9]))
