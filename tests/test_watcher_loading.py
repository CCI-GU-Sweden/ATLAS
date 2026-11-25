from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from qtpy.QtWidgets import QApplication

from napari_atlas_cci import AtlasCCIWidget


@pytest.fixture
def widget(monkeypatch):
    app = QApplication.instance() or QApplication([])
    images = []
    viewer = SimpleNamespace(add_image=lambda image, **kw: images.append((image, kw)))
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
    assert not w.zalign_zshift_button.isEnabled()
    root.child(0).setSelected(True)
    assert w.display_btn.isEnabled()
    w.display_stitched_image()
    np.testing.assert_array_equal(images[0][0], [[0, 1, 2], [4, 5, 6]])
    group['sections']['S_9'].attrs['status'] = 'complete'
    AtlasCCIWidget.refresh_local_folder_tree(w)
    assert w.local_folder_tree.topLevelItem(0).childCount() == 2


def test_raw_folder_takes_precedence_even_when_invalid(widget, tmp_path):
    w, _ = widget
    make_store(tmp_path)
    (tmp_path / 'S_invalid').mkdir()
    w.path_edit.setText(str(tmp_path))
    AtlasCCIWidget.refresh_local_folder_tree(w)
    assert not w._watcher_mode
    assert w.local_folder_tree.topLevelItem(0).childCount() == 0
    assert w.stitching_all_btn.isEnabled()
