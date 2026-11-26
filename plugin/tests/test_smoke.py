from importlib import import_module
from importlib.resources import files


def test_package_imports():
    module = import_module("napari_atlas_cci")

    assert hasattr(module, "AtlasCCIWidget")


def test_napari_manifest_is_packaged():
    manifest = files("napari_atlas_cci").joinpath("napari.yaml")

    assert manifest.is_file()
    assert "napari-atlas-cci.open_widget" in manifest.read_text()


def test_webknossos_upload_name_rejects_path_separators():
    widget_class = import_module("napari_atlas_cci").AtlasCCIWidget
    widget = widget_class.__new__(widget_class)

    upload_name = widget._webknossos_upload_name("abc123/Section Set 5")

    assert upload_name == "abc123_Section_Set_5"
    assert "/" not in upload_name
    assert " " not in upload_name


def test_webknossos_upload_name_has_fallback():
    widget_class = import_module("napari_atlas_cci").AtlasCCIWidget
    widget = widget_class.__new__(widget_class)

    assert widget._webknossos_upload_name("///") == "webknossos_dataset"
    assert widget._webknossos_upload_name(r"folder\dataset") == "folder_dataset"
    assert widget._webknossos_upload_name("Section Set 5") == "Section_Set_5"
