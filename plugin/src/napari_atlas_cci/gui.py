from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from qtpy.QtCore import Qt, QUrl, Signal
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)
from qtpy.QtGui import QIntValidator
from qtpy.QtGui import QDoubleValidator
from napari.utils.notifications import show_error
from ._settings import PluginSettings


def safe_dataset_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip(" ._") or "webknossos_dataset"

class LocalFolderTree(QTreeWidget):
    def __init__(self):
        super().__init__()
        self.setColumnCount(1)
        self.setHeaderLabels(["Atlas Project"])
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

    def populate_from_project(self, project_root: Path, series_folders: list[Path]) -> None:
        """Display one root item (project) and one child item per S_ folder."""
        self.clear()

        root_item = QTreeWidgetItem([project_root.name])
        root_item.setToolTip(0, str(project_root))
        root_item.setData(0, Qt.ItemDataRole.UserRole, str(project_root))
        root_item.setFlags(root_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)

        for folder in sorted(series_folders, key=lambda p: p.name):
            child_item = QTreeWidgetItem([folder.name])
            child_item.setToolTip(0, str(folder))
            child_item.setData(0, Qt.ItemDataRole.UserRole, str(folder))
            root_item.addChild(child_item)

        self.addTopLevelItem(root_item)
        root_item.setExpanded(True)
        self.resizeColumnToContents(0)


class OptionsDialog(QDialog):
    options_applied = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Atlas CCI Options")

        self.main_layout = QVBoxLayout(self)
        self.form_layout = QFormLayout()

        self.thread_count_input = QLineEdit(str(4))
        self.thread_count_input.setToolTip("Number of threads to use for processing.")
        self.thread_count_input.setValidator(QIntValidator(1, 100))
        self.form_layout.addRow("Threads:", self.thread_count_input)

        self.max_shift_pixels_input = QLineEdit(str(500))
        self.max_shift_pixels_input.setToolTip("Maximum shift in pixels for stitching.")
        self.max_shift_pixels_input.setValidator(QIntValidator(0, 1000))
        self.form_layout.addRow("Max shift (px):", self.max_shift_pixels_input)

        self.downsampling_factor_input = QLineEdit(str(10))
        self.downsampling_factor_input.setToolTip("Downsampling factor for alignment.")
        self.downsampling_factor_input.setValidator(QIntValidator(1, 100))
        self.form_layout.addRow("Downsampling factor:", self.downsampling_factor_input)

        self.percentile_low_input = QLineEdit("1.0")
        self.percentile_high_input = QLineEdit("99.95")
        for control in (self.percentile_low_input, self.percentile_high_input):
            control.setValidator(QDoubleValidator(0.0, 100.0, 10, control))
            control.setToolTip(
                "Saturation percentile used during alignment processing for saved "
                "preview and full-resolution outputs, including manual corrections. "
                "Requires 0 ≤ lower < upper ≤ 100."
            )
        self.form_layout.addRow("Lower percentile:", self.percentile_low_input)
        self.form_layout.addRow("Upper percentile:", self.percentile_high_input)

        self.compression_enabled_input = QCheckBox()
        self.compression_enabled_input.setToolTip("Enable ZSTD compression for CZI output.")
        self.compression_enabled_input.setChecked(True)
        self.compression_enabled_input.toggled.connect(
            self._update_compression_level_enabled
        )
        self.form_layout.addRow("Use CZI compression:", self.compression_enabled_input)

        self.compression_level_input = QLineEdit(str(3))
        self.compression_level_input.setToolTip("Compression level for CZI output as ZSTD compression.")
        self.compression_level_input.setValidator(QIntValidator(0, 22))
        self.form_layout.addRow("ZSTD compression level:", self.compression_level_input)

        self.main_layout.addLayout(self.form_layout)

        self.validation_error = QLabel()
        self.validation_error.setWordWrap(True)
        self.validation_error.setStyleSheet("color: #e06060;")
        self.validation_error.hide()
        self.main_layout.addWidget(self.validation_error)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self._accept_options)
        self.button_box.rejected.connect(self.reject)
        self.main_layout.addWidget(self.button_box)

    def values(self) -> dict:
        values = {
            "thread_count": int(self.thread_count_input.text()),
            "max_shift_pixels": int(self.max_shift_pixels_input.text()),
            "downsampling_factor": int(self.downsampling_factor_input.text()),
            "compression_enabled": self.compression_enabled_input.isChecked(),
            "compression_level": int(self.compression_level_input.text()),
            "percentile_low": float(self.percentile_low_input.text()),
            "percentile_high": float(self.percentile_high_input.text()),
        }
        validated = PluginSettings.from_dict(values).to_dict()
        return {key: validated[key] for key in values}

    def set_values(self, values: dict) -> None:
        self.thread_count_input.setText(str(values.get("thread_count", 4)))
        self.max_shift_pixels_input.setText(str(values.get("max_shift_pixels", 500)))
        self.downsampling_factor_input.setText(str(values.get("downsampling_factor", 10)))
        self.compression_enabled_input.setChecked(
            bool(values.get("compression_enabled", True))
        )
        self.compression_level_input.setText(str(values.get("compression_level", 3)))
        self.percentile_low_input.setText(str(values.get("percentile_low", 1.0)))
        self.percentile_high_input.setText(str(values.get("percentile_high", 99.95)))
        self._update_compression_level_enabled(
            self.compression_enabled_input.isChecked()
        )

    def _accept_options(self) -> None:
        try:
            values = self.values()
        except (ValueError, OverflowError) as exc:
            self.validation_error.setText(f"Invalid options: {exc}")
            self.validation_error.show()
            return
        self.options_applied.emit(values)
        self.accept()

    def _update_compression_level_enabled(self, enabled: bool) -> None:
        self.compression_level_input.setEnabled(enabled)


class UploadDataDialog(QDialog):
    upload_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Upload Data to Webknossos")

        self.main_layout = QVBoxLayout(self)
        self.token_button = QPushButton("Open WebKnossos token page")
        self.token_button.setToolTip("Open the WebKnossos account token page in your browser.")
        self.token_button.clicked.connect(self._open_webknossos_token_page)
        self.main_layout.addWidget(self.token_button)

        self.form_layout = QFormLayout()

        self.xy_pixel_size_input = QLineEdit(str(1.0))  # Can read from the metadata
        self.xy_pixel_size_input.setToolTip("XY pixel size in nanometers.")
        self.xy_pixel_size_input.setValidator(QDoubleValidator(0.001, 100000.0, 6))
        self.form_layout.addRow("XY pixel size (nm):", self.xy_pixel_size_input)

        self.z_pixel_size_input = QLineEdit(str(1.0))  # Can read from the user input in the main UI
        self.z_pixel_size_input.setToolTip("Z pixel size in nanometers.")
        self.z_pixel_size_input.setValidator(QDoubleValidator(0.001, 100000.0, 3))
        self.form_layout.addRow("Z pixel size (nm):", self.z_pixel_size_input)

        self.dataset_name_input = QLineEdit("MyDataset")
        self.dataset_name_input.setToolTip("Name given to the dataset to upload.")
        self.form_layout.addRow("Dataset name:", self.dataset_name_input)
        self.upload_name_preview = QLabel()
        self.form_layout.addRow("Name on WebKnossos:", self.upload_name_preview)
        self.dataset_name_input.textChanged.connect(self._update_upload_name_preview)
        self._update_upload_name_preview(self.dataset_name_input.text())

        self.wkn_token_input = QLineEdit()
        self.wkn_token_input.setToolTip("Webknossos API token for authentication.")
        if hasattr(QLineEdit, "EchoMode"):
            self.wkn_token_input.setEchoMode(QLineEdit.EchoMode.Password)
        else:
            self.wkn_token_input.setEchoMode(QLineEdit.Password)
        self.form_layout.addRow("WebKnossos token:", self.wkn_token_input)

        self.main_layout.addLayout(self.form_layout)

        self.delete_local_dataset_checkbox = QCheckBox(
            "Delete local WebKnossos dataset after successful upload"
        )
        self.delete_local_dataset_checkbox.setChecked(True)
        self.main_layout.addWidget(self.delete_local_dataset_checkbox)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self._request_upload)
        self.button_box.rejected.connect(self.reject)
        self.main_layout.addWidget(self.button_box)

    def values(self) -> dict:
        return {
            "xy_pixel_size": float(self.xy_pixel_size_input.text()),
            "z_pixel_size": float(self.z_pixel_size_input.text()),
            "dataset_name": self.dataset_name_input.text(),
            "wkn_token": self.wkn_token_input.text(),
            "delete_local_dataset": self.delete_local_dataset_checkbox.isChecked(),
        }

    def _update_upload_name_preview(self, name: str) -> None:
        if not name.strip():
            self.upload_name_preview.setText("Enter a dataset name")
        else:
            self.upload_name_preview.setText(safe_dataset_name(name.strip()))

    def set_values(self, values: dict) -> None:
        xy_pixel_size = values.get("xy_pixel_size")
        self.xy_pixel_size_input.setText("" if xy_pixel_size is None else str(xy_pixel_size))
        self.z_pixel_size_input.setText(str(values.get("z_pixel_size", 1.0)))
        self.dataset_name_input.setText(values.get("dataset_name", "MyDataset"))
        self.wkn_token_input.setText(values.get("wkn_token", ""))
        self.delete_local_dataset_checkbox.setChecked(values.get("delete_local_dataset", True))

    def _request_upload(self) -> None:
        if not self.xy_pixel_size_input.hasAcceptableInput() or not self.z_pixel_size_input.hasAcceptableInput():
            show_error("Enter valid positive XY and Z pixel sizes in nanometers.")
            return
        self.upload_requested.emit(self.values())
        self.accept()

    def _open_webknossos_token_page(self) -> None:
        QDesktopServices.openUrl(QUrl("https://webknossos.org/account/token"))


class DownloadDataDialog(QDialog):
    download_requested = Signal(dict)
    _open_webknossos_token_page = UploadDataDialog._open_webknossos_token_page

    def __init__(self, parent=None, location: Path | None = None):
        super().__init__(parent)
        self.setWindowTitle("Download from webknossos")
        layout = QVBoxLayout(self)
        self.token_button = QPushButton("Open WebKnossos token page")
        self.token_button.clicked.connect(self._open_webknossos_token_page)
        layout.addWidget(self.token_button)
        form = QFormLayout()
        self.wkn_token_input = QLineEdit()
        self.wkn_token_input.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("WebKnossos token:", self.wkn_token_input)
        self.dataset_id_input = QLineEdit()
        self.dataset_id_input.setToolTip(
            "ID or full URL of the fine-aligned dataset."
        )
        form.addRow("Dataset ID or URL:", self.dataset_id_input)
        self.output_name_input = QLineEdit("downloaded_volume.czi")
        form.addRow("Output filename:", self.output_name_input)
        self.location_input = QLineEdit(str(location or Path.cwd()))
        row = QHBoxLayout()
        row.addWidget(self.location_input)
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._browse_location)
        row.addWidget(self.browse_button)
        form.addRow("Save in:", row)
        layout.addLayout(form)
        info = QLabel(
            "Downloads the full-resolution color layer. Uses CZI compression from Options."
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        self.validation_error = QLabel()
        self.validation_error.setWordWrap(True)
        self.validation_error.hide()
        layout.addWidget(self.validation_error)
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self._request_download)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Download")
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Download")
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def _browse_location(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Save downloaded CZI in", self.location_input.text()
        )
        if folder:
            self.location_input.setText(folder)

    def values(self):
        token = self.wkn_token_input.text().strip()
        dataset_id = self.dataset_id_input.text().strip()
        name = self.output_name_input.text().strip()
        location = Path(self.location_input.text().strip()).expanduser()
        if not token or not dataset_id:
            raise ValueError(
                "Please provide a WebKnossos token and dataset ID or URL."
            )
        if (
            not name
            or any(character in name for character in '<>:"/\\|?*')
            or name in (".", "..")
        ):
            raise ValueError(
                "Provide a filename without directory separators or special characters."
            )
        if not Path(name).suffix:
            name += ".czi"
        if Path(name).suffix.lower() != ".czi" or not Path(name).stem:
            raise ValueError("The output filename must end in .czi.")
        if not self.location_input.text().strip() or not location.is_dir():
            raise ValueError("Choose an existing output folder.")
        path = location / name
        if path.exists():
            raise ValueError(
                "The output file already exists. Choose a different filename."
            )
        return {
            "wkn_token": token,
            "dataset_id": dataset_id,
            "output_path": path,
        }

    def _request_download(self):
        try:
            values = self.values()
        except ValueError as exc:
            self.validation_error.setText(str(exc))
            self.validation_error.show()
            return
        self.download_requested.emit(values)
        self.accept()


class ZarrImageViewer:
    def __init__(self, viewer: Any):
        self.viewer = viewer

    def remove_layer(self, layer_name: str) -> bool:
        for layer in list(self.viewer.layers):
            if layer.name == layer_name:
                self.viewer.layers.remove(layer)
                return True
        return False

    def display_ome_zarr(
        self,
        ome_zarr_path: Path,
        layer_name: str | None = None,
        contrast_limits: tuple[float, float] | None = None,
    ) -> tuple[bool, str]:
        import dask.array as da
        import zarr

        if not ome_zarr_path.exists():
            return False, f"Zarr output not found: {ome_zarr_path}"

        try:
            root_group = zarr.open_group(str(ome_zarr_path), mode="r")
            attrs = dict(root_group.attrs)

            multiscales = attrs.get("multiscales", [])

            if not multiscales:
                ome_metadata = attrs.get("ome", {})
                if isinstance(ome_metadata, dict):
                    multiscales = ome_metadata.get("multiscales", [])

            if not multiscales:
                return False, "Zarr output has no multiscales metadata."

            datasets = multiscales[0].get("datasets", [])
            if not datasets:
                return False, "Zarr output has no pyramid datasets."

            display_name = layer_name or ome_zarr_path.stem
            load_id = uuid4().hex

            pyramid = [
                da.from_zarr(
                    str(ome_zarr_path.joinpath(str(dataset["path"]))),
                    name=f"{display_name}-{load_id}-{idx}",
                )
                for idx, dataset in enumerate(datasets)
            ]

            self.remove_layer(display_name)

            add_kwargs = {
                "multiscale": True,
                "name": display_name,
            }
            if contrast_limits is not None:
                add_kwargs["contrast_limits"] = contrast_limits

            self.viewer.add_image(pyramid, **add_kwargs)
            return True, ""
        except Exception as exc:
            return False, f"Failed to display Zarr output: {exc}"
