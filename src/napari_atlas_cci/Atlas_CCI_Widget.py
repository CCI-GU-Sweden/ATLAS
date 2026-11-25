import napari
import re
import textwrap
import os
import tempfile
from queue import SimpleQueue, Empty
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import cast
import tifffile as tiff
import pandas as pd
import numpy as np
from napari.layers import Image
from napari.utils.notifications import show_error
from napari.utils.notifications import show_info
from qtpy.QtCore import Qt, QTimer, QSettings
from qtpy.QtGui import QBrush, QColor, QDoubleValidator
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QComboBox
)

from .gui import (
    LocalFolderTree,
    OptionsDialog,
    UploadDataDialog,
    DownloadDataDialog,
    ZarrImageViewer,
    safe_dataset_name,
)
from ._settings import PluginSettings, load_settings, save_settings

from atlas.io import extract_s_number

#status colors
PROCESSED = QColor("#66CC66")
ONGOING = QColor("#FF9566")
PENDING = QColor("#FFCC66")
FAILED = QColor("#FF3333")
UNPROCESSED = QColor("#BEBEBE")
SELECTED_SERIES_BACKGROUND = QColor("#2F5F90")
SERIES_STATUS_COLOR_ROLE = getattr(
    Qt.ItemDataRole.UserRole,
    "value",
    Qt.ItemDataRole.UserRole,
) + 1

AXIAL_PIXEL_SIZE = 0.300
AXIAL_PIXEL_SIZE_UNITS_OPTIONS = ["nm", "µm"]
ZALIGN_STATUS_MAX_CHARS = 120
ALIGNMENT_RESULTS_PATH = Path("alignment_results") / "z_alignment_results.pkl"

def _preview_layer_name(root_folder: Path) -> str:
    return f"{root_folder.name}_downsample"


class AtlasCCIWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        settings_error = None
        try:
            self.settings = load_settings()
        except (OSError, ValueError) as exc:
            self.settings = PluginSettings.from_dict({})
            settings_error = f"Could not load Atlas CCI settings; using defaults: {exc}"
        self.viewer = viewer
        self.zarr_viewer = ZarrImageViewer(viewer)
        self.main_directory = Path.cwd()
        self.pixel_size = {"Axial": self.settings.axial_pixel_size}

        self.pending_series: set[Path] = set()
        self.ongoing_series: set[Path] = set()
        self.failed_series: set[Path] = set()
        self._is_processing = False
        self._executor: ThreadPoolExecutor | None = None
        self._series_futures: dict[Path, Future] = {}
        self._stitch_events = SimpleQueue()
        self._master_sections_cache = None
        self._watcher_mode = False
        self._watcher_sections = {}
        self._processing_order: list[Path] = []
        self._next_queue_index = 0
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(150)
        self._status_timer.timeout.connect(self._poll_processing_futures)

        self._zalign_executor = ThreadPoolExecutor(max_workers=1)
        self._zalign_future: Future | None = None
        self._zalign_active_action: str | None = None
        self._zalign_spinner_frames = ("|", "/", "-", "\\")
        self._zalign_spinner_index = 0
        self._zalign_timer = QTimer(self)
        self._zalign_timer.setInterval(120)
        self._zalign_timer.timeout.connect(self._poll_zalign_future)

        self._export_executor = ThreadPoolExecutor(max_workers=1)
        self._export_future: Future | None = None
        self._export_active_action: str | None = None
        self._export_spinner_index = 0
        self._export_timer = QTimer(self)
        self._export_timer.setInterval(120)
        self._export_timer.timeout.connect(self._poll_export_future)
        self.setWindowTitle("Atlas CCI")

        self.main_layout = QVBoxLayout(self)

        # --- path row ---
        self.path_layout = QHBoxLayout()
        self.path_layout.addWidget(QLabel("Path:"))
        self.path_edit = QLineEdit()
        self.path_edit.setFixedWidth(200)
        self.path_edit.setText(str(self.main_directory))
        self.path_layout.addWidget(self.path_edit)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_for_atlas_project)
        self.path_layout.addWidget(self.browse_btn)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(lambda: self.refresh_local_folder_tree(allow_import=True))
        self.refresh_btn.setVisible(False)
        self.path_layout.addWidget(self.refresh_btn)

        self.options_button = QPushButton("Options...")
        self.options_button.clicked.connect(self.open_options_dialog)
        self.path_layout.addWidget(self.options_button)

        self.main_layout.addLayout(self.path_layout)

        # Add the local folder tree widget
        self.local_folder_tree = LocalFolderTree()
        self.local_folder_tree.itemSelectionChanged.connect(self._on_tree_selection_changed)
        self.main_layout.addWidget(self.local_folder_tree)
    
        # --- Tile stitching row ---
        self.stitching_layout = QHBoxLayout()
        self.stitching_all_btn = QPushButton("Stitch ALL Series")
        self.stitching_all_btn.clicked.connect(self.process_all_series)
        self.stitching_one_btn = QPushButton("Stitch Selected Series")
        self.stitching_one_btn.clicked.connect(self.process_selected_series)
        self.stitching_layout.addWidget(self.stitching_all_btn)
        self.stitching_layout.addWidget(self.stitching_one_btn)
        self.main_layout.addLayout(self.stitching_layout)

        # --- Tile display row ---
        self.tile_display_layout = QHBoxLayout()

        self.display_btn = QPushButton("Display Stitched Image")
        self.display_btn.clicked.connect(self.display_stitched_image)
        self.display_btn.setEnabled(False)
        self.tile_display_layout.addWidget(self.display_btn)

        self.main_layout.addLayout(self.tile_display_layout)

        # --- Z align part ---
        self.z_align_layout_pixel = QVBoxLayout()
        self.pixel_size_input = QHBoxLayout()
        self.pixel_axial_label = QLabel("Axial Pixel Size:")
        self.axial_pixel_size_input = QLineEdit(str(self.settings.axial_pixel_size))
        self.axial_pixel_size_input.setFixedWidth(200)
        self.axial_pixel_size_input.setToolTip("Axial pixel size in microns (µm) or nanometers (nm).")
        self.axial_pixel_size_input.setValidator(QDoubleValidator(0.01, 1000.0, 3))
        self.axial_pixel_size_input.editingFinished.connect(self.update_pixel_size_from_input)

        self.axial_pixel_unit_dropdown = QComboBox()
        self.axial_pixel_unit_dropdown.addItems(AXIAL_PIXEL_SIZE_UNITS_OPTIONS)
        self.axial_pixel_unit_dropdown.setCurrentText(self.settings.axial_pixel_unit)
        self.axial_pixel_unit_dropdown.currentTextChanged.connect(self.update_pixel_size_from_input)

        self.pixel_size_input_widget = QWidget()
        self.pixel_size_input.addWidget(self.pixel_axial_label)
        self.pixel_size_input.addWidget(self.axial_pixel_size_input)
        self.pixel_size_input.addWidget(self.axial_pixel_unit_dropdown)
        self.pixel_size_input_widget.setLayout(self.pixel_size_input)
        self.z_align_layout_pixel.addWidget(self.pixel_size_input_widget)

        self.initial_zalign_layout = QHBoxLayout()

        self.zalign_zshift_button = QPushButton("Initial Z Alignment")
        self.zalign_zshift_button.setToolTip("Calculate Z shifts and apply a downsampled Z alignment.")
        self.zalign_zshift_button.clicked.connect(self.initial_alignement)
        self.zalign_zshift_button.setEnabled(False)
        self.initial_zalign_layout.addWidget(self.zalign_zshift_button)

        self.zalign_preview_button = QPushButton("Preview")
        self.zalign_preview_button.setToolTip("Display the downsampled Z alignment OME-Zarr output.")
        self.zalign_preview_button.clicked.connect(self.display_initial_zalign_preview)
        self.zalign_preview_button.setEnabled(False)
        self.initial_zalign_layout.addWidget(self.zalign_preview_button)

        self.initial_zalign_widget = QWidget()
        self.initial_zalign_widget.setLayout(self.initial_zalign_layout)
        self.z_align_layout_pixel.addWidget(self.initial_zalign_widget)

        self.correction_layout = QHBoxLayout()
        self.correction_mode_dropdown = QComboBox()
        self.correction_mode_dropdown.addItem("Trace: one point per plane", "trace")
        self.correction_mode_dropdown.addItem("Explicit point pairs", "pairs")
        self.correction_mode_dropdown.setToolTip(
            "Trace: follow an object in Reference Points, one point per consecutive "
            "Z plane, forward or backward. Gaps start separate traces. "
            "Explicit pairs: match Reference Points with Adjustable Points."
        )
        self.z_align_layout_pixel.addWidget(self.correction_mode_dropdown)
        self.correction_Z_btn = QPushButton("Enter Correction")
        self.correction_Z_btn.setToolTip("Create the points layer(s) for the selected correction mode.")
        self.correction_Z_btn.clicked.connect(self.create_correction_points_layer)
        self.correction_Z_btn.setEnabled(False)

        self.correction_apply_btn = QPushButton("Apply Correction")
        self.correction_apply_btn.setToolTip("Apply manual Z correction to the downsampled preview OME-Zarr output.")
        self.correction_apply_btn.clicked.connect(self.apply_correction)
        self.correction_apply_btn.setEnabled(False)

        self.correction_layout.addWidget(self.correction_Z_btn)
        self.correction_layout.addWidget(self.correction_apply_btn)
        self.correction_widget = QWidget()
        self.correction_widget.setLayout(self.correction_layout)
        self.z_align_layout_pixel.addWidget(self.correction_widget)

        self.final_zalign_layout = QHBoxLayout()

        self.zalign_final_button = QPushButton("Final Z Alignment")
        self.zalign_final_button.setToolTip("Apply previously calculated Z shifts to the stitched series and save as aligned Zarr output.")
        self.zalign_final_button.clicked.connect(self.finalize_alignement)
        self.zalign_final_button.setEnabled(False)
        self.final_zalign_layout.addWidget(self.zalign_final_button)

        self.zalign_view_button = QPushButton("View")
        self.zalign_view_button.setToolTip("Display the final Z alignment OME-Zarr output.")
        self.zalign_view_button.clicked.connect(self.display_final_zalign_output)
        self.zalign_view_button.setEnabled(False)
        self.final_zalign_layout.addWidget(self.zalign_view_button)

        self.final_zalign_widget = QWidget()
        self.final_zalign_widget.setLayout(self.final_zalign_layout)
        self.z_align_layout_pixel.addWidget(self.final_zalign_widget)

        self.zalign_status_label = QLabel("Z Align: [UNPROCESSED] Idle")
        self.zalign_status_label.setWordWrap(True)
        self.zalign_status_label.setMinimumWidth(0)
        self.zalign_status_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self.z_align_layout_pixel.addWidget(self.zalign_status_label)
        self._set_zalign_status("UNPROCESSED", "Idle", UNPROCESSED)

        self.main_layout.addLayout(self.z_align_layout_pixel)

        self.export_layout = QHBoxLayout()

        self.export_czi_button = QPushButton("Export to CZI")
        self.export_czi_button.setToolTip("Export the final aligned Zarr output to CZI format.")
        self.export_czi_button.clicked.connect(self.export_to_czi)
        self.export_czi_button.setEnabled(False)
        self.export_layout.addWidget(self.export_czi_button)

        self.upload_wkn_button = QPushButton("Upload to Webknossos...")
        self.upload_wkn_button.setToolTip("Upload the final aligned Zarr output to Webknossos.")
        self.upload_wkn_button.clicked.connect(self.upload_to_webknossos)
        self.upload_wkn_button.setEnabled(False)
        self.export_layout.addWidget(self.upload_wkn_button)

        self.export_widget = QWidget()
        self.export_widget.setLayout(self.export_layout)
        self.main_layout.addWidget(self.export_widget)

        self.download_wkn_button = QPushButton("Download from webknossos...")
        self.download_wkn_button.setToolTip("Download a fine-aligned WebKnossos dataset as a tiled CZI.")
        self.download_wkn_button.clicked.connect(self.download_from_webknossos)
        self.main_layout.addWidget(self.download_wkn_button)

        layer_events = getattr(getattr(self.viewer, "layers", None), "events", None)
        if layer_events is not None:
            for name in ("inserted", "removed", "renamed"):
                emitter = getattr(layer_events, name, None)
                if emitter is not None:
                    emitter.connect(self._update_correction_controls)

        self.refresh_local_folder_tree()
        if settings_error:
            show_error(settings_error)

    def search_for_atlas_project(self, path: str|Path) -> list[Path]:
        from atlas.io import get_valid_slice_folders
        series_list, _ = get_valid_slice_folders(Path(path))
        return series_list

    def refresh_local_folder_tree(self, *, allow_import: bool = False) -> None:
        if self._is_processing or self._is_zalign_running() or self._is_export_running():
            show_error("Wait for the current operation to finish before refreshing the project.")
            return
        path_text = self.path_edit.text().strip()
        if not path_text:
            show_error("Please provide an atlas project folder path.")
            self.path_edit.setText(str(self.main_directory))
            return

        project_root = Path(path_text).expanduser()
        if not project_root.exists() or not project_root.is_dir():
            show_error("Invalid path. Please provide an existing folder.")
            self.path_edit.setText(str(self.main_directory))
            return

        try:
            from atlas.io import get_valid_slice_folders
            valid, failed = get_valid_slice_folders(project_root)
            # Failed records may omit the folder path; their presence still indicates raw data.
            has_raw = bool(valid or failed)
            watcher_mode = not has_raw and (project_root / "stitched_sections.zarr").is_dir()
            records = self._read_watcher_sections(project_root) if watcher_mode else {}
            series_folders = ([project_root / f"S_{n:04d}" for n in records]
                              if watcher_mode else valid)
        except Exception as exc:
            show_error(f"Could not read folder: {exc}")
            self.path_edit.setText(str(self.main_directory))
            return

        self.main_directory = project_root
        self._watcher_mode = watcher_mode
        self._watcher_sections = records
        self.refresh_btn.setVisible(watcher_mode)
        self.stitching_all_btn.setEnabled(not watcher_mode)
        self.stitching_one_btn.setEnabled(not watcher_mode)

        # Reset transient runtime statuses on project reload.
        self.pending_series.clear()
        self.ongoing_series.clear()
        self.failed_series.clear()

        self.local_folder_tree.populate_from_project(project_root, series_folders)
        self.update_series_status_indicators(project_root)
        if watcher_mode and allow_import:
            self._offer_watcher_alignment_import(project_root)
            self._update_project_leaf_and_zalign_controls(project_root)

    def _offer_watcher_alignment_import(self, root_folder: Path) -> None:
        source = root_folder / "z_alignment_results.pkl"
        target = root_folder / ALIGNMENT_RESULTS_PATH
        if not source.is_file() or target.exists():
            return
        answer = QMessageBox.question(
            self, "Import watcher alignment",
            "Import the watcher alignment results from this project?\n\n"
            "Pickle files can execute arbitrary code when opened. Only import this "
            "file if you trust its source:\n" + str(source),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._import_watcher_alignment(root_folder, source, target)
        except Exception as exc:
            show_error(f"Could not import watcher alignment results: {exc}")

    def _import_watcher_alignment(self, root_folder: Path, source: Path, target: Path) -> None:
        from atlas.alignment.utils import ROI

        frame = pd.read_pickle(source)
        required = {"store_path", "moving_section_number", "moving_z_index",
                    "moving_ROI", "cumulative_ROI", "current_shift", "down_scale"}
        if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique:
            raise ValueError("Expected a nonempty alignment table with unique columns")
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing alignment columns: {sorted(missing)}")
        group = self._open_watcher_store(root_folder)
        expected = {}
        for key in group["sections"]:
            attrs = group["sections"][key].attrs
            if attrs.get("status") == "complete":
                number = int(attrs["atlas_section_number"])
                if number in expected:
                    raise ValueError("Duplicate completed section in stitched store")
                expected[number] = attrs["z_index"]
        seen = set()
        for _, row in frame.iterrows():
            number, z_index = row["moving_section_number"], row["moving_z_index"]
            if (isinstance(number, (bool, np.bool_)) or not isinstance(number, (int, np.integer))
                    or isinstance(z_index, (bool, np.bool_)) or not isinstance(z_index, (int, np.integer))
                    or number not in expected or number in seen or z_index != expected[number]):
                raise ValueError("Section numbers or Z indices do not match completed stitched sections")
            seen.add(number)
            if not isinstance(row["moving_ROI"], ROI) or not isinstance(row["cumulative_ROI"], ROI):
                raise ValueError(f"Missing or invalid ROI for section {number}")
            for roi in (row["moving_ROI"], row["cumulative_ROI"]):
                if (any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in roi)
                        or roi.x1 <= roi.x0 or roi.y1 <= roi.y0):
                    raise ValueError(f"Invalid ROI bounds for section {number}")
            shift = row["current_shift"]
            if (not isinstance(shift, np.ndarray) or shift.shape != (2,)
                    or shift.dtype.kind not in "iuf" or not np.isfinite(shift).all()):
                raise ValueError(f"Invalid pairwise shift for section {number}")
            scale = row["down_scale"]
            if (not isinstance(scale, (int, float, np.integer, np.floating))
                    or not np.isfinite(scale) or scale < 1 or int(scale) != scale):
                raise ValueError(f"Invalid down_scale for section {number}")
        if seen != set(expected):
            raise ValueError("Alignment table does not cover every completed stitched section")
        # Correction uses the preview's zero-based plane numbers as row labels.
        frame = frame.reset_index(drop=True)
        frame["store_path"] = str((root_folder / "stitched_sections.zarr").resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=".watcher-alignment-", suffix=".pkl", dir=target.parent)
        os.close(fd)
        try:
            frame.to_pickle(temp_name)
            try:
                os.link(temp_name, target)
            except FileExistsError:
                return
        finally:
            os.unlink(temp_name)

    @staticmethod
    def _open_watcher_store(root_folder: Path):
        import zarr
        group = zarr.open_group(str(root_folder / "stitched_sections.zarr"), mode="r")
        if (group.attrs.get("atlas_stitched_store_version") != 1
                or group.attrs.get("axes") != ["z", "y", "x"]):
            raise ValueError("Unsupported stitched store format")
        return group

    def _read_watcher_sections(self, root_folder: Path):
        group = self._open_watcher_store(root_folder)
        records = {}
        if "sections" not in group:
            return records
        sections = group["sections"]
        for key in sections:
            attrs = sections[key].attrs
            if attrs.get("status") == "complete":
                records[int(attrs["atlas_section_number"])] = key
        return dict(sorted(records.items()))

    def _update_watcher_tree(self, root_folder: Path):
        root = self.local_folder_tree.topLevelItem(0)
        root.setText(0, f"{root_folder.name}: {len(self._watcher_sections)} completed sections available")
        root.setToolTip(0, f"Watcher output: {root_folder}\nRefresh to check for more completed sections.")
        for index in range(root.childCount()):
            item = root.child(index)
            item.setText(0, f"[PROCESSED] {Path(item.data(0, Qt.ItemDataRole.UserRole)).name}")
            item.setData(0, SERIES_STATUS_COLOR_ROLE, PROCESSED)
        self._update_project_leaf_and_zalign_controls(root_folder)
        self._update_display_button_state(root_folder)

    def _update_preview_control(self, root_folder: Path) -> None:
        has_preview = self._has_preview_ome_zarr_output(root_folder)
        has_alignment = self._has_z_alignment_results(root_folder)
        self.zalign_preview_button.setText("Preview" if has_preview or not has_alignment else "Render and Preview")
        self.zalign_preview_button.setEnabled(
            has_preview or (has_alignment and not self._is_processing)
        )

    def browse_for_atlas_project(self) -> None:
        if self._is_processing or self._is_zalign_running() or self._is_export_running():
            show_error("Wait for the current operation to finish before changing projects.")
            return
        preferences = QSettings("CCI-GU-Sweden", "napari-atlas-cci")
        remembered = preferences.value("last_project_folder", "", type=str)
        start = Path(remembered or self.path_edit.text().strip() or str(self.main_directory)).expanduser()
        while not start.is_dir() and start != start.parent:
            start = start.parent
        if not start.is_dir():
            start = Path.home()
        selected_dir = QFileDialog.getExistingDirectory(
            self, "Select Atlas Project Folder", str(start),
        )
        if not selected_dir:
            return

        preferences.setValue("last_project_folder", selected_dir)

        self.path_edit.setText(selected_dir)
        self.refresh_local_folder_tree(allow_import=True)

    def _series_identifier_candidates(self, series_folder: Path) -> list[str]:
        """Return likely identifiers for output files across naming conventions."""
        candidates: list[str] = [series_folder.name]

        try:
            extracted_id = str(extract_s_number(series_folder.name)).strip()
        except Exception:
            extracted_id = ""

        if extracted_id and extracted_id not in candidates:
            candidates.append(extracted_id)

        if extracted_id.startswith("S_"):
            numeric_part = extracted_id[2:]
            if numeric_part and numeric_part not in candidates:
                candidates.append(numeric_part)
        elif extracted_id:
            prefixed = f"S_{extracted_id}"
            if prefixed not in candidates:
                candidates.append(prefixed)

        match = re.match(r"^S_(\d+)", series_folder.name)
        if match:
            normalized_num = str(int(match.group(1)))
            normalized_prefixed = f"S_{normalized_num}"
            if normalized_num not in candidates:
                candidates.append(normalized_num)
            if normalized_prefixed not in candidates:
                candidates.append(normalized_prefixed)

        return candidates

    def _series_sort_key(self, tiff_path: Path) -> tuple[int, str]:
        """Sort stitched series files robustly across naming convention variants."""
        try:
            extracted = str(extract_s_number(tiff_path)).strip()
        except Exception:
            extracted = tiff_path.stem

        match = re.search(r"S_(\d+)", extracted)
        if match is None:
            match = re.search(r"S_(\d+)", tiff_path.stem)
        if match is None:
            match = re.search(r"(\d+)", extracted)
        if match is None:
            match = re.search(r"(\d+)", tiff_path.stem)

        if match is None:
            return (10**12, tiff_path.name.lower())

        return (int(match.group(1)), tiff_path.name.lower())

    def _has_expected_output_files(self, root_folder: Path, series_folder: Path) -> tuple[bool, bool, bool]:
        """Check if stitched TIFF, phaseCC CSV, and transforms JSON exist for a series."""
        ids = self._series_identifier_candidates(series_folder)

        has_tiff = False
        has_csv = False
        has_json = False

        for series_id in ids:
            if any(root_folder.glob(f"stitched_image_{series_id}.tif*")):
                has_tiff = True
            if any(root_folder.glob(f"phaseCC_stitching_{series_id}.csv")):
                has_csv = True
            if any(root_folder.glob(f"transforms_{series_id}.json")):
                has_json = True

            if has_tiff and has_csv and has_json:
                break

        return has_tiff, has_csv, has_json

    def _get_stitched_image_path(self, root_folder: Path, series_folder: Path) -> Path | None:
        """Return the stitched image path for a series if present."""
        ids = self._series_identifier_candidates(series_folder)
        for series_id in ids:
            tiff_path = root_folder.joinpath(f"stitched_image_{series_id}.tiff")
            if tiff_path.exists():
                return tiff_path

            tif_path = root_folder.joinpath(f"stitched_image_{series_id}.tif")
            if tif_path.exists():
                return tif_path

        return None

    def _read_master_sections(self, root_folder: Path):
        store_path = root_folder / "stitched_sections.zarr"
        if not store_path.is_dir():
            return {}
        from atlas.io import StitchedZarrStore
        store = StitchedZarrStore.open(store_path)
        records = {}
        for key in store.sections:
            record = store.sections[key].attrs
            source = record.get("source_folder")
            if record.get("status") == "complete" and source:
                records[Path(source).resolve()] = int(record["atlas_section_number"])
        return records

    def _master_store_section_number(self, root_folder: Path, series_folder: Path):
        records = self._master_sections_cache
        if records is None:
            records = self._read_master_sections(root_folder)
        return records.get(series_folder.resolve())

    def _evaluate_series_status(self, root_folder: Path, series_folder: Path) -> tuple[str, QColor]:
        if series_folder in self.ongoing_series:
            return "ONGOING", ONGOING
        if series_folder in self.pending_series:
            return "PENDING", PENDING
        if series_folder in self.failed_series:
            return "FAILED", FAILED

        if self._master_store_section_number(root_folder, series_folder) is not None:
            return "PROCESSED", PROCESSED

        has_tiff, has_csv, has_json = self._has_expected_output_files(root_folder, series_folder)
        has_any_output = has_tiff or has_csv or has_json

        has_mif = any(
            file.is_file() and file.suffix.lower() == ".ve-mif"
            for file in series_folder.iterdir()
        )
        has_raw_tif = any(
            file.is_file() and file.suffix.lower() in {".tif", ".tiff"}
            for file in series_folder.iterdir()
        )

        if has_tiff and has_csv and has_json:
            return "PROCESSED", PROCESSED
        if (has_csv or has_json) and not has_tiff:
            return "FAILED", FAILED
        if has_any_output:
            return "ONGOING", ONGOING
        if has_mif and has_raw_tif:
            return "UNPROCESSED", UNPROCESSED
        return "UNPROCESSED", UNPROCESSED

    def _is_series_processed(self, root_folder: Path, series_folder: Path) -> bool:
        if self._master_store_section_number(root_folder, series_folder) is not None:
            return True
        has_tiff, has_csv, has_json = self._has_expected_output_files(root_folder, series_folder)
        return has_tiff and has_csv and has_json

    def _has_z_alignment_results(self, root_folder: Path) -> bool:
        return (root_folder / ALIGNMENT_RESULTS_PATH).exists()

    def _aligned_zarr_path(self, root_folder: Path) -> Path:
        return root_folder.joinpath(f"{root_folder.name}_aligned.zarr")

    def _aligned_ome_zarr_path(self, root_folder: Path) -> Path:
        return root_folder.joinpath(f"{root_folder.name}_aligned.ome.zarr")

    def _preview_ome_zarr_path(self, root_folder: Path) -> Path:
        return root_folder / f"{_preview_layer_name(root_folder)}.ome.zarr"

    def _legacy_alignment_zarr_path(self, root_folder: Path) -> Path:
        return root_folder.joinpath(f"{root_folder.name}.zarr")

    def _has_preview_ome_zarr_output(self, root_folder: Path) -> bool:
        return self._preview_ome_zarr_path(root_folder).exists()

    def _has_aligned_zarr_output(self, root_folder: Path) -> bool:
        return self._aligned_zarr_path(root_folder).exists()

    def _has_aligned_ome_zarr_output(self, root_folder: Path) -> bool:
        return self._aligned_ome_zarr_path(root_folder).exists()

    def _all_series_processed(self, root_folder: Path) -> bool:
        root_item = self.local_folder_tree.topLevelItem(0)
        if root_item is None or root_item.childCount() == 0:
            return False

        for idx in range(root_item.childCount()):
            child_item = root_item.child(idx)
            series_path = self._get_series_path_from_item(child_item)
            if series_path is None:
                continue
            if not self._is_series_processed(root_folder, series_path):
                return False

        return True

    def _update_project_leaf_and_zalign_controls(self, root_folder: Path) -> None:
        if self._watcher_mode:
            self._update_alignment_controls(
                root_folder, initial_ready=bool(self._watcher_sections)
            )
            return

        root_item = self.local_folder_tree.topLevelItem(0)
        if root_item is None:
            return

        root_path_text = root_item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(root_path_text, str) and root_path_text:
            project_name = Path(root_path_text).name
        else:
            project_name = root_folder.name

        all_processed = self._all_series_processed(root_folder)
        has_zcalc = self._has_z_alignment_results(root_folder)
        has_preview_ome_zarr = self._has_preview_ome_zarr_output(root_folder)
        has_zarr = self._has_aligned_zarr_output(root_folder)
        has_aligned_ome_zarr = self._has_aligned_ome_zarr_output(root_folder)

        if has_zarr:
            project_state_label = "[Completed]"
            project_state_color = PROCESSED
            project_state_details = "Aligned Zarr output present"
        elif has_zcalc:
            project_state_label = "[Z-Calculation READY]"
            project_state_color = ONGOING
            project_state_details = "Z-shift results ready"
        elif all_processed:
            project_state_label = "[Z-Align READY]"
            project_state_color = PENDING
            project_state_details = "All 2D stitching outputs found"
        else:
            project_state_label = "[Waiting 2D Stitch]"
            project_state_color = UNPROCESSED
            project_state_details = "Waiting for all series to be processed"

        root_item.setText(0, f"{project_state_label} {project_name}")
        root_item.setForeground(0, QBrush(project_state_color))
        root_item.setToolTip(
            0,
            (
                f"{root_folder}\n"
                f"State: {project_state_label} {project_state_details}\n"
                f"all stitched: {'yes' if all_processed else 'no'}\n"
                f"z_alignment_results.pkl: {'yes' if has_zcalc else 'no'}\n"
                f"preview OME-Zarr output: {'yes' if has_preview_ome_zarr else 'no'}\n"
                f"aligned Zarr output: {'yes' if has_zarr else 'no'}\n"
                f"aligned OME-Zarr output: {'yes' if has_aligned_ome_zarr else 'no'}"
            ),
        )

        has_master = (root_folder / "stitched_sections.zarr").is_dir()
        self._update_alignment_controls(root_folder, initial_ready=all_processed or has_master)

    def _update_alignment_controls(self, root_folder: Path, *, initial_ready: bool) -> None:
        if self._is_zalign_running() or self._is_export_running():
            return

        has_zcalc = self._has_z_alignment_results(root_folder)
        has_preview = self._has_preview_ome_zarr_output(root_folder)
        has_zarr = self._has_aligned_zarr_output(root_folder)
        self.zalign_zshift_button.setEnabled(initial_ready and not self._is_processing)
        self._update_preview_control(root_folder)
        self._update_correction_controls()
        self.zalign_final_button.setEnabled(has_zcalc)
        self.zalign_view_button.setEnabled(self._has_aligned_ome_zarr_output(root_folder))
        self.export_czi_button.setEnabled(has_zarr)
        self.upload_wkn_button.setEnabled(has_zarr)
        self._update_zalign_status_from_outputs(
            all_processed=initial_ready,
            has_zcalc=has_zcalc,
            has_preview_ome_zarr=has_preview,
            has_zarr=has_zarr,
        )

    def _update_correction_controls(self, _event=None) -> None:
        root_folder = Path(self.main_directory)
        preview_name = _preview_layer_name(root_folder)
        can_correct = (
            not self._is_zalign_running()
            and not self._is_export_running()
            and self._has_z_alignment_results(root_folder)
            and any(
                isinstance(layer, Image) and layer.name == preview_name
                for layer in getattr(self.viewer, "layers", ())
            )
        )
        self.correction_Z_btn.setEnabled(can_correct)
        self.correction_apply_btn.setEnabled(can_correct)

    def update_series_status_indicators(self, root_folder: Path) -> None:
        """Update status labels/colors for each S_ child in the tree."""
        root_item = self.local_folder_tree.topLevelItem(0)
        if root_item is None:
            return

        if self._watcher_mode:
            self._update_watcher_tree(root_folder)
            return

        self._master_sections_cache = self._read_master_sections(root_folder)
        for idx in range(root_item.childCount()):
            child_item = root_item.child(idx)
            series_path = self._get_series_path_from_item(child_item)
            if series_path is None:
                continue
            status_text, status_color = self._evaluate_series_status(root_folder, series_path)

            child_item.setText(0, f"[{status_text}] {series_path.name}")
            child_item.setData(0, SERIES_STATUS_COLOR_ROLE, status_color)

            has_tiff, has_csv, has_json = self._has_expected_output_files(root_folder, series_path)
            master_section = self._master_store_section_number(root_folder, series_path)
            child_item.setToolTip(
                0,
                (
                    f"{series_path}\n"
                    f"Status: {status_text}\n"
                    f"master Zarr: {f'S_{master_section}' if master_section is not None else 'not complete'}\n"
                    f"stitched tiff: {'yes' if has_tiff else 'no'}\n"
                    f"phaseCC csv: {'yes' if has_csv else 'no'}\n"
                    f"transform json: {'yes' if has_json else 'no'}"
                ),
            )

        self._update_selected_series_visuals()
        self.local_folder_tree.viewport().update()
        self._update_project_leaf_and_zalign_controls(root_folder)
        self._update_display_button_state(root_folder)
        self._master_sections_cache = None

    def _start_processing_queue(self, series_paths: list[Path]) -> None:
        if self._is_processing:
            show_error("Processing is already running.")
            return

        if getattr(self, "_zalign_active_action", None) == "preview":
            show_error("Wait for preview rendering to finish before stitching.")
            return

        unique_series_paths: list[Path] = []
        for series_path in series_paths:
            normalized_path = Path(series_path)
            if normalized_path not in unique_series_paths:
                unique_series_paths.append(normalized_path)

        root_folder = Path(self.main_directory)
        to_process = [
            series_path
            for series_path in unique_series_paths
            if not self._is_series_processed(root_folder, series_path)
        ]

        if not to_process:
            show_error("Selected series are already processed. Nothing to run.")
            self.update_series_status_indicators(root_folder)
            return

        worker_count = self.settings.thread_count

        # Queue bookkeeping for status labels.
        self._is_processing = True
        self._processing_order = to_process
        self._stitch_events = SimpleQueue()
        self.pending_series = set(to_process)
        self.ongoing_series = set()
        self.failed_series.difference_update(set(to_process))

        self._series_futures = {series_path: Future() for series_path in to_process}
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._batch_future = self._executor.submit(
            self._run_stitching_batch, root_folder, to_process,
            worker_count, self._series_futures.copy(),
        )

        self.update_series_status_indicators(root_folder)
        self._status_timer.start()

    def _poll_processing_futures(self) -> None:
        if not self._series_futures:
            if self._batch_future.done():
                self._finish_processing_queue()
            return

        root_folder = Path(self.main_directory)
        done_series: list[Path] = []
        started = False
        while True:
            try:
                path = self._stitch_events.get_nowait()
            except Empty:
                break
            self.pending_series.discard(path)
            self.ongoing_series.add(path)
            started = True

        for series_path, future in self._series_futures.items():
            if not future.done():
                continue

            done_series.append(series_path)
            success = False
            error_message = ""
            try:
                success, error_message = future.result()
            except Exception as exc:
                success = False
                error_message = str(exc)

            self.ongoing_series.discard(series_path)
            self.pending_series.discard(series_path)
            if not success:
                self.failed_series.add(series_path)
                print(f"Series failed for {series_path.name}: {error_message}")

        for series_path in done_series:
            self._series_futures.pop(series_path, None)

        if done_series or started:
            self.update_series_status_indicators(root_folder)

        if not self._series_futures and self._batch_future.done():
            self._finish_processing_queue()

    def _finish_processing_queue(self) -> None:
        self._status_timer.stop()
        self.pending_series.clear()
        self.ongoing_series.clear()
        self._processing_order = []
        self._next_queue_index = 0
        self._series_futures.clear()

        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

        self._is_processing = False
        self.update_series_status_indicators(Path(self.main_directory))

    def _is_zalign_running(self) -> bool:
        return self._zalign_future is not None and not self._zalign_future.done()

    def _is_export_running(self) -> bool:
        return self._export_future is not None and not self._export_future.done()

    def _set_zalign_status(self, state: str, details: str, color: QColor) -> None:
        full_text = f"Z Align: [{state}] {details}"
        compact_details = textwrap.shorten(
            " ".join(str(details).split()),
            width=ZALIGN_STATUS_MAX_CHARS,
            placeholder="...",
        )
        self.zalign_status_label.setText(f"Z Align: [{state}] {compact_details}")
        self.zalign_status_label.setToolTip(full_text)
        self.zalign_status_label.setStyleSheet(f"color: {color.name()}; font-weight: 600;")

    def _update_zalign_status_from_outputs(
        self,
        all_processed: bool,
        has_zcalc: bool,
        has_preview_ome_zarr: bool,
        has_zarr: bool,
    ) -> None:
        if has_zarr:
            self._set_zalign_status("PROCESSED", "Final aligned Zarr complete", PROCESSED)
        elif has_preview_ome_zarr:
            self._set_zalign_status("PROCESSED", "Preview OME-Zarr complete; final alignment ready", PROCESSED)
        elif has_zcalc:
            self._set_zalign_status("PENDING", "Initial Z alignment complete; final alignment ready", PENDING)
        elif all_processed:
            self._set_zalign_status("PENDING", "Ready for initial Z alignment", PENDING)
        else:
            self._set_zalign_status("UNPROCESSED", "Waiting for 2D stitching outputs", UNPROCESSED)

    def _set_zalign_buttons_busy(self) -> None:
        self.zalign_zshift_button.setEnabled(False)
        self.zalign_preview_button.setEnabled(False)
        self.correction_Z_btn.setEnabled(False)
        self.correction_apply_btn.setEnabled(False)
        self.zalign_final_button.setEnabled(False)
        self.zalign_view_button.setEnabled(False)
        self.export_czi_button.setEnabled(False)
        self.upload_wkn_button.setEnabled(False)
        self.download_wkn_button.setEnabled(False)
        if self._zalign_active_action == "initial":
            self._set_zalign_status("ONGOING", "Running initial Z alignment", ONGOING)
        elif self._zalign_active_action == "preview":
            self._set_zalign_status("ONGOING", "Rendering alignment preview", ONGOING)
        elif self._zalign_active_action == "correction":
            self._set_zalign_status("ONGOING", "Applying manual Z correction preview", ONGOING)
        elif self._zalign_active_action == "final":
            self._set_zalign_status("ONGOING", "Running final Z alignment", ONGOING)
        self._zalign_spinner_index = 0
        self._update_zalign_busy_text()
        self._zalign_timer.start()

    def _reset_zalign_buttons_idle(self) -> None:
        self.download_wkn_button.setEnabled(not self._is_export_running())
        self._zalign_timer.stop()
        self.zalign_zshift_button.setText("Initial Z Alignment")
        self.zalign_preview_button.setText("Preview")
        self.correction_Z_btn.setText("Enter Correction")
        self.correction_apply_btn.setText("Apply Correction")
        self.zalign_final_button.setText("Final Z Alignment")
        self.zalign_view_button.setText("View")
        self._update_project_leaf_and_zalign_controls(Path(self.main_directory))

    def _set_export_buttons_busy(self) -> None:
        self.zalign_zshift_button.setEnabled(False)
        self.zalign_preview_button.setEnabled(False)
        self.correction_Z_btn.setEnabled(False)
        self.correction_apply_btn.setEnabled(False)
        self.zalign_final_button.setEnabled(False)
        self.zalign_view_button.setEnabled(False)
        self.export_czi_button.setEnabled(False)
        self.upload_wkn_button.setEnabled(False)
        self.download_wkn_button.setEnabled(False)
        self._export_spinner_index = 0
        self._update_export_busy_text()
        self._export_timer.start()

    def _update_export_busy_text(self) -> None:
        self.download_wkn_button.setText("Download from webknossos...")
        frame = self._zalign_spinner_frames[self._export_spinner_index]
        self._export_spinner_index = (self._export_spinner_index + 1) % len(self._zalign_spinner_frames)

        if self._export_active_action == "czi":
            self.export_czi_button.setText(f"Export to CZI {frame}")
            self.upload_wkn_button.setText("Upload to Webknossos...")
        elif self._export_active_action == "webknossos":
            self.export_czi_button.setText("Export to CZI")
            self.upload_wkn_button.setText(f"Upload to Webknossos {frame}")

        elif self._export_active_action == "webknossos_download":
            self.download_wkn_button.setText(f"Download from webknossos {frame}")

    def _reset_export_buttons_idle(self) -> None:
        self.download_wkn_button.setText("Download from webknossos...")
        self.download_wkn_button.setEnabled(not self._is_zalign_running())
        self._export_timer.stop()
        self.export_czi_button.setText("Export to CZI")
        self.upload_wkn_button.setText("Upload to Webknossos...")
        self._update_project_leaf_and_zalign_controls(Path(self.main_directory))

    def _poll_export_future(self) -> None:
        future = self._export_future
        if future is None:
            self._reset_export_buttons_idle()
            return

        if not future.done():
            self._update_export_busy_text()
            return

        action = self._export_active_action
        self._export_future = None
        self._export_active_action = None
        self._reset_export_buttons_idle()

        try:
            success, message, payload = future.result()
        except Exception as exc:
            show_error(f"Export task failed: {exc}")
            return

        if not success:
            show_error(message)
            return

        if action in {"czi", "webknossos_download"}:
            show_info(f"Saved CZI: {payload}")
        elif action == "webknossos":
            show_info(f"{message} {payload}")
            self.update_series_status_indicators(Path(self.main_directory))

    def open_options_dialog(self) -> None:
        dialog = OptionsDialog(self)
        dialog.set_values(self.settings.to_dict())
        dialog.options_applied.connect(self.apply_options)
        self._exec_dialog(dialog)

    def apply_options(self, values: dict) -> None:
        try:
            settings = PluginSettings.from_dict(self.settings.to_dict() | values)
        except ValueError as exc:
            show_error(f"Invalid Atlas CCI options: {exc}")
            return
        self.settings = settings
        self._save_user_settings()
        show_info("Atlas CCI options updated.")

    def _save_user_settings(self) -> None:
        try:
            save_settings(self.settings)
        except OSError as exc:
            show_error(f"Could not save Atlas CCI settings: {exc}")

    def _exec_dialog(self, dialog) -> int:
        if hasattr(dialog, "exec"):
            return int(dialog.exec())
        return int(dialog.exec_())

    def _update_zalign_busy_text(self) -> None:
        frame = self._zalign_spinner_frames[self._zalign_spinner_index]
        self._zalign_spinner_index = (self._zalign_spinner_index + 1) % len(self._zalign_spinner_frames)

        if self._zalign_active_action == "initial":
            self.zalign_zshift_button.setText(f"Initial Z Alignment {frame}")
            self.zalign_preview_button.setText("Preview")
            self.correction_apply_btn.setText("Apply Correction")
            self.zalign_final_button.setText("Final Z Alignment")
            self.zalign_view_button.setText("View")
        elif self._zalign_active_action == "preview":
            self.zalign_preview_button.setText(f"Rendering Preview {frame}")
        elif self._zalign_active_action == "correction":
            self.correction_apply_btn.setText(f"Apply Correction {frame}")
            self.zalign_zshift_button.setText("Initial Z Alignment")
            self.zalign_preview_button.setText("Preview")
            self.zalign_final_button.setText("Final Z Alignment")
            self.zalign_view_button.setText("View")
        elif self._zalign_active_action == "final":
            self.zalign_final_button.setText(f"Final Z Alignment {frame}")
            self.zalign_zshift_button.setText("Initial Z Alignment")
            self.zalign_preview_button.setText("Preview")
            self.correction_apply_btn.setText("Apply Correction")
            self.zalign_view_button.setText("View")

    def _poll_zalign_future(self) -> None:
        future = self._zalign_future
        if future is None:
            self._reset_zalign_buttons_idle()
            return

        if not future.done():
            self._update_zalign_busy_text()
            return

        action = self._zalign_active_action
        self._zalign_future = None
        self._zalign_active_action = None
        self._reset_zalign_buttons_idle()

        try:
            success, message, payload = future.result()
        except Exception as exc:
            self._set_zalign_status("FAILED", str(exc), FAILED)
            show_error(f"Z alignment task failed: {exc}")
            return

        if not success:
            self._set_zalign_status("FAILED", message, FAILED)
            show_error(message)
            return

        if action in {"initial", "preview", "correction", "final"}:
            typed_payload = cast(dict[str, str | float | list[str]], payload if isinstance(payload, dict) else {})
            if action == "initial":
                axial_value = typed_payload.get("Axial")
                axial_unit = typed_payload.get("Unit")
                value = typed_payload.get("Value")
                if isinstance(axial_value, float) and isinstance(value, float) and isinstance(axial_unit, str):
                    self.pixel_size = {
                        "Value": value,
                        "Axial": axial_value,
                        "Unit": axial_unit,
                    }
            finalize_success, finalize_message = self._finalize_apply_z_shifts(typed_payload)
            if not finalize_success:
                self._set_zalign_status("FAILED", finalize_message, FAILED)
                show_error(finalize_message)
                self.update_series_status_indicators(Path(self.main_directory))
                return
            self.update_series_status_indicators(Path(self.main_directory))
            if action == "initial":
                self._set_zalign_status("PROCESSED", "Initial Z alignment complete", PROCESSED)
            elif action == "preview":
                self._set_zalign_status("PROCESSED", "Preview Z alignment displayed", PROCESSED)
            elif action == "correction":
                points_count = int(cast(float, typed_payload.get("points_count", 0)))
                self._set_zalign_status(
                    "PROCESSED",
                    f"Manual correction preview applied from {points_count} point pairs",
                    PROCESSED,
                )
            else:
                self._set_zalign_status("PROCESSED", "Final Z alignment complete", PROCESSED)

    def _get_series_path_from_item(self, item) -> Path | None:
        """Read canonical series path from tree item data, with tooltip fallback."""
        data_path = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(data_path, str) and data_path:
            return Path(data_path)

        tooltip_text = item.toolTip(0)
        if tooltip_text:
            first_line = tooltip_text.splitlines()[0].strip()
            if first_line:
                return Path(first_line)

        return None

    def _on_tree_selection_changed(self, *_) -> None:
        root_folder = Path(self.main_directory)
        self._update_selected_series_visuals()
        self._update_display_button_state(root_folder)

    def _update_selected_series_visuals(self) -> None:
        root_item = self.local_folder_tree.topLevelItem(0)
        if root_item is None:
            return

        selected_paths = {
            selected_path
            for selected_item in self.local_folder_tree.selectedItems()
            if (selected_path := self._get_series_path_from_item(selected_item)) is not None
        }

        for idx in range(root_item.childCount()):
            child_item = root_item.child(idx)
            series_path = self._get_series_path_from_item(child_item)
            if series_path is None:
                continue

            is_selected = series_path in selected_paths
            status_color = child_item.data(0, SERIES_STATUS_COLOR_ROLE)
            if not isinstance(status_color, QColor):
                status_color = UNPROCESSED

            child_item.setForeground(
                0,
                QBrush(QColor("#FFFFFF") if is_selected else status_color),
            )
            child_item.setBackground(
                0,
                QBrush(SELECTED_SERIES_BACKGROUND) if is_selected else QBrush(),
            )
            font = child_item.font(0)
            font.setBold(is_selected)
            child_item.setFont(0, font)

        self.local_folder_tree.viewport().update()

    def _update_display_button_state(self, root_folder: Path) -> None:
        selected_items = self.local_folder_tree.selectedItems()
        if len(selected_items) != 1:
            self.display_btn.setEnabled(False)
            return

        series_path = self._get_series_path_from_item(selected_items[0])
        if self._watcher_mode:
            self.display_btn.setEnabled(
                series_path is not None and series_path.name.startswith("S_")
                and series_path.name[2:].isdigit()
                and int(series_path.name[2:]) in self._watcher_sections)
            return
        if series_path is None or not series_path.is_dir() or not series_path.name.startswith("S_"):
            self.display_btn.setEnabled(False)
            return

        enabled = self._is_series_processed(root_folder, series_path)
        self.display_btn.setEnabled(enabled)

    def _run_stitching_batch(self, root_folder, series_paths, worker_count, results):
        """Own the shared writer until all stitching threads have stopped."""
        from atlas.io import StitchedZarrStore, get_metadata_for_stitched_tif

        prepared = []
        try:
            for path in series_paths:
                try:
                    mifs = [p for p in path.iterdir()
                            if p.is_file() and p.suffix.lower() == ".ve-mif"]
                    if len(mifs) != 1:
                        raise ValueError(f"Expected one .ve-mif file; found {len(mifs)}")
                    metadata = get_metadata_for_stitched_tif(mifs[0])
                    prepared.append((path, mifs[0], metadata))
                except Exception as exc:
                    results[path].set_result((False, str(exc)))
            numbers = [item[2]["atlas_section_number"] for item in prepared]
            if len(numbers) != len(set(numbers)):
                raise ValueError("Selected folders contain duplicate ATLAS section numbers")
            if not prepared:
                return
            prepared.sort(key=lambda item: item[2]["atlas_section_number"])
            store = StitchedZarrStore.open(root_folder / "stitched_sections.zarr")
            with store.parallel_writer([item[2]["atlas_section_number"] for item in prepared]) as writer:
                with ThreadPoolExecutor(max_workers=worker_count) as workers:
                    futures = {}
                    for path, mif, metadata in prepared:
                        number = metadata["atlas_section_number"]
                        if writer.is_complete(number):
                            source = store.sections[f"S_{number}"].attrs.get("source_folder")
                            if source and Path(source).resolve() == path.resolve():
                                results[path].set_result((True, ""))
                            else:
                                results[path].set_result((False, f"S_{number} belongs to another source folder"))
                            continue
                        futures[workers.submit(self._process_one_series, path, mif, metadata, writer)] = (path, number)
                    for future in as_completed(futures):
                        path, number = futures[future]
                        try:
                            future.result()
                            results[path].set_result((True, ""))
                        except Exception as exc:
                            writer.mark_failed(number, exc)
                            results[path].set_result((False, str(exc)))
        except Exception as exc:
            for path in series_paths:
                if not results[path].done():
                    results[path].set_result((False, str(exc)))
            print(f"Stitching batch failed: {exc}", flush=True)

    def _process_one_series(self, series_path, mif_file, metadata, writer):
        from atlas.stitching import stitch_ATLAS_tiles

        number = metadata["atlas_section_number"]
        self._stitch_events.put(series_path)
        print(f"Stitching S_{number} into the master Zarr...", flush=True)
        image, tile_df, transforms, stitching_metadata = stitch_ATLAS_tiles(
            mif_file, buffer_microns=1, max_shift_pixels=self.settings.max_shift_pixels,
        )
        writer.write_section(
            atlas_section_number=number, image=np.flipud(image),
            acquisition_metadata=metadata, stitching_metadata=stitching_metadata,
            transforms={key: value.tolist() for key, value in transforms.items()},
            source_folder=series_path, tile_dataframe=tile_df,
        )
        print(f"Finished stitching S_{number} into the master Zarr.", flush=True)

    def process_selected_series(
        self,
        series_paths: list[Path] | None | bool = None,
    ) -> None:
        # QPushButton.clicked emits a bool; ignore it and use tree selection.
        if isinstance(series_paths, bool):
            series_paths = None

        if series_paths is None:
            selected_items = self.local_folder_tree.selectedItems()
            if not selected_items:
                show_error("Please select one or more series folders to process.")
                return

            series_paths = []
            for item in selected_items:
                item_path = self._get_series_path_from_item(item)
                if item_path is not None and item_path.is_dir() and item_path.name.startswith("S_"):
                    series_paths.append(item_path)

            if not series_paths:
                show_error("No valid S_ series folder selected.")
                return

        self._start_processing_queue([Path(series_path) for series_path in series_paths])

    def process_all_series(self) -> None:
        all_series_paths = self.search_for_atlas_project(self.main_directory)
        if not all_series_paths:
            show_error("No S_ series folders found in the atlas project.")
            return

        # Reuse selected-series workflow for consistent behavior.
        self._start_processing_queue(all_series_paths)


    def display_stitched_image(self) -> None:
        selected_items = self.local_folder_tree.selectedItems()
        if not selected_items:
            show_error("Please select one series folder to display.")
            return

        if len(selected_items) > 1:
            show_error("Please select only one series folder to display.")
            return

        series_path = self._get_series_path_from_item(selected_items[0])
        if self._watcher_mode:
            try:
                number = int(series_path.name[2:])
                group = self._open_watcher_store(Path(self.main_directory))
                record = group["sections"][self._watcher_sections[number]].attrs
                if record.get("status") != "complete":
                    raise ValueError("Section is no longer complete; refresh the project")
                height, width = record["shape_yx"]
                image = group["images"][record["z_index"], :height, :width]
                self.viewer.add_image(image, name=f"Stitched Image S_{number}")
            except Exception as exc:
                show_error(f"Failed to load stitched image: {exc}")
            return
        if series_path is None or not series_path.is_dir() or not series_path.name.startswith("S_"):
            show_error("Selected item is not a valid S_ series folder.")
            return

        root_folder = Path(self.main_directory)
        if not self._is_series_processed(root_folder, series_path):
            show_error("Only PROCESSED series can be displayed.")
            return

        series_id = extract_s_number(series_path.name)
        number = self._master_store_section_number(root_folder, series_path)
        if number is not None:
            try:
                from atlas.io import StitchedZarrStore
                store = StitchedZarrStore.open(root_folder / "stitched_sections.zarr")
                self.viewer.add_image(store.read_section(number), name=f"Stitched Image S_{number}")
            except Exception as exc:
                show_error(f"Failed to load stitched image: {exc}")
            return
        stitched_image_path = self._get_stitched_image_path(root_folder, series_path)

        if stitched_image_path is None:
            show_error(f"Stitched image file does not exist for {series_id}.")
            return

        try:
            stitched_image = tiff.imread(stitched_image_path)
            self.viewer.add_image(stitched_image, name=f"Stitched Image {series_id}")
        except Exception as e:
            show_error(f"Failed to load stitched image: {e}")

    def calculate_z_shifts(self, tif_list_sorted: list[Path], output_root: Path | None = None) -> None:
        from atlas.io import create_empty_folder
        from atlas.alignment import initialize_alignment_df, pairwise_alignment, calculate_cumulative_shifts

        output_path = (output_root if output_root is not None else Path(self.main_directory)).joinpath("alignment_results")
        create_empty_folder(output_path)
        z_align_df = initialize_alignment_df(tif_list_sorted, self.settings.downsampling_factor)
        z_align_df = pairwise_alignment(z_align_df)
        z_align_df = calculate_cumulative_shifts(z_align_df)

        z_align_df_path = output_path.joinpath("z_alignment_results.pkl")
        z_align_df.to_pickle(z_align_df_path)

    def _prepare_alignement_worker(
        self,
        root_folder: Path,
        axial_value: float,
        axial_unit: str,
    ) -> tuple[bool, str, dict[str, float | str] | None]:
        store_path = root_folder / "stitched_sections.zarr"
        if store_path.is_dir():
            from atlas.io import StitchedZarrStore
            from atlas.alignment import (
                initialize_zarr_alignment_df, pairwise_zarr_alignment,
                calculate_cumulative_shifts,
            )
            store = StitchedZarrStore.open(store_path)
            checks = store.validate_section_pixel_sizes()
            xy_um = float(checks["Acquisition Pixel size"]["first_um"])
            order = store.ordered_sections(order_path=root_folder / "correct_order.txt")
            frame = initialize_zarr_alignment_df(store, order, self.settings.downsampling_factor)
            frame = pairwise_zarr_alignment(frame, store)
            frame = calculate_cumulative_shifts(frame)
            alignment_path = root_folder / ALIGNMENT_RESULTS_PATH
            alignment_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_pickle(alignment_path)
            return True, "", {"Value": xy_um, "Axial": float(axial_value), "Unit": axial_unit}

        csv_candidates = sorted(root_folder.glob("phaseCC_stitching_*.csv"))
        if not csv_candidates:
            csv_candidates = sorted(root_folder.glob("PhaseCC_stitching_*.csv"))

        csv_file = csv_candidates[0] if csv_candidates else None
        if csv_file is None:
            return False, "No PhaseCC CSV file found in the main directory.", None

        df = pd.read_csv(csv_file)
        pix_size_micron = float(AXIAL_PIXEL_SIZE)
        if "PixelSizeMicron" in df.columns:
            pix_size_micron = float(df["PixelSizeMicron"].max())

        tif_list = [
            file
            for file in root_folder.iterdir()
            if file.is_file() and file.name.endswith(".tiff")
        ]
        if not tif_list:
            return False, "No stitched TIFF files found for Z shift calculation.", None

        tif_list_sorted = sorted(tif_list, key=self._series_sort_key)
        self.calculate_z_shifts(tif_list_sorted, output_root=root_folder)

        return (
            True,
            "",
            {
                "Value": pix_size_micron,
                "Axial": float(axial_value),
                "Unit": axial_unit,
            },
        )

    def _normalize_aligned_array_axes(self, data):
        from ._zarr_axes import normalize_aligned_array_to_zyx

        return normalize_aligned_array_to_zyx(data)

    def _replace_directory_from_temp(self, temp_path: Path, final_path: Path) -> None:
        import gc
        import shutil

        if final_path.exists():
            shutil.rmtree(final_path)

        gc.collect()
        try:
            temp_path.rename(final_path)
        except PermissionError:
            # Some Windows network drives reject directory renames immediately
            # after Zarr writes. Copying is slower but avoids that failure mode.
            shutil.copytree(temp_path, final_path)
            shutil.rmtree(temp_path)

    def _write_ome_zarr_pyramid_from_array(
        self,
        data,
        ome_zarr_path: Path,
        layer_name: str,
        atlas_metadata: dict | None = None,
    ) -> tuple[bool, str, dict[str, str | float | list[str]] | None]:
        from ome_zarr.io import parse_url
        from ome_zarr.scale import Scaler
        from ome_zarr.writer import write_image
        import dask
        import dask.array as da
        import shutil
        import zarr

        temp_ome_zarr_path = ome_zarr_path.with_name(f"{ome_zarr_path.name}.tmp")
        if temp_ome_zarr_path.exists():
            shutil.rmtree(temp_ome_zarr_path)

        location = parse_url(str(temp_ome_zarr_path), mode="w")
        if location is None:
            return False, f"Could not create OME-Zarr store at {temp_ome_zarr_path}", None

        root_group = zarr.group(store=location.store)

        # Bound chunk size and concurrency for large volumes.
        if data.ndim == 3:
            chunk_shape = (
                int(min(4, data.shape[0])),
                int(min(512, data.shape[1])),
                int(min(512, data.shape[2])),
            )
            axes = "zyx"
        else:
            chunk_shape = (
                int(min(1024, data.shape[0])),
                int(min(1024, data.shape[1])),
            )
            axes = "yx"

        data = da.asarray(data).rechunk(chunk_shape)
        print(
            f"Creating OME-Zarr pyramid at {temp_ome_zarr_path} "
            "(this may take several minutes)...",
            flush=True,
        )
        with dask.config.set(scheduler="threads", num_workers=2):
            write_image(
                image=data,
                group=root_group,
                scaler=Scaler(max_layer=4),
                axes=axes,
                storage_options={"chunks": chunk_shape},
            )

        if atlas_metadata is not None:
            root_group.attrs["AtlasMetadata"] = atlas_metadata

        print(f"Finished writing OME-Zarr pyramid: {temp_ome_zarr_path}", flush=True)

        attrs = dict(root_group.attrs)
        multiscales = attrs.get("multiscales", [])
                
        if not multiscales:
            ome_metadata = attrs.get("ome", {})
            if isinstance(ome_metadata, dict):
                multiscales = ome_metadata.get("multiscales", [])
        
        if not multiscales:
            return False, "OME-Zarr file was written, but no multiscales metadata was found.", None

        datasets = multiscales[0].get("datasets", [])
        if not datasets:
            return False, "OME-Zarr file was written, but no pyramid datasets were found.", None

        self._replace_directory_from_temp(temp_ome_zarr_path, ome_zarr_path)

        # A regular sample keeps display contrast estimation bounded in memory.
        print("Calculating display contrast from a sample...", flush=True)
        stride = max(1, int(np.ceil((data.size / 1_000_000) ** (1 / data.ndim))))
        stats_data = data[tuple(slice(None, None, stride) for _ in data.shape)].compute()
        if np.issubdtype(stats_data.dtype, np.integer):
            p_low, p_high = np.percentile(stats_data, [1.0, 99.8])
        else:
            p_low, p_high = np.percentile(stats_data, [0.5, 99.5])
        if float(p_low) == float(p_high):
            p_low = float(stats_data.min())
            p_high = float(stats_data.max())

        return (
            True,
            "",
            {
                "ome_zarr_path": str(ome_zarr_path),
                "dataset_paths": [dataset["path"] for dataset in datasets],
                "layer_name": layer_name,
                "p_low": float(p_low),
                "p_high": float(p_high),
            },
        )

    def _apply_z_shifts_preview_worker(
        self,
        root_folder: Path,
        *, alignment_path: Path | None = None,
        preview_path: Path | None = None,
        ome_path: Path | None = None,
    ) -> tuple[bool, str, dict[str, str | float | list[str]] | None]:
        from atlas.io import apply_alignment
        import shutil

        z_align_df_path = alignment_path or (root_folder / ALIGNMENT_RESULTS_PATH)
        if not z_align_df_path.exists():
            return False, "Z alignment results not found. Please run the alignment first.", None

        z_align_df = pd.read_pickle(z_align_df_path)
        print(z_align_df.head(5))

        settings = self.settings
        store_path = root_folder / "stitched_sections.zarr"
        if store_path.is_dir() and "store_path" in z_align_df.columns:
            from atlas.io import StitchedZarrStore, apply_alignment_zarr
            store = StitchedZarrStore.open(store_path)
            preview_path = preview_path or root_folder / "aligned_preview.zarr"
            zarr_array, _ = apply_alignment_zarr(
                z_align_df, store, output_path=preview_path,
                buffer_pixels=20, percentile_low=settings.percentile_low,
                percentile_high=settings.percentile_high, use_down_sample=True,
            )
        else:
            zarr_array, _ = apply_alignment(
                z_align_df, buffer_pixels=20,
                percentile_low=settings.percentile_low,
                percentile_high=settings.percentile_high, use_down_sample=True,
            )

        try:
            data, atlas_metadata = self._normalize_aligned_array_axes(zarr_array)
        except ValueError as exc:
            return False, str(exc), None

        success, message, payload = self._write_ome_zarr_pyramid_from_array(
            data,
            ome_path or self._preview_ome_zarr_path(root_folder),
            _preview_layer_name(root_folder),
            atlas_metadata,
        )
        legacy_zarr_path = self._legacy_alignment_zarr_path(root_folder)
        if legacy_zarr_path != store_path and legacy_zarr_path.exists():
            shutil.rmtree(legacy_zarr_path)

        return success, message, payload

    def _write_aligned_zarr_worker(
        self,
        root_folder: Path,
    ) -> tuple[bool, str, dict[str, str | float | list[str]] | None]:
        from atlas.io import apply_alignment
        import dask.array as da
        import shutil

        z_align_df_path = root_folder / ALIGNMENT_RESULTS_PATH
        if not z_align_df_path.exists():
            return False, "Z alignment results not found. Please run the alignment first.", None

        z_align_df = pd.read_pickle(z_align_df_path)
        print(z_align_df.head(5))

        settings = self.settings
        aligned_zarr_path = self._aligned_zarr_path(root_folder)
        temp_aligned_zarr_path = aligned_zarr_path.with_name(f"{aligned_zarr_path.name}.tmp")
        if temp_aligned_zarr_path.exists():
            shutil.rmtree(temp_aligned_zarr_path)

        import inspect

        alignment_kwargs = {
            "buffer_pixels": 20,
            "percentile_low": settings.percentile_low,
            "percentile_high": settings.percentile_high,
            "use_down_sample": False,
        }
        master_input = "store_path" in z_align_df.columns
        if master_input:
            from atlas.io import StitchedZarrStore, apply_alignment_zarr
            master_path = root_folder / "stitched_sections.zarr"
            if not master_path.is_dir():
                return False, "Master stitched Zarr store not found.", None
            store = StitchedZarrStore.open(master_path)
            print(f"Writing full-resolution aligned Zarr to {temp_aligned_zarr_path}...", flush=True)
            zarr_array, _ = apply_alignment_zarr(
                z_align_df, store, **alignment_kwargs, output_path=temp_aligned_zarr_path,
            )
            accepts_output_path = True
        else:
            accepts_output_path = "output_path" in inspect.signature(apply_alignment).parameters
        if not master_input and accepts_output_path:
            print(
                f"Writing full-resolution aligned Zarr to {temp_aligned_zarr_path}...",
                flush=True,
            )
            zarr_array, _ = apply_alignment(
                z_align_df,
                **alignment_kwargs,
                output_path=temp_aligned_zarr_path,
            )
        elif not master_input:
            # Older backend releases only write beside the source TIFFs.
            zarr_array, _ = apply_alignment(z_align_df, **alignment_kwargs)

        try:
            aligned_array, atlas_metadata = self._normalize_aligned_array_axes(
                zarr_array
            )
        except ValueError as exc:
            return False, str(exc), None

        original_metadata = getattr(zarr_array, "attrs", {}).get(
            "AtlasMetadata", {}
        )
        needs_reorder = original_metadata.get("axes") == "YXZ"
        if not accepts_output_path or needs_reorder:
            if accepts_output_path:
                normalized_temp_path = temp_aligned_zarr_path.with_name(
                    f"{temp_aligned_zarr_path.name}.zyx"
                )
                if normalized_temp_path.exists():
                    shutil.rmtree(normalized_temp_path)
                output_path = normalized_temp_path
            else:
                output_path = temp_aligned_zarr_path

            print(
                f"Writing normalized full-resolution ZYX data to {output_path}...",
                flush=True,
            )
            if isinstance(aligned_array, da.Array):
                dask_array = aligned_array
            else:
                chunks = getattr(aligned_array, "chunks", None)
                dask_array = da.from_array(
                    aligned_array, chunks=chunks or "auto"
                )
            da.to_zarr(dask_array, str(output_path), overwrite=True)
            from ._zarr_axes import write_atlas_metadata

            write_atlas_metadata(output_path, atlas_metadata)
            if accepts_output_path:
                del zarr_array, aligned_array, dask_array
                self._replace_directory_from_temp(
                    output_path, temp_aligned_zarr_path
                )
            else:
                del zarr_array, aligned_array, dask_array
        else:
            # The backend already wrote the declared ZYX metadata into this
            # direct output. Release its open array before finalizing the path.
            del zarr_array, aligned_array

        self._replace_directory_from_temp(temp_aligned_zarr_path, aligned_zarr_path)
        from ._zarr_axes import write_atlas_metadata_sidecar

        write_atlas_metadata_sidecar(aligned_zarr_path, atlas_metadata)
        temp_metadata_sidecar = temp_aligned_zarr_path.with_suffix(".json")
        if temp_metadata_sidecar != aligned_zarr_path.with_suffix(".json"):
            temp_metadata_sidecar.unlink(missing_ok=True)
        print(f"Finished writing aligned Zarr: {aligned_zarr_path}", flush=True)

        ome_success, ome_message, ome_payload = self._write_ome_zarr_pyramid_from_array(
            da.from_zarr(str(aligned_zarr_path)),
            self._aligned_ome_zarr_path(root_folder),
            root_folder.name,
            atlas_metadata,
        )
        if not ome_success or ome_payload is None:
            return False, ome_message, None

        legacy_zarr_path = self._legacy_alignment_zarr_path(root_folder)
        if legacy_zarr_path != root_folder / "stitched_sections.zarr" and legacy_zarr_path.exists():
            shutil.rmtree(legacy_zarr_path)

        print(f"Finished saving OME-Zarr: {self._aligned_ome_zarr_path(root_folder)}", flush=True)
        return True, "", {"aligned_zarr_path": str(aligned_zarr_path), **ome_payload}

    def _initial_alignement_worker(
        self,
        root_folder: Path,
        axial_value: float,
        axial_unit: str,
    ) -> tuple[bool, str, dict[str, str | float | list[str]] | None]:
        prepare_success, prepare_message, prepare_payload = self._prepare_alignement_worker(
            root_folder,
            axial_value,
            axial_unit,
        )
        if not prepare_success or prepare_payload is None:
            return False, prepare_message, None

        apply_success, apply_message, apply_payload = self._apply_z_shifts_preview_worker(root_folder)
        if not apply_success or apply_payload is None:
            return False, apply_message, None

        return True, "", {**prepare_payload, **apply_payload}

    def _apply_correction_worker(
        self,
        root_folder: Path,
        fixed_points: np.ndarray,
        moving_points: np.ndarray,
    ) -> tuple[bool, str, dict[str, str | float | list[str]] | None]:
        from atlas.alignment import correct_z_alignment_from_points
        import shutil

        z_align_df_path = root_folder / ALIGNMENT_RESULTS_PATH
        if not z_align_df_path.exists():
            return False, "Z alignment results not found. Please run initial Z alignment first.", None

        z_align_df = pd.read_pickle(z_align_df_path)
        corrected_z_align_df = correct_z_alignment_from_points(
            z_align_df,
            fixed_points,
            moving_points,
        )
        if corrected_z_align_df is None:
            corrected_z_align_df = z_align_df

        backup_path = z_align_df_path.with_name("z_alignment_results_before_manual_correction.pkl")
        if not backup_path.exists():
            shutil.copy2(z_align_df_path, backup_path)

        if "store_path" not in corrected_z_align_df.columns:
            corrected_z_align_df.to_pickle(z_align_df_path)
            success, message, payload = self._apply_z_shifts_preview_worker(root_folder)
            if not success or payload is None:
                return False, message, None
            return True, "", {**payload, "points_count": float(len(fixed_points))}

        # Stage every artifact separately. Rename old outputs aside while
        # finalizing so ordinary filesystem failures can be rolled back.
        staged_df = z_align_df_path.with_name("z_alignment_results.correction.tmp.pkl")
        staged_preview = root_folder / "aligned_preview.correction.zarr"
        final_preview = root_folder / "aligned_preview.zarr"
        final_ome = self._preview_ome_zarr_path(root_folder)
        staged_ome = final_ome.with_name(final_ome.name + ".correction")
        pairs = [(staged_preview, final_preview),
                 (staged_preview.with_suffix(".json"), final_preview.with_suffix(".json")),
                 (staged_ome, final_ome), (staged_df, z_align_df_path)]
        backups = []
        installed = []
        try:
            corrected_z_align_df.to_pickle(staged_df)
            success, message, payload = self._apply_z_shifts_preview_worker(
                root_folder, alignment_path=staged_df,
                preview_path=staged_preview, ome_path=staged_ome,
            )
            if not success or payload is None:
                return False, message, None
            for staged, final in pairs:
                backup = final.with_name(final.name + ".before_correction.tmp")
                if backup.exists():
                    raise FileExistsError(f"Recovery backup already exists: {backup}")
                if final.exists():
                    final.rename(backup)
                    backups.append((backup, final))
                staged.rename(final)
                installed.append(final)
            payload["ome_zarr_path"] = str(final_ome)
            for backup, _ in backups:
                try:
                    if backup.is_dir():
                        shutil.rmtree(backup)
                    else:
                        backup.unlink()
                except OSError as exc:
                    print(f"Could not remove correction backup {backup}: {exc}", flush=True)
            return True, "", {**payload, "points_count": float(len(fixed_points))}
        except Exception:
            for final in reversed(installed):
                if final.is_dir():
                    shutil.rmtree(final)
                elif final.exists():
                    final.unlink()
            for backup, final in reversed(backups):
                backup.rename(final)
            raise
        finally:
            for staged, _ in pairs:
                if staged.is_dir():
                    shutil.rmtree(staged)
                elif staged.exists():
                    staged.unlink()

    def _finalize_apply_z_shifts(self, payload: dict[str, str | float | list[str]]) -> tuple[bool, str]:
        try:
            ome_zarr_path = Path(str(payload.get("ome_zarr_path", "")))
            layer_name = str(payload.get("layer_name", "aligned"))
            p_low = float(str(payload.get("p_low", 0.0)))
            p_high = float(str(payload.get("p_high", 1.0)))

            return self.zarr_viewer.display_ome_zarr(
                ome_zarr_path,
                layer_name=layer_name,
                contrast_limits=(p_low, p_high),
            )
        except Exception as exc:
            return False, f"Failed to display aligned image: {exc}"
    
    def update_pixel_size_from_input(self) -> bool:
        try:
            new_pixel_size = float(self.axial_pixel_size_input.text())
            if not np.isfinite(new_pixel_size) or not 0.01 <= new_pixel_size <= 1000:
                raise ValueError("Pixel size must be between 0.01 and 1000.")
            self.pixel_size['Axial'] = new_pixel_size

            new_pixel_unit = self.axial_pixel_unit_dropdown.currentText()
            if new_pixel_unit not in AXIAL_PIXEL_SIZE_UNITS_OPTIONS:
                raise ValueError(f"Invalid pixel size unit: {new_pixel_unit}")

            settings = PluginSettings.from_dict(self.settings.to_dict() | {
                "axial_pixel_size": new_pixel_size,
                "axial_pixel_unit": new_pixel_unit,
            })
            if settings != self.settings:
                self.settings = settings
                self._save_user_settings()
            print(f"Updated axial pixel size to: {new_pixel_size} {new_pixel_unit}")
            return True
        except ValueError as e:
            show_error(f"Invalid pixel size input: {e}")
            return False

    def initial_alignement(self):
        if not self.update_pixel_size_from_input():
            return
        if self._is_zalign_running():
            show_error("A Z alignment task is already running.")
            return

        root_folder = Path(self.main_directory)
        if self._is_processing:
            show_error("Wait for stitching to finish before starting Z alignment.")
            return
        if not (root_folder / "stitched_sections.zarr").is_dir() and not self._all_series_processed(root_folder):
            show_error("Initial Z alignment requires all series to be processed first.")
            self.update_series_status_indicators(root_folder)
            return

        axial_value = self.pixel_size.get("Axial", AXIAL_PIXEL_SIZE)
        axial_unit = self.axial_pixel_unit_dropdown.currentText()

        self._remove_zalign_image_layer(use_downsample=True)

        self._zalign_active_action = "initial"
        self._zalign_future = self._zalign_executor.submit(
            self._initial_alignement_worker,
            root_folder,
            float(axial_value),
            str(axial_unit),
        )
        self._set_zalign_buttons_busy()

    def finalize_alignement(self):
        if not self.update_pixel_size_from_input():
            return
        if self._is_zalign_running():
            show_error("A Z alignment task is already running.")
            return

        root_folder = Path(self.main_directory)
        if not self._has_z_alignment_results(root_folder):
            show_error("Z alignment results not found. Please run initial Z alignment first.")
            self.update_series_status_indicators(root_folder)
            return

        self._remove_zalign_image_layer(use_downsample=False)

        self._zalign_active_action = "final"
        self._zalign_future = self._zalign_executor.submit(
            self._write_aligned_zarr_worker,
            root_folder,
        )
        self._set_zalign_buttons_busy()

    def _display_zalign_ome_zarr(self, use_downsample: bool) -> None:
        root_folder = Path(self.main_directory)
        ome_zarr_path = (
            self._preview_ome_zarr_path(root_folder)
            if use_downsample
            else self._aligned_ome_zarr_path(root_folder)
        )
        if not ome_zarr_path.exists():
            output_type = "preview" if use_downsample else "final"
            show_error(f"No {output_type} Z alignment OME-Zarr output found.")
            self.update_series_status_indicators(root_folder)
            return

        success, message = self.zarr_viewer.display_ome_zarr(
            ome_zarr_path,
            layer_name=(_preview_layer_name(root_folder) if use_downsample else root_folder.name),
        )
        if not success:
            self._set_zalign_status("FAILED", message, FAILED)
            show_error(message)
            return

        self.update_series_status_indicators(root_folder)
        if use_downsample:
            self._set_zalign_status("PROCESSED", "Preview Z alignment displayed", PROCESSED)
        else:
            self._set_zalign_status("PROCESSED", "Final Z alignment displayed", PROCESSED)

    def display_initial_zalign_preview(self) -> None:
        root_folder = Path(self.main_directory)
        if self._has_preview_ome_zarr_output(root_folder):
            self._display_zalign_ome_zarr(use_downsample=True)
            return
        if self._is_zalign_running() or self._is_export_running():
            show_error("Wait for the current operation to finish before rendering the preview.")
            return
        if not self._has_z_alignment_results(root_folder):
            show_error("Z alignment results not found. Please run initial Z alignment first.")
            self.update_series_status_indicators(root_folder)
            return
        if self._is_processing:
            show_error("Wait for stitching to finish before rendering the preview.")
            return
        self._zalign_active_action = "preview"
        self._zalign_future = self._zalign_executor.submit(self._apply_z_shifts_preview_worker, root_folder)
        self._set_zalign_buttons_busy()

    def display_final_zalign_output(self) -> None:
        self._display_zalign_ome_zarr(use_downsample=False)

    def _remove_zalign_image_layer(self, use_downsample: bool) -> None:
        import gc

        root_folder = Path(self.main_directory)
        layer_name = (_preview_layer_name(root_folder) if use_downsample else root_folder.name)
        if self.zarr_viewer.remove_layer(layer_name):
            QApplication.processEvents()
            gc.collect()

    def _get_viewer_layer(self, name: str):
        for layer in self.viewer.layers:
            if layer.name == name:
                return layer
        return None

    def _is_points_layer(self, layer) -> bool:
        return layer.__class__.__name__ == "Points"

    def _viewer_ndim(self) -> int:
        active_layer = self.viewer.layers.selection.active
        if active_layer is not None and hasattr(active_layer, "ndim"):
            return int(active_layer.ndim)
        return int(self.viewer.dims.ndim)

    def _set_points_layer_style(self, layer, face_color: str) -> None:
        layer.size = 8
        layer.symbol = "cross"
        layer.face_color = face_color
        if hasattr(layer, "border_color"):
            layer.border_color = "black"
        elif hasattr(layer, "edge_color"):
            layer.edge_color = "black"

    def _add_empty_points_layer(self, name: str, face_color: str):
        ndim = self._viewer_ndim()
        add_kwargs = {
            "data": None,
            "ndim": ndim,
            "name": name,
            "size": 8,
            "symbol": "cross",
            "face_color": face_color,
            "border_color": "black",
        }
        try:
            return self.viewer.add_points(**add_kwargs) # pyright: ignore[reportAttributeAccessIssue]
        except TypeError:
            add_kwargs.pop("border_color", None)
            add_kwargs["edge_color"] = "black"
            try:
                return self.viewer.add_points(**add_kwargs) # pyright: ignore[reportAttributeAccessIssue]
            except TypeError:
                add_kwargs.pop("edge_color", None)
                try:
                    return self.viewer.add_points(**add_kwargs) # pyright: ignore[reportAttributeAccessIssue]
                except TypeError:
                    add_kwargs.pop("ndim", None)
                    add_kwargs["data"] = np.empty((0, ndim))
                    return self.viewer.add_points(**add_kwargs) # pyright: ignore[reportAttributeAccessIssue]

    def _get_or_create_correction_points_layer(
        self,
        name: str,
        face_color: str,
    ):
        layer = self._get_viewer_layer(name)
        if layer is None:
            layer = self._add_empty_points_layer(name, face_color)
        else:
            if not self._is_points_layer(layer):
                show_error(f"Layer '{name}' already exists but is not a Points layer.")
                return None
            self._set_points_layer_style(layer, face_color)

        layer.mode = "add"
        return layer

    def create_correction_points_layer(self) -> None:
        root_folder = Path(self.main_directory)
        if not self._has_z_alignment_results(root_folder):
            show_error(
                "Z alignment results not found. Please run initial Z alignment first."
            )
            self.update_series_status_indicators(root_folder)
            return

        fixed_layer = self._get_or_create_correction_points_layer(
            "Reference Points",
            "red",
        )
        if fixed_layer is None:
            return
        trace_mode = self.correction_mode_dropdown.currentData() == "trace"
        if not trace_mode:
            moving_layer = self._get_or_create_correction_points_layer(
                "Adjustable Points",
                "blue",
            )
            if moving_layer is None:
                return

        self.viewer.layers.selection.active = fixed_layer
        self._set_zalign_status(
            "PENDING",
            (
                "Trace in Reference Points: one point per consecutive Z plane; gaps start new traces"
                if trace_mode
                else "Add matching points in Reference Points and Adjustable Points, then apply correction"
            ),
            PENDING,
        )

    @staticmethod
    def _trace_correction_pairs(
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Derive adjacent pairs in click order, without joining separate traces."""
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Trace points must have coordinates (z, y, x).")
        if len(points) < 2:
            raise ValueError(
                "Add at least two Reference Points on consecutive Z slices."
            )
        if not np.isfinite(points).all():
            raise ValueError(
                "Trace points must contain only finite coordinates."
            )
        z = points[:, 0]
        if np.any(z < 0) or np.any(z != np.floor(z)):
            raise ValueError(
                "Trace Z coordinates must be nonnegative integer slice indices."
            )
        slices, counts = np.unique(z, return_counts=True)
        duplicates = slices[counts > 1]
        if len(duplicates):
            planes = ", ".join(f"{value:g}" for value in duplicates)
            raise ValueError(
                f"Only one Reference Point per Z slice is allowed; duplicates on slice(s): {planes}."
            )
        isolated = z[~(np.isin(z - 1, z) | np.isin(z + 1, z))]
        if len(isolated):
            planes = ", ".join(f"{value:g}" for value in isolated)
            raise ValueError(
                f"Each trace needs at least two consecutive Z slices; isolated point(s) on slice(s): {planes}."
            )

        adjacent = np.abs(np.diff(z)) == 1
        # Every occupied adjacent slice must also be adjacent in click order.
        # This rejects shuffled/interleaved traces rather than silently sorting.
        if np.count_nonzero(adjacent) != np.count_nonzero(
            np.diff(slices) == 1
        ):
            raise ValueError(
                "Enter each trace in consecutive forward or backward Z order; use gaps between traces."
            )
        return points[:-1][adjacent].copy(), points[1:][adjacent].copy()

    def apply_correction(self) -> None:
        if self._is_zalign_running():
            show_error("A Z alignment task is already running.")
            return

        root_folder = Path(self.main_directory)
        if not self._has_z_alignment_results(root_folder):
            show_error(
                "Z alignment results not found. Please run initial Z alignment first."
            )
            self.update_series_status_indicators(root_folder)
            return

        fixed_layer = self._get_viewer_layer("Reference Points")
        if fixed_layer is None:
            show_error("Create Reference Points before applying correction.")
            return
        if not self._is_points_layer(fixed_layer):
            show_error("Reference Points must be a napari Points layer.")
            return

        moving_layer = None
        if self.correction_mode_dropdown.currentData() == "trace":
            try:
                fixed_points, moving_points = self._trace_correction_pairs(
                    fixed_layer.data
                )
            except ValueError as exc:
                show_error(str(exc))
                return
            # ATLAS expects fixed z = moving z - 1. Reverse backward-traced
            # pairs to that convention while retaining their trace order.
            backward = moving_points[:, 0] < fixed_points[:, 0]
            fixed_points[backward], moving_points[backward] = (
                moving_points[backward].copy(),
                fixed_points[backward].copy(),
            )
        else:
            moving_layer = self._get_viewer_layer("Adjustable Points")
            if moving_layer is None or not self._is_points_layer(moving_layer):
                show_error(
                    "Explicit pairs require an Adjustable Points layer."
                )
                return
            fixed_points = np.asarray(fixed_layer.data, dtype=float)
            moving_points = np.asarray(moving_layer.data, dtype=float)

        print(
            f"Applying correction with {len(fixed_points)} fixed points and {len(moving_points)} moving points."
        )
        print(f"Fixed points: {fixed_points}")
        print(f"Moving points: {moving_points}")

        if fixed_points.ndim != 2 or moving_points.ndim != 2:
            show_error(
                "Correction point layers must contain coordinate arrays."
            )
            return

        if len(fixed_points) == 0:
            show_error(
                "Add at least one correction point pair before applying correction."
            )
            return

        if len(fixed_points) != len(moving_points):
            show_error(
                "The number of points in the Reference Points layer and the Adjustable Points layer must be the same."
            )
            return

        if fixed_points.shape != moving_points.shape:
            show_error(
                "Reference and Adjustable correction points must have the same dimensionality."
            )
            return

        self._remove_zalign_image_layer(use_downsample=True)

        self._zalign_active_action = "correction"
        self._zalign_future = self._zalign_executor.submit(
            self._apply_correction_worker,
            root_folder,
            fixed_points.copy(),
            moving_points.copy(),
        )
        self._set_zalign_buttons_busy()

        # delete the points layers after applying correction
        self.viewer.layers.remove(fixed_layer)
        if moving_layer is not None:
            self.viewer.layers.remove(moving_layer)

    def _final_zarr_path(self) -> Path:
        return self._aligned_zarr_path(Path(self.main_directory))

    def _aligned_czi_path(self) -> Path:
        root_folder = Path(self.main_directory)
        return root_folder.joinpath(f"{root_folder.name}_aligned.czi")

    def _safe_output_name(self, name: str) -> str:
        return safe_dataset_name(name)

    def _webknossos_upload_name(self, dataset_name: str) -> str:
        return self._safe_output_name(dataset_name)

    def _webknossos_output_root(self, root_folder: Path) -> Path:
        return root_folder.joinpath("webknossos")

    def _webknossos_dataset_path(self, root_folder: Path, dataset_name: str) -> Path:
        safe_name = self._safe_output_name(dataset_name)
        return self._webknossos_output_root(root_folder).joinpath(safe_name)

    def _axial_pixel_size_nm(self) -> float:
        axial_value = float(self.axial_pixel_size_input.text())
        axial_unit = self.axial_pixel_unit_dropdown.currentText()
        if axial_unit == "µm":
            return axial_value * 1000.0
        return axial_value

    def _project_xy_pixel_size_um(self, root_folder: Path) -> float | None:
        """Read XY calibration from the master store, or legacy stitching CSV."""
        master_path = root_folder / "stitched_sections.zarr"
        if master_path.is_dir():
            from atlas.io import StitchedZarrStore
            checks = StitchedZarrStore.open(master_path).validate_section_pixel_sizes()
            value = float(checks["Acquisition Pixel size"]["first_um"])
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Invalid acquisition pixel size in master Zarr.")
            return value
        csv_files = sorted(root_folder.glob("[pP]haseCC_stitching_*.csv"))
        if csv_files:
            try:
                values = pd.to_numeric(pd.read_csv(csv_files[0])["PixelSizeMicron"], errors="coerce").dropna()
                if not values.empty and np.isfinite(values).all() and (values > 0).all():
                    return float(values.max())
            except (OSError, ValueError, KeyError, pd.errors.ParserError):
                pass
        return None

    def export_to_czi(self) -> None:
        if self._is_export_running():
            show_error("An export, upload, or download task is already running.")
            return

        zarr_path = self._final_zarr_path()
        if not zarr_path.exists():
            show_error("Final aligned Zarr output not found.")
            return

        if not self.update_pixel_size_from_input():
            return

        czi_path = self._aligned_czi_path()
        try:
            pix_xy = self._project_xy_pixel_size_um(Path(self.main_directory))
        except Exception as exc:
            show_error(f"Could not read XY pixel size: {exc}")
            return
        if pix_xy is None:
            pix_xy = float(self.pixel_size.get("Value", 1.0))
        pix_z = self._axial_pixel_size_nm() * 1e-3
        compression_options = self._czi_compression_options()

        self._export_active_action = "czi"
        self._export_future = self._export_executor.submit(
            self._export_to_czi_worker,
            zarr_path,
            czi_path,
            pix_xy,
            pix_z,
            compression_options,
        )
        self._set_export_buttons_busy()

    def _export_to_czi_worker(
        self,
        zarr_path: Path,
        czi_path: Path,
        pix_xy: float,
        pix_z: float,
        compression_options: str | None,
    ) -> tuple[bool, str, Path | None]:
        from ._czi import write_czi_volume
        import zarr

        try:
            source = zarr.open_array(str(zarr_path), mode="r")
            source, _ = self._normalize_aligned_array_axes(source)
            planes = ((z, 0, source[z, :, :]) for z in range(source.shape[0]))
            write_czi_volume(
                czi_path,
                planes,
                (pix_xy * 1e-6, pix_xy * 1e-6, pix_z * 1e-6),
                compression_options,
                {0: "White"},
                exist_ok=True,
            )
        except Exception as exc:
            return False, f"Failed to export CZI: {exc}", None
        return True, "CZI export complete.", czi_path

    def _czi_compression_options(self) -> str | None:
        return (
            f"zstd0:ExplicitLevel={self.settings.compression_level}"
            if self.settings.compression_enabled
            else None
        )

    def download_from_webknossos(self) -> None:
        if self._is_export_running() or self._is_zalign_running():
            show_error("Wait for the current processing task to finish.")
            return
        dialog = DownloadDataDialog(self, location=Path(self.main_directory))
        dialog.download_requested.connect(self._download_from_webknossos)
        self._exec_dialog(dialog)

    def _download_from_webknossos(self, values: dict) -> None:
        if self._is_export_running() or self._is_zalign_running():
            show_error("Wait for the current processing task to finish.")
            return
        self._export_active_action = "webknossos_download"
        self._export_future = self._export_executor.submit(
            self._download_from_webknossos_worker,
            values["wkn_token"],
            values["dataset_id"],
            Path(values["output_path"]),
            self._czi_compression_options(),
        )
        self._set_export_buttons_busy()

    def _download_from_webknossos_worker(
        self,
        token: str,
        dataset_id: str,
        output_path: Path,
        compression_options: str | None,
    ) -> tuple[bool, str, Path | None]:
        from ._czi import write_czi_volume
        from urllib.parse import urlsplit
        import time
        import webknossos as wk

        total_start = time.perf_counter()
        try:
            if output_path.exists():
                raise FileExistsError(f"Output already exists: {output_path}")
            if not token.strip() or not dataset_id.strip():
                raise ValueError(
                    "Provide a WebKnossos token and dataset ID or URL."
                )
            dataset_id = dataset_id.strip()
            url = urlsplit(dataset_id)
            is_url = bool(url.scheme)
            if is_url and (
                url.scheme not in ("https", "http") or not url.netloc
            ):
                raise ValueError(
                    "Provide a dataset ID or an HTTP(S) dataset URL."
                )
            server = (
                f"{url.scheme}://{url.netloc}"
                if is_url
                else "https://webknossos.org"
            )
            with wk.webknossos_context(token=token, url=server):
                kwargs = (
                    {"dataset_name_or_url": dataset_id}
                    if is_url
                    else {"dataset_id": dataset_id}
                )
                dataset = wk.Dataset.open_remote(**kwargs, read_only=True)
                if "color" not in dataset.layers:
                    raise ValueError(
                        f"Dataset has no color layer. Available layers: {list(dataset.layers)}"
                    )
                layer = dataset.get_layer("color")
                mag = layer.get_mag(wk.Mag(1))
                print(
                    f"Downloading {dataset.name} from bounding box {mag.bounding_box} to {output_path}"
                )
                with mag.get_buffered_slice_reader(
                    buffer_size=1, dimension="z"
                ) as reader:

                    def planes():
                        for z, data_cxy in enumerate(reader):
                            if data_cxy.ndim != 3:
                                raise ValueError(
                                    f"Expected CXY slices, got shape {data_cxy.shape}."
                                )
                            for channel in range(data_cxy.shape[0]):
                                yield z, channel, data_cxy[channel].T
                            del data_cxy

                    write_czi_volume(
                        output_path,
                        planes(),
                        tuple(float(v) * 1e-9 for v in dataset.voxel_size),
                        compression_options,
                        {
                            c: (
                                "color"
                                if layer.num_channels == 1
                                else f"color {c}"
                            )
                            for c in range(layer.num_channels)
                        },
                        document_name=dataset.name,
                    )
            print(
                f"WebKnossos total download/CZI time: {time.perf_counter() - total_start:.2f}s"
            )
            return True, "WebKnossos download complete.", output_path
        except Exception as exc:
            return (
                False,
                f"Failed to download WebKnossos dataset as CZI: {exc}",
                None,
            )

    def upload_to_webknossos(self) -> None:
        if self._is_export_running():
            show_error("An export, upload, or download task is already running.")
            return

        final_zarr_path = self._final_zarr_path()
        if not final_zarr_path.exists():
            show_error("Final aligned Zarr output not found.")
            return

        if not self.update_pixel_size_from_input():
            return

        try:
            xy_um = self._project_xy_pixel_size_um(Path(self.main_directory))
        except Exception as exc:
            show_error(f"Could not read XY pixel size: {exc}")
            return
        xy_nm = round(xy_um * 1000, 6) if xy_um is not None else None

        dialog = UploadDataDialog(self)
        dialog.set_values(
            {
                "xy_pixel_size": xy_nm,
                "z_pixel_size": self._axial_pixel_size_nm(),
                "dataset_name": Path(self.main_directory).name,
                "wkn_token": "",
            }
        )
        dialog.upload_requested.connect(self._upload_to_webknossos)
        self._exec_dialog(dialog)

    def _upload_to_webknossos(self, values: dict) -> None:
        if self._is_export_running():
            show_error("An export, upload, or download task is already running.")
            return

        token = str(values.get("wkn_token", "")).strip()
        dataset_name = str(values.get("dataset_name", "")).strip()
        if not token:
            show_error("Please provide a WebKnossos API token.")
            return
        if not dataset_name:
            show_error("Please provide a WebKnossos dataset name.")
            return

        final_zarr_path = self._final_zarr_path()
        if not final_zarr_path.exists():
            show_error("Final aligned Zarr output not found.")
            return

        xy_nm = float(values["xy_pixel_size"])
        voxel_size = (
            xy_nm,
            xy_nm,
            float(values.get("z_pixel_size", self._axial_pixel_size_nm())),
        )

        self._export_active_action = "webknossos"
        self._export_future = self._export_executor.submit(
            self._upload_to_webknossos_worker,
            final_zarr_path,
            Path(self.main_directory),
            token,
            dataset_name,
            voxel_size,
            bool(values.get("delete_local_dataset", True)),
        )
        self._set_export_buttons_busy()

    def _upload_to_webknossos_worker(
        self,
        final_zarr_path: Path,
        root_folder: Path,
        token: str,
        dataset_name: str,
        voxel_size: tuple[float, float, float],
        delete_local_dataset: bool = False,
    ) -> tuple[bool, str, str | None]:
        import time
        import shutil
        from webknossos import webknossos_context
        from ._webknossos import prepare_dataset

        try:
            total_start = time.perf_counter()
            upload_dataset_name = self._webknossos_upload_name(dataset_name)
            dataset_path = self._webknossos_dataset_path(root_folder, dataset_name)
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset = prepare_dataset(
                final_zarr_path, dataset_path, upload_dataset_name, voxel_size
            )
            with webknossos_context(token=token):
                print(f"Uploading {upload_dataset_name} from {dataset_path}...", flush=True)
                remote_dataset = dataset.upload(new_dataset_name=upload_dataset_name)
                print(f"WebKnossos total build/upload time: {time.perf_counter() - total_start:.2f}s")
                remote_url = str(remote_dataset.url)
                print(f"Successfully uploaded {remote_url}")
                message = "WebKnossos upload complete."
                if delete_local_dataset:
                    try:
                        shutil.rmtree(dataset_path)
                        print(f"Removed local WebKnossos dataset: {dataset_path}", flush=True)
                        try:
                            dataset_path.parent.rmdir()
                        except OSError:
                            # Keep the parent when it contains other datasets.
                            pass
                        message += " Local WebKnossos dataset deleted."
                    except OSError as exc:
                        message += f" Could not delete local dataset {dataset_path}: {exc}"
                        print(message, flush=True)
                return True, message, remote_url
        except Exception as exc:
            return False, f"Failed to upload to WebKnossos: {exc}", None
