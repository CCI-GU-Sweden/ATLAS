"""Validated user preferences, stored separately from project data."""

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from qtpy.QtCore import QStandardPaths


@dataclass(frozen=True)
class PluginSettings:
    thread_count: int = 4
    max_shift_pixels: int = 500
    downsampling_factor: int = 10
    compression_enabled: bool = True
    compression_level: int = 3
    percentile_low: float = 1.0
    percentile_high: float = 99.95
    axial_pixel_size: float = 0.3
    axial_pixel_unit: str = "µm"

    @classmethod
    def from_dict(cls, values: dict) -> "PluginSettings":
        if not isinstance(values, dict):
            raise ValueError("Settings must be a JSON object.")
        data = asdict(cls())
        data.update(
            {key: value for key, value in values.items() if key in data}
        )
        for key, minimum, maximum in (
            ("thread_count", 1, 100),
            ("max_shift_pixels", 0, 1000),
            ("downsampling_factor", 1, 100),
            ("compression_level", 0, 22),
        ):
            if (
                type(data[key]) is not int
                or not minimum <= data[key] <= maximum
            ):
                raise ValueError(
                    f"{key} must be an integer from {minimum} to {maximum}."
                )
        if type(data["compression_enabled"]) is not bool:
            raise ValueError("compression_enabled must be true or false.")
        for key in ("percentile_low", "percentile_high", "axial_pixel_size"):
            if type(data[key]) not in (int, float) or not math.isfinite(
                data[key]
            ):
                raise ValueError(f"{key} must be a finite number.")
        if not 0 <= data["percentile_low"] < data["percentile_high"] <= 100:
            raise ValueError(
                "Percentiles must satisfy 0 ≤ lower < upper ≤ 100."
            )
        if not 0.01 <= data["axial_pixel_size"] <= 1000:
            raise ValueError("Axial pixel size must be between 0.01 and 1000.")
        if data["axial_pixel_unit"] not in ("nm", "µm"):
            raise ValueError("Axial pixel unit must be nm or µm.")
        # Preserve the existing CPU limit and save the effective value.
        data["thread_count"] = min(
            data["thread_count"], max(1, (os.cpu_count() or 2) - 1)
        )
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)


def settings_path() -> Path:
    root = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.GenericConfigLocation
    )
    return Path(root) / "napari-atlas-cci" / "settings.json"


def load_settings(path: Path | None = None) -> PluginSettings:
    path = settings_path() if path is None else path
    if not path.exists():
        return PluginSettings.from_dict({})
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("Unsupported settings schema version (expected 1).")
    if "settings" not in document:
        raise ValueError("The settings file has no settings object.")
    return PluginSettings.from_dict(document["settings"])


def save_settings(settings: PluginSettings, path: Path | None = None) -> None:
    path = settings_path() if path is None else path
    # Only the explicit preference fields are serialized; upload tokens and
    # dataset/project paths are never included.
    document = {"schema_version": 1, "settings": settings.to_dict()}
    path.parent.mkdir(parents=True, exist_ok=True)
    from qtpy.QtCore import QIODevice, QSaveFile

    output = QSaveFile(str(path))
    if not output.open(QIODevice.OpenModeFlag.WriteOnly):
        raise OSError(output.errorString())
    payload = (
        json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    if output.write(payload) != len(payload) or not output.commit():
        raise OSError(output.errorString())
