"""Download an annotation's image region and optionally upload it elsewhere.

The annotation must belong to the fine-aligned output dataset. Configure the
inputs below, then run this script with atlas-env. No annotations are uploaded.
"""

import json
import os
from pathlib import Path

import webknossos as wk


# User inputs. Tokens may be set here or through the environment variables.
ANNOTATION_ID = "6ad8bc086d9958fcd0aa68b5"
SOURCE_TOKEN = os.environ.get("WK_SOURCE_TOKEN", "")
TARGET_TOKEN = os.environ.get("WK_TARGET_TOKEN", "")
SOURCE_URL = "https://webknossos.org"
TARGET_URL = "https://webknossos.org"
MAG = "32-32-8"

# None selects the only box automatically. Set a zero-based index if several
# boxes are listed. Coordinates stay in full-resolution (mag-1) voxel units.
BOUNDING_BOX_INDEX = None
LAYER_NAMES = None  # None downloads all image/color layers, excluding segmentation.
OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "private-data" / "wk-transfer" / f"{ANNOTATION_ID}_mag{MAG}"
)
TARGET_DATASET_NAME = None  # None uses <source dataset name>_mag<MAG>.
TARGET_ORGANIZATION_ID = ""  # Optional: check against the target token's organization.
DO_UPLOAD = True  # Set False to download without uploading.


def select_bounding_box(annotation, index=None):
    """Select a user-defined box without silently choosing between several."""
    boxes = annotation.user_bounding_boxes
    if not boxes:
        raise ValueError("The annotation has no user-defined bounding boxes.")
    for number, box in enumerate(boxes):
        print(f"Bounding box {number}: {box}")
    if index is None:
        if len(boxes) != 1:
            raise ValueError("Set BOUNDING_BOX_INDEX to one of the listed indices.")
        index = 0
    if not 0 <= index < len(boxes):
        raise ValueError(f"Bounding box index {index} is out of range.")
    box = boxes[index]
    if set(box.axes) != {"x", "y", "z"}:
        raise ValueError("This script supports 3D bounding boxes only.")
    bbox = wk.BoundingBox(box.topleft_xyz, box.size_xyz)
    if any(value <= 0 for value in bbox.size):
        raise ValueError("The bounding box must have positive size on every axis.")
    return bbox


def download_subset(annotation, remote, output_dir, mag, layer_names=None, box_index=None):
    """Reuse a completed matching download, never an incomplete directory."""
    bbox = select_bounding_box(annotation, box_index)
    layers = list(layer_names) if layer_names is not None else [
        name for name, layer in remote.layers.items()
        if layer.category == wk.COLOR_CATEGORY
    ]
    if not layers:
        raise ValueError("No image layers selected for download.")
    for name in layers:
        if name not in remote.layers:
            raise ValueError(f"Layer {name!r} does not exist in the source dataset.")
        layer = remote.layers[name]
        if mag not in layer.mags:
            raise ValueError(f"Layer {name!r} has no mag {mag}; available: {list(layer.mags)}")
        if layer.coordinate_transformations:
            raise ValueError(
                f"Layer {name!r} uses display transformations. Use a materialized "
                "aligned dataset so the bounding box refers to downloaded voxels."
            )

    request = {
        "source_url": SOURCE_URL,
        "annotation_id": ANNOTATION_ID,
        "dataset_id": remote.dataset_id,
        "topleft_xyz": list(bbox.topleft),
        "size_xyz": list(bbox.size),
        "layers": layers,
        "mag": str(mag),
    }
    output_dir = Path(output_dir)
    marker = output_dir / ".atlas-transfer.json"
    if output_dir.exists():
        if not marker.is_file():
            raise ValueError(
                f"{output_dir} exists without a completed download record. "
                "Choose a new OUTPUT_DIR or move the incomplete folder aside."
            )
        if json.loads(marker.read_text())["download"] != request:
            raise ValueError("OUTPUT_DIR belongs to a different download; choose a new path.")
        print(f"Reusing completed download: {output_dir}")
        return wk.Dataset.open(output_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Source dataset: {remote.name} ({remote.dataset_id})")
    print(f"Downloading layers {layers}, mag {mag}, bounding box {bbox}")
    print(f"Local folder: {output_dir}")
    dataset = remote.download(
        path=output_dir,
        bounding_box=bbox,
        layers=layers,
        mags=[mag],
        data_format=wk.DataFormat.Zarr3,
    )
    # Only written after the SDK finishes: a failed upload can reuse this copy.
    marker.write_text(json.dumps({"download": request}, indent=2) + "\n")
    return dataset


def main(do_upload=False):
    if not SOURCE_TOKEN:
        raise ValueError("Set SOURCE_TOKEN or the WK_SOURCE_TOKEN environment variable.")
    if do_upload and not TARGET_TOKEN:
        raise ValueError("Set TARGET_TOKEN or the WK_TARGET_TOKEN environment variable.")

    with wk.webknossos_context(token=SOURCE_TOKEN, url=SOURCE_URL):
        annotation = wk.Annotation.download(ANNOTATION_ID, skip_volume_data=True)
        remote = annotation.get_remote_base_dataset(webknossos_url=SOURCE_URL)
        target_dataset_name = TARGET_DATASET_NAME or f"{remote.name}_mag{MAG}"
        dataset = download_subset(
            annotation, remote, OUTPUT_DIR, wk.Mag(MAG),
            layer_names=LAYER_NAMES, box_index=BOUNDING_BOX_INDEX,
        )

    if not do_upload:
        print("Download complete. Set DO_UPLOAD = True to upload this local copy.")
        print(f"Local download retained at: {Path(OUTPUT_DIR).resolve()}")
        return

    with wk.webknossos_context(token=TARGET_TOKEN, url=TARGET_URL):
        organization = wk.User.get_current_user().organization_id
        if TARGET_ORGANIZATION_ID and organization != TARGET_ORGANIZATION_ID:
            raise ValueError("The target token belongs to a different organization than configured.")
        marker = Path(OUTPUT_DIR) / ".atlas-transfer.json"
        record = json.loads(marker.read_text())
        destination = {
            "url": TARGET_URL, "organization_id": organization,
            "name": target_dataset_name,
        }
        if record.get("upload", {}).get("destination") == destination:
            print(f"Already uploaded: {record['upload']['dataset_url']}")
            print(f"Local download retained at: {Path(OUTPUT_DIR).resolve()}")
            return
        print(f"Uploading {target_dataset_name} to organization {organization}")
        uploaded = dataset.upload(new_dataset_name=target_dataset_name)
        record["upload"] = {"destination": destination, "dataset_url": uploaded.url}
        marker.write_text(json.dumps(record, indent=2) + "\n")
        print(f"Upload complete: {uploaded.url}")
        print(f"Local download retained at: {Path(OUTPUT_DIR).resolve()}")


if __name__ == "__main__":
    main(do_upload=DO_UPLOAD)
