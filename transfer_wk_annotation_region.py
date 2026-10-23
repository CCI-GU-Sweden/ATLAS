"""Download an annotation's image region and optionally upload it elsewhere.

The annotation must belong to the fine-aligned output dataset. Configure the
inputs below, then run this script with atlas-env. No annotations are uploaded.
"""

import json
import os
from pathlib import Path

import webknossos as wk
from cluster_tools import get_executor


# User inputs. Tokens may be set here or through the environment variables.
ANNOTATION_ID = "6ad8bc086d9958fcd0aa68b5"
SOURCE_TOKEN = os.environ.get("WK_SOURCE_TOKEN", "")
TARGET_TOKEN = os.environ.get("WK_TARGET_TOKEN", "")
SOURCE_URL = "https://webknossos.org"
TARGET_URL = "https://webknossos.org"
MAG = "32-32-8"
BUILD_PYRAMID = True
PYRAMID_COARSEST_MAG = "64"  # Must be coarser than MAG.
PYRAMID_JOBS = 4

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


def magnification_signature(dataset):
    """Record which resolutions an upload contains, for each layer."""
    return {
        name: sorted(str(mag) for mag in layer.mags)
        for name, layer in dataset.layers.items()
    }


def ensure_pyramid(dataset, output_dir):
    """Build coarser levels, including when reusing an older download.

    A completion record is written only after all downsampling finishes.
    Interrupted builds are regenerated from the downloaded resolution, so
    partially written magnifications are not treated as complete inputs.
    """
    marker = Path(output_dir) / ".atlas-transfer.json"
    record = json.loads(marker.read_text())
    base_mag = wk.Mag(record["download"]["mag"])
    coarsest_mag = wk.Mag(PYRAMID_COARSEST_MAG)
    if (
        coarsest_mag.max_dim <= base_mag.max_dim
        or any(a < b for a, b in zip(coarsest_mag.to_vec3_int(), base_mag.to_vec3_int()))
    ):
        raise ValueError("PYRAMID_COARSEST_MAG must be coarser than the downloaded MAG.")
    settings = {
        "from_mag": str(base_mag),
        "coarsest_mag": str(coarsest_mag),
        "sampling_mode": "anisotropic",
        "interpolation_mode": "default",
    }
    expected = {**settings, "magnifications": magnification_signature(dataset)}
    if record.get("pyramid") == expected:
        print(f"Reusing completed pyramid: {expected['magnifications']}")
        return

    record.pop("pyramid", None)
    marker.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Building compressed pyramid from {base_mag} up to {coarsest_mag}")
    with get_executor("multiprocessing", max_workers=PYRAMID_JOBS) as executor:
        for layer in dataset.layers.values():
            layer.downsample(
                from_mag=base_mag,
                coarsest_mag=coarsest_mag,
                sampling_mode=wk.SamplingModes.ANISOTROPIC,
                compress=True,
                allow_overwrite=True,
                executor=executor,
            )
    record["pyramid"] = {**settings, "magnifications": magnification_signature(dataset)}
    marker.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Pyramid complete: {record['pyramid']['magnifications']}")


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

    if BUILD_PYRAMID:
        ensure_pyramid(dataset, OUTPUT_DIR)

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
        magnifications = magnification_signature(dataset)
        previous_upload = record.get("upload", {})
        if (
            previous_upload.get("destination") == destination
            and previous_upload.get("magnifications") == magnifications
        ):
            print(f"Already uploaded: {record['upload']['dataset_url']}")
            print(f"Local download retained at: {Path(OUTPUT_DIR).resolve()}")
            return
        if previous_upload:
            print(f"Previous upload retained at: {previous_upload['dataset_url']}")
            print("Creating a new uploaded copy with the current magnifications.")
        print(f"Uploading {target_dataset_name} to organization {organization}")
        uploaded = dataset.upload(new_dataset_name=target_dataset_name)
        record["upload"] = {
            "destination": destination,
            "dataset_url": uploaded.url,
            "magnifications": magnifications,
        }
        marker.write_text(json.dumps(record, indent=2) + "\n")
        print(f"Upload complete: {uploaded.url}")
        print(f"Local download retained at: {Path(OUTPUT_DIR).resolve()}")


if __name__ == "__main__":
    main(do_upload=DO_UPLOAD)
