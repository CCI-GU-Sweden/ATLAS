import os
from pathlib import Path

from webknossos import COLOR_CATEGORY, Dataset, webknossos_context
from webknossos.dataset.sampling_modes import SamplingModes
from webknossos.geometry.mag import Mag

name_no_extension = "your_dataset"
data_dir = Path("/path/to/czi-folder")
xy_pix_nm = 10
z_pix_nm = 75

INPUT_DIR = data_dir / f"{name_no_extension}.czi"
OUTPUT_DIR = Path("/path/to/output-dataset")
TOKEN = os.environ.get("WK_TARGET_TOKEN", "")
MY_SET_NAME = name_no_extension.replace(" ", "_")

def main(do_upload=False) -> None:
    if do_upload and not TOKEN:
        raise SystemExit("Set WK_TARGET_TOKEN before uploading to WEBKNOSSOS.")
    with webknossos_context(token=TOKEN):
            
        """Convert czi file to a WEBKNOSSOS dataset."""
        if (OUTPUT_DIR / "datasource-properties.json").exists():
            print(f"Opening existing dataset at {OUTPUT_DIR}")
            dataset = Dataset.open(OUTPUT_DIR)
        else:
            dataset = Dataset.from_images(
                input_path=INPUT_DIR,
                output_path=OUTPUT_DIR,
                voxel_size=(xy_pix_nm, xy_pix_nm, z_pix_nm),
                name=MY_SET_NAME,
                layer_category=COLOR_CATEGORY,
                compress=True,
            )

            dataset.downsample(
                sampling_mode=SamplingModes.ANISOTROPIC,
                coarsest_mag=Mag(32),
            )

        if do_upload:
            print("uploading dataset")
            # The data will be uploaded
            remote_dataset = dataset.upload(new_dataset_name=MY_SET_NAME)

            url = remote_dataset.url
            print(f"Successfully uploaded {url}")


if __name__ == "__main__":
    main()
