from pathlib import Path

from webknossos import COLOR_CATEGORY, Dataset, webknossos_context
from webknossos.dataset.sampling_modes import SamplingModes
from webknossos.geometry.mag import Mag

INPUT_DIR = Path(r"E:\PROJECTS\EM\HHD\Atlas Projects\exports\S5\S5-full-stack_aligned.czi")
OUTPUT_DIR = Path(r"E:\PROJECTS\EM\HHD\Atlas Projects\exports\S5-out")
ORG_ID = "8f72d77498ee7bdf"
TOKEN = "rFtwzykCV-7d-XWkS5kfxQ"
MY_SET_NAME = "HHD-S5"

def main(do_upload=False) -> None:
    with webknossos_context(token=TOKEN):
            
        """Convert czi file to a WEBKNOSSOS dataset."""
        dataset = Dataset.from_images(
            input_path=INPUT_DIR,
            output_path=OUTPUT_DIR,
            voxel_size=(15, 15, 300),
            name=MY_SET_NAME,
            layer_category=COLOR_CATEGORY,
            compress=True,
        )

        dataset.downsample(sampling_mode=SamplingModes.ANISOTROPIC, coarsest_mag=Mag(32))

        if do_upload:
            print("uploading dataset")
            # The data will be uploaded
            remote_dataset = dataset.upload()

            url = remote_dataset.url
            print(f"Successfully uploaded {url}")


if __name__ == "__main__":
    main(True)
