from pathlib import Path

from webknossos import COLOR_CATEGORY, Dataset, webknossos_context
from webknossos.dataset.sampling_modes import SamplingModes
from webknossos.geometry.mag import Mag

INPUT_DIR = Path(r"E:\PROJECTS\EM\Filipa\M2-2\ATLAS-projects\proj-4-20260302\NEW LOCATION20260303_data\session_1149164643\Site 1\M2-2_ROI-2.czi")
OUTPUT_DIR = Path(r"E:\PROJECTS\EM\Rafa\test\out")
ORG_ID = "8f72d77498ee7bdf"
TOKEN = "rFtwzykCV-7d-CHANGEIT"
MY_SET_NAME = "my_test_set_5"

def main(do_upload=False) -> None:
    with webknossos_context(token=TOKEN):
            
        """Convert czi file to a WEBKNOSSOS dataset."""
        dataset = Dataset.from_images(
            input_path=INPUT_DIR,
            output_path=OUTPUT_DIR,
            voxel_size=(20, 20, 200),
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
