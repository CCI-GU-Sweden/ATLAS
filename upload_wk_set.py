import webknossos as wk
from webknossos import LayerToLink
from webknossos.dataset.remote_folder import RemoteFolder

from pathlib import Path

INPUT_DIR = Path(r"E:\PROJECTS\EM\LUKE\TA27\out_1b")
ORG_ID = "8f72d77498ee7bdf"
TOKEN = "UiH6sS4CktIU9xYT7MfYkQ"
MY_SET_NAME = "TA_27_ROI_1B_10nm"

def main() -> None:
    with wk.webknossos_context(token=TOKEN):
        # Open your unpublished dataset
        dataset = wk.Dataset.open(INPUT_DIR)

        # Publish the dataset to webknossos
        # The data will be uploaded to the upload destination folder
        remote_dataset = dataset.upload()

        url = remote_dataset.url
        print(f"Successfully uploaded {url}")


if __name__ == "__main__":
    main()