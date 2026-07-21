import matplotlib.pyplot as plt
import numpy as np


def show_two_images(img1, img2, title1="Image 1", title2="Image 2"):
    """
    Display two 2D images side by side using independent intensity ranges.

    Each image is displayed in grayscale with its color limits set to its own
    minimum and maximum values.

    Parameters
    ----------
    img1 : np.ndarray
        First two-dimensional image.
    img2 : np.ndarray
        Second two-dimensional image.
    title1 : str, optional
        Title for the first image (default is ``"Image 1"``).
    title2 : str, optional
        Title for the second image (default is ``"Image 2"``).
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    vmin1, vmax1 = np.min(img1), np.max(img1)
    vmin2, vmax2 = np.min(img2), np.max(img2)

    axes[0].imshow(img1, cmap="gray", vmin=vmin1, vmax=vmax1)
    axes[0].set_title(title1)
    axes[0].axis("off")

    axes[1].imshow(img2, cmap="gray", vmin=vmin2, vmax=vmax2)
    axes[1].set_title(title2)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
