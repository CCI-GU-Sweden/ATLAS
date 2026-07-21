import matplotlib.pyplot as plt
import numpy as np
from skimage.registration import phase_cross_correlation
from skimage.transform import rotate, warp_polar


def crop_to_center_overlap(img1, img2):
    """
    Crops two images to the largest common size, centered on their respective centers.

    Parameters:
    -----------
    img1, img2 : np.ndarray
        Input images of possibly different shapes.

    Returns:
    --------
    cropped1, cropped2 : np.ndarray
        Cropped images of the same shape, center-aligned.
    """
    shape1 = np.array(img1.shape)
    shape2 = np.array(img2.shape)

    # Determine minimum shape in each dimension
    min_shape = np.minimum(shape1, shape2)

    # Compute center points
    center1 = shape1 // 2
    center2 = shape2 // 2

    # Compute start and end indices for cropping
    start1 = center1 - min_shape // 2
    end1   = start1 + min_shape

    start2 = center2 - min_shape // 2
    end2   = start2 + min_shape

    cropped1 = img1[start1[0]:end1[0], start1[1]:end1[1]]
    cropped2 = img2[start2[0]:end2[0], start2[1]:end2[1]]

    return cropped1, cropped2

def estimate_rotation_translation_fourier(reference_img, moving_img, upsample_factor=10):
    from scipy.fft import fft2, fftshift
    from scipy.ndimage import shift as nd_shift
    assert reference_img.shape == moving_img.shape, "Images must be the same shape"

    # Store original dtype
    orig_dtype = moving_img.dtype

    # Step 1: FFT magnitude
    f_ref = fftshift(fft2(reference_img))
    f_mov = fftshift(fft2(moving_img))
    mag_ref = np.abs(f_ref)
    mag_mov = np.abs(f_mov)

    # Step 2: Polar transform (linear for rotation only)
    center = np.array(reference_img.shape) / 2
    polar_ref = warp_polar(mag_ref, center=center, scaling='linear', preserve_range=True)
    polar_mov = warp_polar(mag_mov, center=center, scaling='linear', preserve_range=True)

    # Step 3: Estimate rotation angle
    shift, _, _ = phase_cross_correlation(polar_ref, polar_mov, upsample_factor=upsample_factor)
    rotation_deg = -shift[0] * 360 / polar_ref.shape[0]

    # Step 4: Apply rotation (preserve original data range)
    rotated = rotate(moving_img, rotation_deg, resize=False, order=2, mode='constant', cval=0.0, preserve_range=True)

    # Step 5: Estimate translation
    translation_shift, _, _ = phase_cross_correlation(reference_img, rotated, upsample_factor=upsample_factor)

    # Step 6: Apply translation (preserve range by keeping dtype conversion to the end)
    aligned = nd_shift(rotated, shift=translation_shift, order=2, mode='constant', cval=0.0)

    # Step 7: Cast back to original dtype
    aligned_final = aligned.astype(orig_dtype)

    return rotation_deg, translation_shift, aligned_final

def estimate_rotation_translation_fourier_no_rotation(reference_img, moving_img, upsample_factor=10):
    from scipy.fft import fft2, fftshift
    assert reference_img.shape == moving_img.shape, "Images must be the same shape"
    #orig_dtype = moving_img.dtype

    # Step 1: FFT magnitude
    f_ref = fftshift(fft2(reference_img))
    f_mov = fftshift(fft2(moving_img))
    mag_ref = np.abs(f_ref)
    mag_mov = np.abs(f_mov)

    # Step 2: Polar transform (linear for rotation only)
    center = np.array(reference_img.shape) / 2
    polar_ref = warp_polar(mag_ref, center=center, scaling='linear', preserve_range=True)
    polar_mov = warp_polar(mag_mov, center=center, scaling='linear', preserve_range=True)

    # Step 3: Estimate rotation angle
    shift, _, _ = phase_cross_correlation(polar_ref, polar_mov, upsample_factor=upsample_factor)
    rotation_deg = -shift[0] * 360 / polar_ref.shape[0]

    # Step 4: Rotate reference image instead of moving image (keep moving image unchanged)
    rotated_ref = rotate(reference_img, -rotation_deg, resize=False, order=2, mode='constant', cval=0.0, preserve_range=True)

    # Step 5: Estimate translation between rotated ref and original moving image
    translation_shift, _, _ = phase_cross_correlation(rotated_ref, moving_img, upsample_factor=upsample_factor)

    # Return rotation and translation (both in original image coordinates)
    return rotation_deg, translation_shift


def split_into_quadrants_with_centers(ref_img: np.ndarray, mov_img: np.ndarray):
    """
    Splits two identically shaped 2D images into 4 quadrants (2x2) without dropping any pixels.
    Returns matching quadrant pairs and their center coordinates in NumPy (dim0, dim1) = (y, x) order.

    Parameters:
    -----------
    ref_img : np.ndarray
        The reference image (2D).
    mov_img : np.ndarray
        The moving image (2D), must have same shape and dtype as reference.

    Returns:
    --------
    List of tuples: (ref_quad, mov_quad, center_yx)
        ref_quad : np.ndarray
        mov_quad : np.ndarray
        center_yx : np.ndarray (y, x) in full image coordinates
    """
    if ref_img.shape != mov_img.shape:
        raise ValueError("Input images must have the same shape.")
    if ref_img.dtype != mov_img.dtype:
        raise ValueError("Input images must have the same dtype.")

    h, w = ref_img.shape
    h_mid = (h + 1) // 2  # ceil for odd sizes
    w_mid = (w + 1) // 2

    quadrants = []

    slices = [
        (slice(0, h_mid), slice(0, w_mid)),       # Top-left
        (slice(0, h_mid), slice(w_mid, w)),       # Top-right
        (slice(h_mid, h), slice(0, w_mid)),       # Bottom-left
        (slice(h_mid, h), slice(w_mid, w))        # Bottom-right
    ]

    for sl_y, sl_x in slices:
        ref_q = ref_img[sl_y, sl_x]
        mov_q = mov_img[sl_y, sl_x]

        center_y = (sl_y.start + sl_y.stop - 1) / 2
        center_x = (sl_x.start + sl_x.stop - 1) / 2
        center_yx = np.array([center_y, center_x])  # NumPy dim0, dim1 = y, x

        quadrants.append((ref_q, mov_q, center_yx))

    return quadrants


def plot_alignment_with_grid(ref_img, aligned_img, title1="Reference", title2="Aligned Moving Image"):
    """
    Plot the reference and aligned images side by side with grid lines for comparison.

    Parameters:
    -----------
    ref_img : np.ndarray
        Reference image (2D).
    aligned_img : np.ndarray
        Aligned moving image (same shape as ref_img).
    title1 : str
        Title for the reference image subplot.
    title2 : str
        Title for the aligned image subplot.
    """
    if ref_img.shape != aligned_img.shape:
        raise ValueError("Both images must have the same shape for comparison.")

    vmin = min(ref_img.min(), aligned_img.min())
    vmax = max(ref_img.max(), aligned_img.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, img, title in zip(axes, [ref_img, aligned_img], [title1, title2]):
        ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(np.linspace(0, img.shape[1], 5))
        ax.set_yticks(np.linspace(0, img.shape[0], 5))
        ax.grid(color='cyan', linestyle='--', linewidth=0.5)
        ax.tick_params(color='cyan', labelcolor='black')
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")

    plt.tight_layout()
    plt.show()

def compute_common_canvas(ref_shape, mov_shape, tform):
    """
    Compute a common bounding box (canvas) that contains both:
    - the reference image (at identity position), and
    - the moving image after transformation by `tform`.

    Parameters:
    -----------
    ref_shape : tuple (h, w)
        Shape of the reference image.
    mov_shape : tuple (h, w)
        Shape of the moving image.
    tform : skimage.transform.Transform
        Transform from moving image coordinates to reference.

    Returns:
    --------
    canvas_shape : tuple (h, w)
        Size of the canvas that includes both images.
    offset : np.ndarray
        Shift (y, x) to apply to both images to place them correctly in the canvas.
    """
    h_ref, w_ref = ref_shape
    h_mov, w_mov = mov_shape

    # Corners of reference image
    corners_ref = np.array([
        [0, 0],
        [0, w_ref - 1],
        [h_ref - 1, 0],
        [h_ref - 1, w_ref - 1]
    ])
    #corners_ref_tf = tform(corners_ref)  # transform and restore (y, x) format
    #print("Ref (rounded):")
    #print('\n'.join(['  ' + str(np.round(row).astype(int)) for row in corners_ref]))

    # Corners of moving image, transformed into reference space
    corners_mov = np.array([
        [0, 0],
        [0, w_mov - 1],
        [h_mov - 1, 0],
        [h_mov - 1, w_mov - 1]
    ])
    corners_mov_tf = tform.inverse(corners_mov[:, ::-1])[:, ::-1]  # (y, x) → (x, y) → back to (y, x)
    corners_mov_tf = corners_mov_tf
    #print("Transformed corners (rounded):")
    #print('\n'.join(['  ' + str(np.round(row).astype(int)) for row in corners_mov_tf]))

    # Combine all corners
    all_yx = np.vstack([corners_ref, corners_mov_tf])

    # Compute bounding box
    min_yx = np.floor(all_yx.min(axis=0)).astype(int)
    max_yx = np.ceil(all_yx.max(axis=0)).astype(int)

    canvas_shape = (max_yx - min_yx).astype(int)
    offset = min_yx  # offset needed to bring all coords >= 0

    print("Canvas shape (h, w):", canvas_shape)
    print("Offset to apply (y, x):", offset)

    return tuple(canvas_shape), offset
