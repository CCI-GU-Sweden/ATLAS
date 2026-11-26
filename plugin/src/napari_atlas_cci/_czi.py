"""Shared streaming CZI writer for local exports and remote downloads."""

from pathlib import Path

import numpy as np


def write_czi_plane(writer, image_yx, z_index, channel=0, tile_size=1024):
    image_yx = np.asarray(image_yx)
    if image_yx.ndim != 2:
        raise ValueError(f"Expected a YX plane, got shape {image_yx.shape}.")
    # Keep each write comfortably below libCZI's 10 MB block limit.
    tile_size = min(
        tile_size, int(np.sqrt(8_000_000 / image_yx.dtype.itemsize))
    )
    height, width = image_yx.shape
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = np.ascontiguousarray(
                image_yx[y : y + tile_size, x : x + tile_size, None]
            )
            # pylibCZIrw expects Y, X, channels, including singleton channels.
            if (
                writer.write(
                    data=tile,
                    location=(x, y),
                    plane={"Z": z_index, "C": channel},
                )
                is False
            ):
                raise OSError(
                    f"CZI write failed at Z={z_index}, C={channel}, X={x}, Y={y}."
                )


def write_czi_volume(
    path: Path,
    planes,
    voxel_size_m,
    compression_options,
    channel_names,
    document_name=None,
    exist_ok=False,
):
    """Consume (Z, channel, YX plane) tuples without materializing the volume."""
    from pylibCZIrw import czi

    with czi.create_czi(
        path, exist_ok=exist_ok, compression_options=compression_options
    ) as writer:
        count = 0
        for z, channel, image_yx in planes:
            write_czi_plane(writer, image_yx, z, channel)
            del image_yx  # Release this view before requesting the next plane.
            count += 1
            print(f"CZI wrote Z={z}, C={channel}")
        if not count:
            raise ValueError("The dataset has no image planes to export.")
        writer.write_metadata(
            document_name=document_name or path.stem,
            channel_names=channel_names,
            scale_x=voxel_size_m[0],
            scale_y=voxel_size_m[1],
            scale_z=voxel_size_m[2],
        )
