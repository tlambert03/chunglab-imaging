import numpy as np
from skimage import draw
from skimage.feature import blob_log


def blobs2mask(
    blobs: np.ndarray, shape: tuple[int, ...], grow: float = 2
) -> np.ndarray:
    """Create mask for `blobs` (coordinates, with radii) with `shape`."""

    mask = np.zeros(shape, bool)
    for *c, r in blobs:
        coords = draw.disk(c[-2:], r + grow, shape=shape[-2:])
        if len(c) == 3:
            mask[c[0]][coords] = True
        else:
            mask[coords] = True

    return mask


def find_lds(
    data: np.ndarray,
    min_sigma: float = 2,
    max_sigma: float = 8,
    threshold: float = 1000,
    overlap: float = 0.2,
) -> np.ndarray:
    """Find lipid droplets in 2D or 3D data."""
    if data.ndim == 2:
        result = blob_log(
            data.astype(float),
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            threshold=threshold,
            overlap=overlap,
            exclude_border=True,
        )
        return result.astype(int)  # type: ignore
    if data.ndim == 3:
        blobs = []
        for z, plane in enumerate(data):
            _b = find_lds(plane)
            blobs.append(np.hstack([np.ones((len(_b), 1), int) * z, _b]))
        return np.vstack(blobs)
    raise ValueError("data must be 2d or 3d")
