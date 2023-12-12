from pathlib import Path
from typing import Annotated

import numpy as np
from magicgui import magic_factory
from skimage.feature import blob_log


def read_nd2(path: str | Path) -> np.ndarray:
    import nd2

    return nd2.imread(path)


def imread(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".nd2":
        return read_nd2(path)
    raise ValueError(f"Unrecognized file extension: {path.name}")


def blobs2mask(blobs, shape: tuple[int, ...], grow=2) -> np.ndarray:
    """Create mask for `blobs` (coordinates, with radii) with `shape`."""
    from skimage import draw

    mask = np.zeros(shape, bool)
    for *c, r in blobs:
        coords = draw.disk(c[-2:], r + grow, shape=shape[-2:])
        if len(c) == 3:
            mask[c[0]][coords] = True
        else:
            mask[coords] = True

    return mask


def find_lds(data: np.ndarray, min_sigma=2, max_sigma=8, threshold=1000, overlap=0.2):
    """Find lipid droplets in 2D or 3D data."""
    if data.ndim == 2:
        return blob_log(
            data.astype(float),
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            threshold=threshold,
            overlap=overlap,
            exclude_border=True,
        ).astype(int)
    if data.ndim == 3:
        blobs = []
        for z, plane in enumerate(data):
            _b = find_lds(plane)
            blobs.append(np.hstack([np.ones((len(_b), 1), int) * z, _b]))
        return np.vstack(blobs)
    raise ValueError("data must be 2d or 3d")


from magicgui.widgets import RangeSlider

Range = Annotated[tuple[int, int], {"widget_type": RangeSlider, 'min': 0, 'max': 10}]


@magic_factory
def ld_param_widget(sigmas: Range) -> None:
    print(sigmas)
