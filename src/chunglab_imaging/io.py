from pathlib import Path

import numpy as np


def imread(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".nd2":
        import nd2

        return nd2.imread(path)
    raise ValueError(f"Unrecognized file extension: {path.name}")
