import logging
from typing import Optional, Tuple

import numpy as np

from ..types import FLOAT_TORCH, NUMPY_TORCH, is_numpy, is_torch

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    e = "Torch is disabled."
    logger.warning(e)


# Simulator module
def generate_events(
    n_events: int,
    height: int,
    width: int,
    tmin: float = 0.0,
    tmax: float = 0.5,
    dist: str = "uniform",
) -> np.ndarray:
    """Generate random events.

    Args:
        n_events (int) ... num of events
        height (int) ... height of the camera
        width (int) ... width of the camera
        tmin (float) ... timestamp min
        tmax (float) ... timestamp max
        dist (str) ... currently only "uniform" is supported.

    Returns:
        events (np.ndarray) ... [n_events x 4] numpy array. (x, y, t, p)
            x indicates height direction.
    """
    x = np.random.randint(0, height, n_events)
    y = np.random.randint(0, width, n_events)
    t = np.random.uniform(tmin, tmax, n_events)
    t = np.sort(t)
    p = np.random.randint(0, 2, n_events)

    events = np.concatenate([x[..., None], y[..., None], t[..., None], p[..., None]], axis=1)
    return events


def crop_event(events: NUMPY_TORCH, x0: int, x1: int, y0: int, y1: int) -> NUMPY_TORCH:
    """Crop events.

    Args:
        events (NUMPY_TORCH): [n x 4]. [x, y, t, p].
        x0 (int): Start of the crop, at row[0]
        x1 (int): End of the crop, at row[0]
        y0 (int): Start of the crop, at row[1]
        y1 (int): End of the crop, at row[1]

    Returns:
        NUMPY_TORCH: Cropped events.
    """
    mask = (
        (x0 <= events[..., 0])
        * (events[..., 0] < x1)
        * (y0 <= events[..., 1])
        * (events[..., 1] < y1)
    )
    cropped = events[mask]
    return cropped


def set_event_origin_to_zero(events: np.ndarray, x0: int, y0: int, t0: float = 0.0) -> np.ndarray:
    """Set each origin of each row to 0.

    Args:
        events (np.ndarray): [n x 4]. [x, y, t, p].
        x0 (int): x origin
        y0 (int): y origin
        t0 (float): t origin

    Returns:
        np.ndarray: [n x 4]. x is in [0, xmax - x0], and so on.
    """
    basis = np.array([x0, y0, t0, 0.0])
    if is_torch(events):
        basis = torch.from_numpy(basis)
    return events - basis


def undistort_events(events, map_x, map_y, h, w):
    """Undistort (rectify) events.
    Args:
        events ... [x, y, t, p]. X is height direction.
        map_x, map_y... meshgrid

    Returns:
        events... events that is in the camera plane after undistortion.
    TODO check overflow
    """
    # k = np.int32(map_y[np.int16(events[:, 1]), np.int16(events[:, 0])])
    # l = np.int32(map_x[np.int16(events[:, 1]), np.int16(events[:, 0])])
    # k = np.int32(map_y[events[:, 1].astype(np.int32), events[:, 0].astype(np.int32)])
    # l = np.int32(map_x[events[:, 1].astype(np.int32), events[:, 0].astype(np.int32)])
    # undistort_events = np.copy(events)
    # undistort_events[:, 0] = l
    # undistort_events[:, 1] = k
    # return undistort_events[((0 <= k) & (k < h)) & ((0 <= l) & (l < w))]

    k = np.int32(map_y[events[:, 0].astype(np.int32), events[:, 1].astype(np.int32)])
    l = np.int32(map_x[events[:, 0].astype(np.int32), events[:, 1].astype(np.int32)])
    undistort_events = np.copy(events)
    undistort_events[:, 0] = k
    undistort_events[:, 1] = l
    return undistort_events[((0 <= k) & (k < h)) & ((0 <= l) & (l < w))]
