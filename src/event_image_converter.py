import logging
from typing import Optional, Tuple, Union

import numpy as np
import torch
from scipy.ndimage.filters import gaussian_filter
from torchvision.transforms.functional import gaussian_blur

from .types import FLOAT_TORCH, NUMPY_TORCH, is_numpy, is_torch

logger = logging.getLogger(__name__)


class EventImageConverter(object):
    """Converter class of image into many different representations as an image.

    Args:
        image_size (tuple)... (H, W)
        outer_padding (int, or tuple) ... Padding to outer to the conversion. This tries to
            avoid events go out of the image.
    """

    def __init__(self, image_size: tuple, outer_padding: Union[int, Tuple[int, int]] = 0):
        if isinstance(outer_padding, (int, float)):
            self.outer_padding = (int(outer_padding), int(outer_padding))
        else:
            self.outer_padding = outer_padding
        self.image_size = tuple(int(i + p * 2) for i, p in zip(image_size, self.outer_padding))

    def update_property(
        self,
        image_size: Optional[tuple] = None,
        outer_padding: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        if image_size is not None:
            self.image_size = image_size
        if outer_padding is not None:
            if isinstance(outer_padding, int):
                self.outer_padding = (outer_padding, outer_padding)
            else:
                self.outer_padding = outer_padding
        self.image_size = tuple(i + p for i, p in zip(self.image_size, self.outer_padding))

    # Higher layer functions
    def create_iwe(
        self,
        events: NUMPY_TORCH,
        method: str = "bilinear_vote",
        sigma: int = 1,
    ) -> NUMPY_TORCH:
        """Create Image of Warped Events (IWE).

        Args:
            events (NUMPY_TORCH): [(b,) n_events, 4]
            method (str): [description]
            sigma (float): [description]

        Returns:
            NUMPY_TORCH: [(b,) H, W]
        """
        if is_numpy(events):
            return self.create_image_from_events_numpy(events, method, sigma=sigma)
        elif is_torch(events):
            return self.create_image_from_events_tensor(events, method, sigma=sigma)
        e = f"Non-supported type of events. {type(events)}"
        logger.error(e)
        raise RuntimeError(e)

    def create_eventmask(self, events: NUMPY_TORCH) -> NUMPY_TORCH:
        """Create mask image where at least one event exists.

        Args:
            events (NUMPY_TORCH): [(b,) n_events, 4]

        Returns:
            NUMPY_TORCH: [(b,) 1, H, W]
        """
        if is_numpy(events):
            return (0 != self.create_image_from_events_numpy(events, sigma=0))[..., None, :, :]
        elif is_torch(events):
            return (0 != self.create_image_from_events_tensor(events, sigma=0))[..., None, :, :]
        raise RuntimeError

    # Lower layer functions
    # Image creation functions
    def create_image_from_events_numpy(
        self,
        events: np.ndarray,
        method: str = "bilinear_vote",
        weight: Union[float, np.ndarray] = 1.0,
        sigma: int = 1,
    ) -> np.ndarray:
        """Create image of events for numpy array.

        Inputs:
            events (np.ndarray) ... [(b,) n_events, 4] Batch of events. 4 is (x, y, t, p). Attention that (x, y) could float.
                Also, x is height dimension and y is the width dimension.
            method (str) ... method to accumulate events. "count", "bilinear_vote", "polarity", etc.
            weight (float or np.ndarray) ... Only applicable when method = "bilinear_vote".
            sigma (int) ... Sigma for the gaussian blur.

        Returns:
            image ... [(b,) H, W]. Each index indicates the sum of the event, based on the specified method.
        """
        if method == "count":
            image = self.count_event_numpy(events)
        elif method == "bilinear_vote":
            image = self.bilinear_vote_numpy(events, weight=weight)
        elif method == "polarity":  # TODO implement
            pos_flag = events[..., 3] > 0
            if is_numpy(weight):
                pos_image = self.bilinear_vote_numpy(events[pos_flag], weight=weight[pos_flag])
                neg_image = self.bilinear_vote_numpy(events[~pos_flag], weight=weight[~pos_flag])
            else:
                pos_image = self.bilinear_vote_numpy(events[pos_flag], weight=weight)
                neg_image = self.bilinear_vote_numpy(events[~pos_flag], weight=weight)
            image = np.stack([pos_image, neg_image], axis=-3)
        else:
            e = f"{method = } is not supported."
            logger.error(e)
            raise NotImplementedError(e)
        if sigma > 0:
            image = gaussian_filter(image, sigma)
        return image

    def create_image_from_events_tensor(
        self,
        events: torch.Tensor,
        method: str = "bilinear_vote",
        weight: FLOAT_TORCH = 1.0,
        sigma: int = 0,
    ) -> torch.Tensor:
        """Create image of events for tensor array.

        Inputs:
            events (torch.Tensor) ... [(b, ) n_events, 4] Batch of events. 4 is (x, y, t, p). Attention that (x, y) could float.
                Also, x is the width dimension and y is the height dimension.
            method (str) ... method to accumulate events. "count", "bilinear_vote", "polarity", etc.
            weight (float or torch.Tensor) ... Only applicable when method = "bilinear_vote".
            sigma (int) ... Sigma for the gaussian blur.

        Returns:
            image ... [(b, ) W, H]. Each index indicates the sum of the event, based on the specified method.
        """
        if method == "count":
            image = self.count_event_tensor(events)
        elif method == "bilinear_vote":
            image = self.bilinear_vote_tensor(events, weight=weight)
        else:
            e = f"{method = } is not implemented"
            logger.error(e)
            raise NotImplementedError(e)
        if sigma > 0:
            if len(image.shape) == 2:
                image = image[None, None, ...]
            elif len(image.shape) == 3:
                image = image[:, None, ...]
            image = gaussian_blur(image, kernel_size=3, sigma=sigma)
        return torch.squeeze(image)

    def count_event_numpy(self, events: np.ndarray):
        """Count event and make image.

        Args:
            events ... [(b,) n_events, 4] Batch of events. 4 is (x, y, t, p). Attention that (x, y) could float.

        Returns:
            image ... [(b,) W, H]. Each index indicates the sum of the event, just counting.
        """
        if len(events.shape) == 2:
            events = events[None, ...]  # 1 x n x 4

        # x-y is height-width
        ph, pw = self.outer_padding
        h, w = self.image_size
        nb = len(events)
        image = np.zeros((nb, h * w), dtype=np.float64)

        floor_xy = np.floor(events[..., :2] + 1e-8)
        floor_to_xy = events[..., :2] - floor_xy

        x1 = floor_xy[..., 1] + pw
        y1 = floor_xy[..., 0] + ph
        inds = np.concatenate(
            [
                x1 + y1 * w,
                x1 + (y1 + 1) * w,
                (x1 + 1) + y1 * w,
                (x1 + 1) + (y1 + 1) * w,
            ],
            axis=-1,
        )
        inds_mask = np.concatenate(
            [
                (0 <= x1) * (x1 < w) * (0 <= y1) * (y1 < h),
                (0 <= x1) * (x1 < w) * (0 <= y1 + 1) * (y1 + 1 < h),
                (0 <= x1 + 1) * (x1 + 1 < w) * (0 <= y1) * (y1 < h),
                (0 <= x1 + 1) * (x1 + 1 < w) * (0 <= y1 + 1) * (y1 + 1 < h),
            ],
            axis=-1,
        )
        vals = np.ones_like(inds)
        inds = (inds * inds_mask).astype(np.int64)
        vals = vals * inds_mask
        for i in range(nb):
            np.add.at(image[i], inds[i], vals[i])
        return image.reshape((nb,) + self.image_size).squeeze()

    def count_event_tensor(self, events: torch.Tensor):
        """Tensor version of `count_event_numpy().`

        Args:
            events (torch.Tensor) ... [(b,) n_events, 4] Batch of events. 4 is (x, y, t, p). Attention that (x, y) could float.

        Returns:
            image ... [(b,) H, W]. Each index indicates the bilinear vote result. If the outer_padding is set,
                the return size will be [H + outer_padding, W + outer_padding].
        """
        if len(events.shape) == 2:
            events = events[None, ...]  # 1 x n x 4

        ph, pw = self.outer_padding
        h, w = self.image_size
        nb = len(events)
        image = events.new_zeros((nb, h * w))

        floor_xy = torch.floor(events[..., :2] + 1e-6)
        floor_to_xy = events[..., :2] - floor_xy
        floor_xy = floor_xy.long()

        x1 = floor_xy[..., 1] + pw
        y1 = floor_xy[..., 0] + ph
        inds = torch.cat(
            [
                x1 + y1 * w,
                x1 + (y1 + 1) * w,
                (x1 + 1) + y1 * w,
                (x1 + 1) + (y1 + 1) * w,
            ],
            dim=-1,
        )  # [(b, ) n_events x 4]
        inds_mask = torch.cat(
            [
                (0 <= x1) * (x1 < w) * (0 <= y1) * (y1 < h),
                (0 <= x1) * (x1 < w) * (0 <= y1 + 1) * (y1 + 1 < h),
                (0 <= x1 + 1) * (x1 + 1 < w) * (0 <= y1) * (y1 < h),
                (0 <= x1 + 1) * (x1 + 1 < w) * (0 <= y1 + 1) * (y1 + 1 < h),
            ],
            axis=-1,
        )
        vals = torch.ones_like(inds)
        inds = (inds * inds_mask).long()
        vals = vals * inds_mask
        image.scatter_add_(1, inds, vals)
        return image.reshape((nb,) + self.image_size).squeeze()

    def bilinear_vote_numpy(self, events: np.ndarray, weight: Union[float, np.ndarray] = 1.0):
        """Use bilinear voting to and make image.

        Args:
            events (np.ndarray) ... [(b, ) n_events, 4] Batch of events. 4 is (x, y, t, p). Attention that (x, y) could float.
            weight (float or np.ndarray) ... Weight to multiply to the voting value.
                If scalar, the weight is all the same among events.
                If it's array-like, it should be the shape of [n_events].
                Defaults to 1.0.

        Returns:
            image ... [(b, ) H, W]. Each index indicates the bilinear vote result. If the outer_padding is set,
                the return size will be [H + outer_padding, W + outer_padding].
        """
        if type(weight) == np.ndarray:
            assert weight.shape == events.shape[:-1]
        if len(events.shape) == 2:
            events = events[None, ...]  # 1 x n x 4

        # x-y is height-width
        ph, pw = self.outer_padding
        h, w = self.image_size
        nb = len(events)
        image = np.zeros((nb, h * w), dtype=np.float64)

        floor_xy = np.floor(events[..., :2] + 1e-8)
        floor_to_xy = events[..., :2] - floor_xy

        x1 = floor_xy[..., 1] + pw
        y1 = floor_xy[..., 0] + ph
        inds = np.concatenate(
            [
                x1 + y1 * w,
                x1 + (y1 + 1) * w,
                (x1 + 1) + y1 * w,
                (x1 + 1) + (y1 + 1) * w,
            ],
            axis=-1,
        )
        inds_mask = np.concatenate(
            [
                (0 <= x1) * (x1 < w) * (0 <= y1) * (y1 < h),
                (0 <= x1) * (x1 < w) * (0 <= y1 + 1) * (y1 + 1 < h),
                (0 <= x1 + 1) * (x1 + 1 < w) * (0 <= y1) * (y1 < h),
                (0 <= x1 + 1) * (x1 + 1 < w) * (0 <= y1 + 1) * (y1 + 1 < h),
            ],
            axis=-1,
        )
        w_pos0 = (1 - floor_to_xy[..., 0]) * (1 - floor_to_xy[..., 1]) * weight
        w_pos1 = floor_to_xy[..., 0] * (1 - floor_to_xy[..., 1]) * weight
        w_pos2 = (1 - floor_to_xy[..., 0]) * floor_to_xy[..., 1] * weight
        w_pos3 = floor_to_xy[..., 0] * floor_to_xy[..., 1] * weight
        vals = np.concatenate([w_pos0, w_pos1, w_pos2, w_pos3], axis=-1)
        inds = (inds * inds_mask).astype(np.int64)
        vals = vals * inds_mask
        for i in range(nb):
            np.add.at(image[i], inds[i], vals[i])
        return image.reshape((nb,) + self.image_size).squeeze()

    def bilinear_vote_tensor(self, events: torch.Tensor, weight: FLOAT_TORCH = 1.0):
        """Tensor version of `bilinear_vote_numpy().`

        Args:
            events (torch.Tensor) ... [(b,) n_events, 4] Batch of events. 4 is (x, y, t, p). Attention that (x, y) could float.
            weight (float or torch.Tensor) ... Weight to multiply to the voting value.
                If scalar, the weight is all the same among events.
                If it's array-like, it should be the shape of [(b,) n_events].
                Defaults to 1.0.

        Returns:
            image ... [(b,) H, W]. Each index indicates the bilinear vote result. If the outer_padding is set,
                the return size will be [H + outer_padding, W + outer_padding].
        """
        if type(weight) == torch.Tensor:
            assert weight.shape == events.shape[:-1]
        if len(events.shape) == 2:
            events = events[None, ...]  # 1 x n x 4

        ph, pw = self.outer_padding
        h, w = self.image_size
        nb = len(events)
        image = events.new_zeros((nb, h * w))

        floor_xy = torch.floor(events[..., :2] + 1e-6)
        floor_to_xy = events[..., :2] - floor_xy
        floor_xy = floor_xy.long()

        x1 = floor_xy[..., 1] + pw
        y1 = floor_xy[..., 0] + ph
        inds = torch.cat(
            [
                x1 + y1 * w,
                x1 + (y1 + 1) * w,
                (x1 + 1) + y1 * w,
                (x1 + 1) + (y1 + 1) * w,
            ],
            dim=-1,
        )  # [(b, ) n_events x 4]
        inds_mask = torch.cat(
            [
                (0 <= x1) * (x1 < w) * (0 <= y1) * (y1 < h),
                (0 <= x1) * (x1 < w) * (0 <= y1 + 1) * (y1 + 1 < h),
                (0 <= x1 + 1) * (x1 + 1 < w) * (0 <= y1) * (y1 < h),
                (0 <= x1 + 1) * (x1 + 1 < w) * (0 <= y1 + 1) * (y1 + 1 < h),
            ],
            axis=-1,
        )

        w_pos0 = (1 - floor_to_xy[..., 0]) * (1 - floor_to_xy[..., 1]) * weight
        w_pos1 = floor_to_xy[..., 0] * (1 - floor_to_xy[..., 1]) * weight
        w_pos2 = (1 - floor_to_xy[..., 0]) * floor_to_xy[..., 1] * weight
        w_pos3 = floor_to_xy[..., 0] * floor_to_xy[..., 1] * weight
        vals = torch.cat([w_pos0, w_pos1, w_pos2, w_pos3], dim=-1)  # [(b,) n_events x 4]

        inds = (inds * inds_mask).long()
        vals = vals * inds_mask
        image.scatter_add_(1, inds, vals)
        return image.reshape((nb,) + self.image_size).squeeze()
