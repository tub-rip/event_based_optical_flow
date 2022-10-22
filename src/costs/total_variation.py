import logging
from typing import Union

import cv2
import numpy as np
import torch

from ..utils import SobelTorch
from . import CostBase

logger = logging.getLogger(__name__)


class TotalVariation(CostBase):
    """Total Variation for regularizer
    Args:
        direction (str) ... 'minimize' or 'maximize' or `natural`.
            Defines the objective function. If natural, it returns more interpretable value.
    """

    name = "total_variation"
    required_keys = ["flow", "omit_boundary"]

    def __init__(
        self,
        direction="minimize",
        store_history: bool = False,
        cuda_available=False,
        precision="32",
        *args,
        **kwargs,
    ):
        super().__init__(direction=direction, store_history=store_history)
        self.torch_sobel = SobelTorch(
            ksize=3, in_channels=2, cuda_available=cuda_available, precision=precision
        )

    @CostBase.register_history  # type: ignore
    @CostBase.catch_key_error  # type: ignore
    def calculate(self, arg: dict) -> Union[float, torch.Tensor]:
        """Calculate total Variation of flow
        Inputs:
            flow (np.ndarray or torch.Tensor) ... [(b,) 2, W, H]. Flow of the image.
            omit_bondary (bool) ... Omit boundary if True.

        Returns:
            (Union[float, torch.Tensor]) ... Total variation of the flow.
        """
        flow = arg["flow"]
        omit_boundary = arg["omit_boundary"]

        if isinstance(flow, torch.Tensor):
            return self.calculate_torch(flow, omit_boundary)
        elif isinstance(flow, np.ndarray):
            return self.calculate_numpy(flow, omit_boundary)
        e = f"Unsupported input type. {type(flow)}."
        logger.error(e)
        raise NotImplementedError(e)

    def calculate_torch(self, flow: torch.Tensor, omit_boundary: bool) -> torch.Tensor:
        """Calculate cost
        Inputs:
            flow (torch.Tensor) ... [(b,) 2, W, H]. Optical flow

        Returns:
            loss (torch.Tensor) ... Total Variation.
        """
        sobel = self.get_sobel_image_torch(flow, omit_boundary)

        loss = torch.mean(torch.abs(sobel))

        if self.direction == "minimize":
            return loss
        logger.warning("The loss is specified as maximize direction")
        return -loss

    def calculate_numpy(self, flow: np.ndarray, omit_boundary: bool) -> float:
        """Calculate cost
        Inputs:
            flow (np.ndarray) ... [(b,) 2, W, W]. Optical flow

        Returns:
            loss (float) ... Total variation.
        """
        sobelxx, sobelxy, sobelyx, sobelyy = self.get_sobel_image_numpy(flow, omit_boundary)

        loss = np.mean(
            np.abs(sobelxx) + np.abs(sobelxy) + np.abs(sobelyx) + np.abs(sobelyy)
        )  # L1 version

        if self.direction == "minimize":
            return loss
        return -loss

    def visualize_sobel_image(self, sobel_image):
        sobel_image = np.abs(sobel_image)
        if len(sobel_image.shape) == 4:
            sobel_image = sobel_image[0]
        if len(sobel_image.shape) == 3:
            sobel_image = np.concatenate(
                [sobel_image[0], sobel_image[1], sobel_image[2], sobel_image[3]]
            )
        sobel_image = (
            (sobel_image - sobel_image.min()) / (sobel_image.max() - sobel_image.min()) * 255
        )
        self.inter_visualizer.visualize_image(
            sobel_image.astype(np.uint8), file_prefix="total_variation"
        )

    def get_sobel_image_torch(self, flow: torch.Tensor, omit_boundary: bool) -> torch.Tensor:
        """Calculate sobel of the flow.
        Inputs:
            flow (torch.Tensor) ... [(b,) 2, W, H]. Optical flow

        Returns:
            loss (torch.Tensor) ... [(b,), 4, W, H]. 4ch is
            [x-component dx, x-component dy, y-component dx, y-component dy]
        """
        if len(flow.shape) == 3:
            flow = flow[None]  # 1, 2, W, H
        sobel = self.torch_sobel(flow) / 8.0

        if omit_boundary:
            if sobel.shape[2] > 2 and sobel.shape[3] > 2:
                sobel = sobel[..., 1:-1, 1:-1]
        return sobel

    def get_sobel_image_numpy(self, flow: np.ndarray, omit_boundary: bool) -> tuple:
        """Calculate sobel images of the flow.
        Inputs:
            flow (np.ndarray) ... [(b,) 2, W, W]. Optical flow

        Returns:
            (tuple) ... [x-component dx, x-component dy, y-component dx, y-component dy]
        """
        if len(flow.shape) == 4:
            raise NotImplementedError
        sobelxx = cv2.Sobel(flow[0], cv2.CV_64F, 1, 0, ksize=3)
        sobelxy = cv2.Sobel(flow[0], cv2.CV_64F, 0, 1, ksize=3)
        sobelyx = cv2.Sobel(flow[1], cv2.CV_64F, 1, 0, ksize=3)
        sobelyy = cv2.Sobel(flow[1], cv2.CV_64F, 0, 1, ksize=3)

        if omit_boundary:
            # Only for 3-dim array
            if sobelxx.shape[0] > 1 and sobelxx.shape[1] > 1:
                sobelxx = sobelxx[1:-1, 1:-1]
                sobelxy = sobelxy[1:-1, 1:-1]
                sobelyx = sobelyx[1:-1, 1:-1]
                sobelyy = sobelyy[1:-1, 1:-1]

        return sobelxx, sobelxy, sobelyx, sobelyy
