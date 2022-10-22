import logging
from typing import Union

import numpy as np
import torch

from . import CostBase

logger = logging.getLogger(__name__)


class ImageVariance(CostBase):
    """Image Variance from Gallego et al. CVPR 2018.
    Args:
        direction (str) ... 'minimize' or 'maximize' or `natural`.
            Defines the objective function. If natural, it returns more interpretable value.
    """

    name = "image_variance"
    required_keys = ["iwe", "omit_boundary"]

    def __init__(self, direction="minimize", store_history: bool = False, *args, **kwargs):
        super().__init__(direction=direction, store_history=store_history)

    @CostBase.register_history  # type: ignore
    @CostBase.catch_key_error  # type: ignore
    def calculate(self, arg: dict) -> Union[float, torch.Tensor]:
        """Calculate contrast of the IWE.
        Inputs:
            iwe (np.ndarray or torch.Tensor) ... [W, H]. Image of warped events
            omit_bondary (bool) ... Omit boundary if True.

        Returns:
            contrast (Union[float, torch.Tensor]) ... contrast of the image.
        """
        iwe = arg["iwe"]
        if arg["omit_boundary"]:
            iwe = iwe[..., 1:-1, 1:-1]  # omit boundary
        if isinstance(iwe, torch.Tensor):
            return self.calculate_torch(iwe)
        elif isinstance(iwe, np.ndarray):
            return self.calculate_numpy(iwe)
        e = f"Unsupported input type. {type(iwe)}."
        logger.error(e)
        raise NotImplementedError(e)

    def calculate_torch(self, iwe: torch.Tensor) -> torch.Tensor:
        """Calculate contrast of the IWE.
        Inputs:
            iwe (torch.Tensor) ... [W, H]. Image of warped events

        Returns:
            loss (torch.Tensor) ... contrast of the image.
        """
        loss = torch.var(iwe)
        if self.direction == "minimize":
            return -loss
        return loss

    def calculate_numpy(self, iwe: np.ndarray) -> float:
        """Calculate contrast of the IWE.
        Inputs:
            iwe (np.ndarray) ... [W, H]. Image of warped events

        Returns:
            contrast (float) ... contrast of the image.
        """
        loss = np.var(iwe)
        if self.direction == "minimize":
            return -loss
        return loss
