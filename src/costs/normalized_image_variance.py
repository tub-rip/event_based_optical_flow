import logging
from typing import Union

import numpy as np
import torch

from . import CostBase

logger = logging.getLogger(__name__)


class NormalizedImageVariance(CostBase):
    """Normalized image variance,
    a.k.a FWP (flow Warp Loss) by Stoffregen et al. ECCV 2020.
    Args:
        direction (str) ... 'minimize' or 'maximize' or `natural`.
            Defines the objective function. If natural, it returns more interpretable value.
    """

    name = "normalized_image_variance"
    required_keys = ["orig_iwe", "iwe", "omit_boundary"]

    def __init__(self, direction="minimize", store_history: bool = False, *args, **kwargs):
        super().__init__(direction=direction, store_history=store_history)

    @CostBase.register_history  # type: ignore
    @CostBase.catch_key_error  # type: ignore
    def calculate(self, arg: dict) -> Union[float, torch.Tensor]:
        """Calculate the normalized contrast of the IWE.
        Inputs:
            iwe (np.ndarray or torch.Tensor) ... [W, H]. Image of warped events
            orig_iwe (np.ndarray or torch.Tensor) ... [W, H]. Image of original events
            omit_bondary (bool) ... Omit boundary if True.

        Returns:
            contrast (Union[float, torch.Tensor]) ... normalized contrast of the image.
        """
        iwe = arg["iwe"]
        orig_iwe = arg["orig_iwe"]
        if arg["omit_boundary"]:
            iwe = iwe[..., 1:-1, 1:-1]  # omit boundary
        if isinstance(iwe, torch.Tensor):
            return self.calculate_torch(iwe, orig_iwe)
        elif isinstance(iwe, np.ndarray):
            return self.calculate_numpy(iwe, orig_iwe)
        e = f"Unsupported input type. {type(iwe)}."
        logger.error(e)
        raise NotImplementedError(e)

    def calculate_torch(self, iwe: torch.Tensor, orig_iwe: torch.Tensor) -> torch.Tensor:
        """Calculate the normalized contrast of the IWE.
        Inputs:
            iwe (torch.Tensor) ... [W, H]. Image of warped events
            orig_iwe (torch.Tensor) ... [W, H]. Image of original events

        Returns:
            loss (torch.Tensor) ... contrast of the image.
        """
        loss1 = torch.var(iwe)
        loss2 = torch.var(orig_iwe)
        if self.direction == "minimize":
            return loss2 / loss1
        logger.warning("The loss is specified as maximize direction")
        return loss1 / loss2

    def calculate_numpy(self, iwe: np.ndarray, orig_iwe: np.ndarray) -> float:
        """Calculate the normalized contrast of the IWE.
        Inputs:
            iwe (np.ndarray) ... [W, H]. Image of warped events
            orig_iwe (np.ndarray) ... [W, H]. Image of original events

        Returns:
            contrast (float) ... contrast of the image.
        """
        loss1 = np.var(iwe)
        loss2 = np.var(orig_iwe)
        if self.direction == "minimize":
            return loss2 / loss1
        return loss1 / loss2
