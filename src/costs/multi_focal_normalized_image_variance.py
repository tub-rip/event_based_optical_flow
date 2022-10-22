import logging
from typing import Optional

import numpy as np
import torch

from ..types import FLOAT_TORCH
from . import CostBase, NormalizedImageVariance

logger = logging.getLogger(__name__)


class MultiFocalNormalizedImageVariance(CostBase):
    """Multi-focus normalized image variance, Shiba et al. ECCV 2022.

    Args:
        direction (str) ... 'minimize' or 'maximize' or `natural`.
            Defines the objective function. If natural, it returns more interpretable value.
    """

    name = "multi_focal_normalized_image_variance"
    required_keys = ["forward_iwe", "backward_iwe", "middle_iwe", "omit_boundary", "orig_iwe"]

    def __init__(self, direction="minimize", store_history: bool = False, *args, **kwargs):
        super().__init__(direction=direction, store_history=store_history)
        self.variance_loss = NormalizedImageVariance(direction=direction)

    @CostBase.register_history  # type: ignore
    @CostBase.catch_key_error  # type: ignore
    def calculate(self, arg: dict) -> FLOAT_TORCH:
        """Calculate multi-focus normalized image variance.
        Inputs:
            orig_iwe (np.ndarray or torch.Tensor) ... Original IWE (before any warp).
            forward_iwe (np.ndarray or torch.Tensor) ... IWE to forward warp.
            backward_iwe (np.ndarray or torch.Tensor) ... IWE to backward warp.
            middle_iwe (Optional[np.ndarray or torch.Tensor]) ... IWE to middle warp.
            omit_bondary (bool) ... Omit boundary if True.

        Returns:
            average_time (Union[float, torch.Tensor]) ... Average timestamp.
        """
        orig_iwe = arg["orig_iwe"]
        forward_iwe = arg["forward_iwe"]
        if "middle_iwe" in arg.keys():
            middle_iwe = arg["middle_iwe"]
        else:
            middle_iwe = None
        backward_iwe = arg["backward_iwe"]
        omit_boundary = arg["omit_boundary"]
        if omit_boundary:
            forward_iwe = forward_iwe[..., 1:-1, 1:-1]  # omit boundary
            backward_iwe = backward_iwe[..., 1:-1, 1:-1]  # omit boundary
            if middle_iwe is not None:
                middle_iwe = middle_iwe[..., 1:-1, 1:-1]  # omit boundary

        if isinstance(forward_iwe, torch.Tensor):
            return self.calculate_torch(orig_iwe, forward_iwe, backward_iwe, middle_iwe)
        elif isinstance(forward_iwe, np.ndarray):
            return self.calculate_numpy(orig_iwe, forward_iwe, backward_iwe, middle_iwe)
        e = f"Unsupported input type. {type(forward_iwe)}."
        logger.error(e)
        raise NotImplementedError(e)

    def calculate_torch(
        self,
        orig_iwe: torch.Tensor,
        forward_iwe: torch.Tensor,
        backward_iwe: torch.Tensor,
        middle_iwe: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Calculate bytorch
        Inputs:
            orig_iwe (torch.Tensor) ... Original IWE (before warp).
            forward_iwe (torch.Tensor) ... IWE to forward warp.
            backward_iwe (torch.Tensor) ... IWE to backward warp.
            middle_iwe (Optional[torch.Tensor]) ... IWE to middle warp.

        Returns:
            loss (torch.Tensor) ... average time loss.
        """
        forward_loss = self.variance_loss.calculate_torch(forward_iwe, orig_iwe)
        backward_loss = self.variance_loss.calculate_torch(backward_iwe, orig_iwe)
        loss = forward_loss + backward_loss

        if middle_iwe is not None:
            loss += self.variance_loss.calculate_torch(middle_iwe, orig_iwe) * 2

        if self.direction in ["minimize", "natural"]:
            return loss
        logger.warning("The loss is specified as maximize direction")
        return -loss

    def calculate_numpy(
        self,
        orig_iwe: np.ndarray,
        forward_iwe: np.ndarray,
        backward_iwe: np.ndarray,
        middle_iwe: Optional[np.ndarray],
    ) -> float:
        """Calculate contrast of the count image.
        Inputs:
            orig_iwe (np.ndarray) ... Original IWE (before warp).
            forward_iwe (np.ndarray) ... IWE to forward warp.
            backward_iwe (np.ndarray) ... IWE to backward warp.
            middle_iwe (Optional[np.ndarray]) ... IWE to middle warp.

        Returns:
            loss (float) ... average time loss
        """
        forward_loss = self.variance_loss.calculate_numpy(forward_iwe, orig_iwe)
        backward_loss = self.variance_loss.calculate_numpy(backward_iwe, orig_iwe)
        loss = forward_loss + backward_loss

        if middle_iwe is not None:
            loss += self.variance_loss.calculate_numpy(middle_iwe, orig_iwe) * 2

        if self.direction in ["minimize", "natural"]:
            return loss
        logger.warning("The loss is specified as maximize direction")
        return -loss
