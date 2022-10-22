import logging
from typing import Optional

import numpy as np
import torch

from ..types import FLOAT_TORCH
from . import CostBase, NormalizedGradientMagnitude

logger = logging.getLogger(__name__)


class MultiFocalNormalizedGradientMagnitude(CostBase):
    """Multi-focus normalized gradient magnitude, Shiba et al. ECCV 2022.

    Args:
        direction (str) ... 'minimize' or 'maximize' or `natural`.
            Defines the objective function. If natural, it returns more interpretable value.
    """

    name = "multi_focal_normalized_gradient_magnitude"
    required_keys = ["forward_iwe", "backward_iwe", "middle_iwe", "omit_boundary", "orig_iwe"]

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
        self.gradient_loss = NormalizedGradientMagnitude(
            direction=direction, cuda_available=cuda_available, precision=precision
        )

    @CostBase.register_history  # type: ignore
    @CostBase.catch_key_error  # type: ignore
    def calculate(self, arg: dict) -> FLOAT_TORCH:
        """Calculate cost.
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

        if isinstance(forward_iwe, torch.Tensor):
            return self.calculate_torch(
                orig_iwe, forward_iwe, backward_iwe, middle_iwe, omit_boundary
            )
        elif isinstance(forward_iwe, np.ndarray):
            return self.calculate_numpy(
                orig_iwe, forward_iwe, backward_iwe, middle_iwe, omit_boundary
            )
        e = f"Unsupported input type. {type(forward_iwe)}."
        logger.error(e)
        raise NotImplementedError(e)

    def calculate_torch(
        self,
        orig_iwe: torch.Tensor,
        forward_iwe: torch.Tensor,
        backward_iwe: torch.Tensor,
        middle_iwe: Optional[torch.Tensor],
        omit_boundary: bool,
    ) -> torch.Tensor:
        """Calculate cost for torch tensor.
        Inputs:
            orig_iwe (torch.Tensor) ... Original IWE (before warp).
            forward_iwe (torch.Tensor) ... IWE to forward warp.
            backward_iwe (torch.Tensor) ... IWE to backward warp.
            middle_iwe (Optional[torch.Tensor]) ... IWE to middle warp.

        Returns:
            loss (torch.Tensor) ... average time loss.
        """
        forward_loss = self.gradient_loss.calculate_torch(forward_iwe, orig_iwe, omit_boundary)
        backward_loss = self.gradient_loss.calculate_torch(backward_iwe, orig_iwe, omit_boundary)
        loss = forward_loss + backward_loss

        if middle_iwe is not None:
            loss += self.gradient_loss.calculate_torch(middle_iwe, orig_iwe, omit_boundary) * 2

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
        omit_boundary: bool,
    ) -> float:
        """Calculate cost for numpy array.
        Inputs:
            orig_iwe (np.ndarray) ... Original IWE (before warp).
            forward_iwe (np.ndarray) ... IWE to forward warp.
            backward_iwe (np.ndarray) ... IWE to backward warp.
            middle_iwe (Optional[np.ndarray]) ... IWE to middle warp.

        Returns:
            loss (float) ... average time loss
        """
        forward_loss = self.gradient_loss.calculate_numpy(forward_iwe, orig_iwe, omit_boundary)
        backward_loss = self.gradient_loss.calculate_numpy(backward_iwe, orig_iwe, omit_boundary)
        loss = forward_loss + backward_loss

        if middle_iwe is not None:
            loss += self.gradient_loss.calculate_numpy(middle_iwe, orig_iwe, omit_boundary) * 2

        if self.direction in ["minimize", "natural"]:
            return loss
        logger.warning("The loss is specified as maximize direction")
        return -loss
