import logging

import numpy as np
import torch

from ..types import FLOAT_TORCH
from . import CostBase, GradientMagnitude

logger = logging.getLogger(__name__)


class NormalizedGradientMagnitude(CostBase):
    """Normalized gradient magnitude.

    Args:
        direction (str) ... 'minimize' or 'maximize' or `natural`.
            Defines the objective function. If natural, it returns more interpretable value.
    """

    name = "normalized_gradient_magnitude"
    required_keys = ["orig_iwe", "iwe", "omit_boundary"]

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
        self.gradient_magnitude = GradientMagnitude(
            direction=direction,
            store_history=store_history,
            cuda_available=cuda_available,
            precision=precision,
        )

    @CostBase.register_history  # type: ignore
    @CostBase.catch_key_error  # type: ignore
    def calculate(self, arg: dict) -> FLOAT_TORCH:
        """Calculate normalized gradiend magnitude of IWE.
        Inputs:
            iwe (np.ndarray or torch.Tensor) ... [W, H]. Image of warped events
            orig_iwe (np.ndarray or torch.Tensor) ... [W, H]. Image of original events
            omit_bondary (bool) ... Omit boundary if True.

        Returns:
            contrast (FLOAT_TORCH) ... contrast of the image.
        """
        iwe = arg["iwe"]
        orig_iwe = arg["orig_iwe"]
        omit_boundary = arg["omit_boundary"]
        if isinstance(iwe, torch.Tensor):
            return self.calculate_torch(iwe, orig_iwe, omit_boundary)
        elif isinstance(iwe, np.ndarray):
            return self.calculate_numpy(iwe, orig_iwe, omit_boundary)
        e = f"Unsupported input type. {type(iwe)}."
        logger.error(e)
        raise NotImplementedError(e)

    def calculate_torch(
        self, iwe: torch.Tensor, orig_iwe: torch.Tensor, omit_boundary: bool
    ) -> torch.Tensor:
        """
        Inputs:
            iwe (torch.Tensor) ... [W, H]. Image of warped events
            orig_iwe (torch.Tensor) ... [W, H]. Image of original events

        Returns:
            loss (torch.Tensor) ... contrast of the image.
        """
        loss1 = self.gradient_magnitude.calculate_torch(iwe, omit_boundary)
        loss2 = self.gradient_magnitude.calculate_torch(orig_iwe, omit_boundary)
        if self.direction == "minimize":
            return loss2 / loss1
        logger.warning("The loss is specified as maximize direction")
        return loss1 / loss2

    def calculate_numpy(self, iwe: np.ndarray, orig_iwe: np.ndarray, omit_boundary: bool) -> float:
        """
        Inputs:
            iwe (np.ndarray) ... [W, H]. Image of warped events
            orig_iwe (np.ndarray) ... [W, H]. Image of original events

        Returns:
            contrast (float) ... contrast of the image.
        """
        loss1 = self.gradient_magnitude.calculate_numpy(iwe, omit_boundary)
        loss2 = self.gradient_magnitude.calculate_numpy(orig_iwe, omit_boundary)
        if self.direction == "minimize":
            return loss2 / loss1
        return loss1 / loss2
