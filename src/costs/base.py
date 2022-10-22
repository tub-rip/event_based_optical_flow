import logging
from typing import Dict, List

import torch

from ..types import FLOAT_TORCH

logger = logging.getLogger(__name__)


class CostBase(object):
    """Base of the Cost class.
    Args:
        direction (str) ... 'minimize' or 'maximize' or `natural`.
            Defines the objective function. If natural, it returns more interpretable value.
    """

    required_keys: List[str] = []

    def __init__(self, direction="minimize", store_history: bool = False, *args, **kwargs):
        if direction not in ["minimize", "maximize", "natural"]:
            e = f"direction should be minimize, maximize, and natural. Got {direction}."
            logger.error(e)
            raise ValueError(e)
        self.direction = direction
        self.store_history = store_history
        self.clear_history()

    def catch_key_error(func):
        """Wrapper utility function to catch the key error."""

        def wrapper(self, arg: dict):
            try:
                return func(self, arg)  # type: ignore
            except KeyError as e:
                logger.error("Input for the cost needs keys of:")
                logger.error(self.required_keys)
                raise e

        return wrapper

    def register_history(func):
        """Registr history of the loss."""

        def wrapper(self, arg: dict):
            loss = func(self, arg)  # type: ignore
            if self.store_history:
                self.history["loss"].append(self.get_item(loss))
            return loss

        return wrapper

    def get_item(self, loss: FLOAT_TORCH) -> float:
        if isinstance(loss, torch.Tensor):
            return loss.item()
        return loss

    def clear_history(self) -> None:
        self.history: Dict[str, list] = {"loss": []}

    def get_history(self) -> dict:
        return self.history.copy()

    def enable_history_register(self) -> None:
        self.store_history = True

    def disable_history_register(self) -> None:
        self.store_history = False

    # Every subclass needs to implement calculate()
    @register_history  # type: ignore
    @catch_key_error  # type: ignore
    def calculate(self, arg: dict) -> FLOAT_TORCH:
        raise NotImplementedError

    catch_key_error = staticmethod(catch_key_error)  # type: ignore
    register_history = staticmethod(register_history)  # type: ignore
