import logging
from typing import Union

import numpy as np
import torch

from . import CostBase, functions

logger = logging.getLogger(__name__)


class HybridCost(CostBase):
    """Hybrid cost function with arbitrary weight.

    Args:
        direction (str) ... 'minimize' or 'maximize'.
        cost_with_weight (dict) ... key is the name of the cost, value is its weight.
    """

    name = "hybrid"

    def __init__(
        self, direction: str, cost_with_weight: dict, store_history: bool = False, *args, **kwargs
    ):
        logger.info(f"Log functions are mix of {cost_with_weight}")
        self.cost_func = {
            key: {
                "func": functions[key](
                    direction=direction, store_history=store_history, *args, **kwargs
                ),
                "weight": value,
            }
            for key, value in cost_with_weight.items()
        }
        super().__init__(direction=direction, store_history=store_history)

        self.required_keys = []
        for name in self.cost_func.keys():
            self.required_keys.extend(self.cost_func[name]["func"].required_keys)

    def update_weight(self, cost_with_weight):
        assert set(self.cost_func.keys()) == set(cost_with_weight.keys())
        for key in cost_with_weight.keys():
            self.cost_func[key]["weight"] = cost_with_weight[key]

    @CostBase.register_history  # type: ignore
    @CostBase.catch_key_error  # type: ignore
    def calculate(self, arg: dict) -> Union[float, torch.Tensor]:
        loss = 0.0
        for name in self.cost_func.keys():
            if self.cost_func[name]["weight"] == "inv":
                _l = 1.0 / self.cost_func[name]["func"].calculate(arg)
                loss += _l
            else:
                _l = self.cost_func[name]["weight"] * self.cost_func[name]["func"].calculate(arg)
                loss += _l
        return loss

    # For hybrid cost function, need to store with its name
    def clear_history(self) -> None:
        self.history = {"loss": []}
        for name in self.cost_func.keys():
            self.cost_func[name]["func"].clear_history()

    def get_history(self) -> dict:
        dic = self.history.copy()
        for name in self.cost_func.keys():
            dic.update({name: self.cost_func[name]["func"].get_history()["loss"]})
        return dic

    def enable_history_register(self) -> None:
        self.store_history = True
        for name in self.cost_func.keys():
            self.cost_func[name]["func"].store_history = True

    def disable_history_register(self) -> None:
        self.store_history = False
        for name in self.cost_func.keys():
            self.cost_func[name]["func"].store_history = False
