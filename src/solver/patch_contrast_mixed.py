import logging
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy
import torch

from .. import utils, visualizer
from . import scipy_autograd
from .base import SCIPY_OPTIMIZERS
from .patch_contrast_base import PatchContrastMaximization

logger = logging.getLogger(__name__)


class MixedPatchContrastMaximization(PatchContrastMaximization):
    """Mixed patch-based CMax.

    Params:
        image_shape (tuple) ... (H, W)
        calibration_parameter (dict) ... dictionary of the calibration parameter
        solver_config (dict) ... solver configuration
        optimizer_config (dict) ... optimizer configuration
        visualize_module ... visualizer.Visualizer
    """

    def __init__(
        self,
        image_shape: tuple,
        calibration_parameter: dict,
        solver_config: dict = {},
        optimizer_config: dict = {},
        output_config: dict = {},
        visualize_module: Optional[visualizer.Visualizer] = None,
    ):
        super().__init__(
            image_shape,
            calibration_parameter,
            solver_config,
            optimizer_config,
            output_config,
            visualize_module,
        )
        self.set_patch_size_and_sliding_window()
        self.patches, self.patch_image_size = self.prepare_patch(
            image_shape, self.patch_size, self.sliding_window
        )
        self.n_patch = len(self.patches.keys())

        # internal variable
        self._patch_motion_model_keys = [
            f"patch{i}_{k}" for i in range(self.n_patch) for k in self.motion_model_keys
        ]

    def optimize(self, events: np.ndarray) -> np.ndarray:
        """Run optimization.

        Inputs:
            events (np.ndarray) ... [n_events x 4] event array. Should be (x, y, t, p).
            n_iteration (int) ... How many iterations to run.

        """
        # Preprocessings
        logger.info("Start optimization.")
        logger.info(f"DoF is {self.motion_vector_size * self.n_patch}")

        if self.opt_method == "optuna":
            opt_result = self.run_optuna(events)
            logger.info(f"End optimization.")
            best_motion = self.get_motion_array_optuna(opt_result.best_params)
        elif self.opt_method in SCIPY_OPTIMIZERS:
            opt_result = self.run_scipy(events)
            logger.info(f"End optimization.\n Best parameters: {opt_result}")
            best_motion = opt_result.x.reshape(
                ((self.motion_vector_size,) + self.patch_image_size)
            )  # / 1000

        logger.info("Profile file saved.")
        if self.visualizer:
            shutil.copy("optimize.prof", self.visualizer.save_dir)
            if self.opt_method in SCIPY_OPTIMIZERS:
                self.visualizer.visualize_scipy_history(
                    self.cost_func.get_history(), self.cost_weight
                )

        logger.info(f"{best_motion}")
        return best_motion

    # Optuna functions
    def objective(self, trial, events: np.ndarray):
        # Parameters setting
        params = {k: self.sampling(trial, k) for k in self._patch_motion_model_keys}
        motion_array = self.get_motion_array_optuna(params)  # 2 x H x W
        if self.normalize_t_in_batch:
            t_scale = np.max(events[:, 2]) - np.min(events[:, 2])
            motion_array *= t_scale
        dense_flow = self.motion_to_dense_flow(motion_array)

        loss = self.calculate_cost(events, dense_flow, self.motion_model_for_dense_warp)
        logger.info(f"{trial.number = } / {loss = }")
        return loss

    def sampling(self, trial, key: str):
        """Sampling function for mixed type patch solution.

        Args:
            trial ([type]): [description]
            key (str): [description]

        Returns:
            [type]: [description]
        """
        key_suffix = key[key.find("_") + 1 :]
        return trial.suggest_uniform(
            key,
            self.opt_config["parameters"][key_suffix]["min"],
            self.opt_config["parameters"][key_suffix]["max"],
        )

    def get_motion_array_optuna(self, params: dict) -> np.ndarray:
        # Returns [n_patch x n_motion_paremter]
        motion_array = np.zeros((self.motion_vector_size, self.n_patch))
        for i in range(self.n_patch):
            param = {k: params[f"patch{i}_{k}"] for k in self.motion_model_keys}
            motion_array[:, i] = self.motion_model_to_motion(param)
        return motion_array.reshape((self.motion_vector_size,) + self.patch_image_size)

    # Scipy
    @utils.profile(
        output_file="optimize.prof", sort_by="cumulative", lines_to_print=300, strip_dirs=True
    )
    def run_scipy(self, events: np.ndarray) -> scipy.optimize.OptimizeResult:
        if self.previous_frame_best_estimation is not None:
            motion0 = np.copy(self.previous_frame_best_estimation)
        else:
            # Initialize with various methods
            if self.slv_config["patch"]["initialize"] == "random":
                motion0 = self.initialize_random()
            elif self.slv_config["patch"]["initialize"] == "zero":
                motion0 = self.initialize_zeros()
            elif self.slv_config["patch"]["initialize"] == "global-best":
                logger.info("sampling initialization")
                best_guess = self.initialize_guess_from_whole_image(events)
                if isinstance(best_guess, torch.Tensor):
                    motion0 = torch.tile(best_guess[None], (self.n_patch, 1)).T.reshape(-1)
                elif isinstance(best_guess, np.ndarray):
                    motion0 = np.tile(best_guess[None], (self.n_patch, 1)).T.reshape(-1)
            elif self.slv_config["patch"]["initialize"] == "grid-best":
                logger.info("sampling initialization")
                best_guess = self.initialize_guess_from_patch(
                    events, patch_index=self.n_patch // 2 - 1
                )
                if isinstance(best_guess, torch.Tensor):
                    motion0 = torch.tile(best_guess[None], (self.n_patch, 1)).T.reshape(-1)
                elif isinstance(best_guess, np.ndarray):
                    motion0 = np.tile(best_guess[None], (self.n_patch, 1)).T.reshape(-1)
                # motion0 += (
                #     np.random.rand(self.motion_vector_size * self.n_patch).astype(np.double64) * 10 - 5
                # )
            elif self.slv_config["patch"]["initialize"] == "optuna-sampling":
                logger.info("Optuna intelligent sampling initialization")
                motion0 = self.initialize_guess_from_optuna_sampling(events)
            self.cost_func.clear_history()

        self.events = torch.from_numpy(events).double().requires_grad_().to(self._device)
        result = scipy_autograd.minimize(
            self.objective_scipy,
            motion0,
            method=self.opt_method,
            options={
                "gtol": 1e-7,
                "disp": True,
                "maxiter": self.opt_config["max_iter"],
                "eps": 0.01,
            },
            precision="float64",
            torch_device=self._device,
            # TODO support bounds
            # bounds=[(-300, 300), (-300, 300)]
        )
        return result

    def objective_scipy(self, motion_array: np.ndarray, suppress_log: bool = False):
        """
        Args:
            motion_array (np.ndarray): [2 * n_patches] array

        Returns:
            [type]: [description]
        """
        if self.normalize_t_in_batch:
            t_scale = self.events[:, 2].max() - self.events[:, 2].min()
        else:
            t_scale = 1.0

        events = self.events.clone()
        dense_flow = self.motion_to_dense_flow(motion_array * t_scale)

        loss = self.calculate_cost(
            events,
            dense_flow,
            self.motion_model_for_dense_warp,
            motion_array.reshape((self.motion_vector_size,) + self.patch_image_size),
        )
        if not suppress_log:
            logger.info(f"{loss = }")
        return loss
