import logging
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import scipy
import skimage
import torch

from .. import costs, event_image_converter, types, utils, visualizer, warp
from . import scipy_autograd
from .base import SCIPY_OPTIMIZERS
from .patch_contrast_base import PatchContrastMaximization

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class PyramidalPatchContrastMaximization(PatchContrastMaximization):
    """Coarse-to-fine method patch-based CMax.

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
        logger.info("Pyramidal patch.")
        super().__init__(
            image_shape,
            calibration_parameter,
            solver_config,
            optimizer_config,
            output_config,
            visualize_module,
        )
        self.coarest_scale = 1
        self.patch_scales = self.slv_config["patch"]["scale"]
        self.cropped_height = self.slv_config["patch"]["crop_height"]
        self.cropped_width = self.slv_config["patch"]["crop_width"]
        self.cropped_image_shape = (self.cropped_height, self.cropped_width)
        self.prepare_pyramidal_patch(
            self.cropped_image_shape, self.coarest_scale, self.patch_scales
        )
        self.overload_patch_configuration(self.coarest_scale)
        self.patch_shift = (
            (self.image_shape[0] - self.cropped_height) // 2,
            (self.image_shape[1] - self.cropped_width) // 2,
        )
        self.loss_func_for_small_patch = costs.NormalizedGradientMagnitude(
            direction="minimize",
            store_history=False,
            precision="64",
            cuda_available=self._cuda_available,
        )

    def prepare_pyramidal_patch(self, image_size: tuple, coarest_scale: int, finest_scale: int):
        """To achieve pyramidal patch, set special member variables.
        You can use `overload_patch_configuration` to set the current scale.

        Args:
            image_size (tuple): [description]
            scales (int): [description]
        """
        self.scaled_patches = {}
        self.scaled_patch_image_size = {}
        self.scaled_n_patch = {}
        self.scaled_patch_size = {}
        self.scaled_sliding_window = {}
        self.total_n_patch = 0
        self.current_scale = self.coarest_scale
        self.scaled_imager = {}
        self.scaled_warper = {}
        for i in range(coarest_scale, finest_scale):
            scaled_size = (image_size[0] // (2**i), image_size[1] // (2**i))
            self.scaled_patch_size[i] = scaled_size
            self.scaled_sliding_window[i] = scaled_size
            self.scaled_patches[i], self.scaled_patch_image_size[i] = self.prepare_patch(
                image_size, scaled_size, scaled_size
            )
            self.scaled_n_patch[i] = len(self.scaled_patches[i].keys())
            self.total_n_patch += self.scaled_n_patch[i]
            self.scaled_imager[i] = event_image_converter.EventImageConverter(
                scaled_size, outer_padding=self.padding
            )
            self.scaled_warper[i] = warp.Warp(
                scaled_size, calculate_feature=False, normalize_t=self.normalize_t_in_batch
            )

    def overload_patch_configuration(self, n_scale: int):
        """Overload the related member variables set to the current scale.

        Args:
            n_scale (int): 0 is original size. 1 is half size, etc.
        """
        self.current_scale = n_scale
        self.patches = self.scaled_patches[n_scale]
        self.patch_image_size = self.scaled_patch_image_size[n_scale]
        self.patches = self.scaled_patches[n_scale]
        self.n_patch = self.scaled_n_patch[n_scale]
        self.sliding_window = self.scaled_sliding_window[n_scale]
        self.patch_size = self.scaled_patch_size[n_scale]
        # Cost weight by scale
        # scaled_cost_weight = self.cost_weight.copy()
        # for k in scaled_cost_weight.keys():
        #     if k in ["flow_smoothness", "total_variation"]:
        #         reg_scale_factor = self.current_scale
        #         scaled_cost_weight[k] *= reg_scale_factor
        # self.cost_func.update_weight(scaled_cost_weight)

        # This is only when you use Timestamp loss
        # self.loss_func_for_small_patch = costs.ZhuAverageTimestamp(
        #     direction="minimize",
        #     store_history=False,
        #     image_size=self.scaled_patch_size[n_scale]
        # )

    def get_motion_array_from_flatten(self, flatten_array: np.ndarray) -> dict:
        motion_dict = {}
        id = 0
        for s in range(self.coarest_scale, self.patch_scales):
            n_patch = self.scaled_n_patch[s]
            patch_image_size = self.scaled_patch_image_size[s]
            motion_dict[s] = flatten_array[:, id : id + n_patch].reshape((2,) + patch_image_size)
            id += n_patch
        return motion_dict

    def flatten_motion_array(self, motion_per_scale: dict) -> np.ndarray:
        motion_flatten = np.hstack(
            [
                motion_per_scale[s].reshape(2, -1)
                for s in range(self.coarest_scale, self.patch_scales)
            ]
        )
        return motion_flatten

    def optimize(self, events: np.ndarray) -> np.ndarray:
        """Run optimization.

        Inputs:
            events (np.ndarray) ... [n_events x 4] event array. Should be (x, y, t, p).
            n_iteration (int) ... How many iterations to run.

        """
        # Preprocessings
        logger.info("Start optimization.")
        logger.info(f"DoF is {self.motion_vector_size * self.total_n_patch}")

        best_motion_per_scale, opt_result = self.run_scipy_over_scale(events)
        logger.info(f"End optimization.")
        logger.debug(f"Best parameters: {best_motion_per_scale}")
        # Fine to coarse
        best_motion_per_scale_feedback = self.update_coarse_from_fine(best_motion_per_scale)

        logger.info("Profile file saved.")
        if self.visualizer:
            shutil.copy("optimize.prof", self.visualizer.save_dir)
            if self.opt_method in SCIPY_OPTIMIZERS:
                self.visualizer.visualize_scipy_history(
                    self.cost_func.get_history(), self.cost_weight
                )

        self.cost_func.clear_history()
        logger.debug(f"{best_motion_per_scale_feedback}")
        return best_motion_per_scale_feedback

    @utils.profile(
        output_file="optimize.prof", sort_by="cumulative", lines_to_print=300, strip_dirs=True
    )
    def run_scipy_over_scale(self, events):
        best_motion_per_scale = {}
        # Coarse to fine
        if self.opt_method in SCIPY_OPTIMIZERS:
            events = torch.from_numpy(events).double().requires_grad_().to(self._device)
        for s in range(self.coarest_scale, self.patch_scales):
            self.overload_patch_configuration(s)
            logger.info(f"Scale {self.current_scale}")
            if self.opt_method == "optuna":
                opt_result = self.run_optuna(np.copy(events))
                best_motion_per_scale[s] = self.get_motion_array_optuna(opt_result.best_params)
            elif self.opt_method in SCIPY_OPTIMIZERS:
                opt_result = self.run_scipy(events, best_motion_per_scale)
                best_motion_per_scale[s] = opt_result.x.reshape(
                    ((self.motion_vector_size,) + self.patch_image_size)
                )
            else:
                e = f"Optimizer {self.opt_method} is not supported"
                logger.error(e)
                raise NotImplementedError(e)

        return best_motion_per_scale, opt_result

    def update_coarse_from_fine(self, motion_per_scale: dict) -> dict:
        """Take average of finer motion and give it feedback toward coarser dimension.

        Args:
            motion_per_scale (dict): [description]

        Returns:
            [dict]: [description]
        """
        finest_scale = max(motion_per_scale.keys())
        coarsest_scale = min(motion_per_scale.keys())
        refined_motion = {finest_scale: motion_per_scale[finest_scale]}
        for i in range(finest_scale, coarsest_scale - 1, -1):
            # average_motion = skimage.transform.pyramid_reduce(motion_per_scale[i], channel_axis=0)
            # refined_motion[i - 1] = (average_motion + motion_per_scale[i - 1]) / 2.0
            refined_motion[i - 1] = skimage.transform.pyramid_reduce(
                motion_per_scale[i], channel_axis=0
            )
        return refined_motion

    # Optuna functions
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
    def run_scipy(self, events: np.ndarray, coarser_motion: dict) -> scipy.optimize.OptimizeResult:
        self.cost_func.disable_history_register()
        if (
            self.previous_frame_best_estimation is not None
            and self.current_scale == self.coarest_scale
        ):
            logger.info("Use previous best motion!")
            motion0 = np.copy(self.previous_frame_best_estimation[self.current_scale])
        elif self.current_scale > self.coarest_scale:
            logger.info("Use the coarser motion!")
            # motion0 = np.repeat(
            #     np.repeat(coarser_motion[self.current_scale - 1], 2, axis=1), 2, axis=2
            # ).reshape(-1)
            motion0 = skimage.transform.pyramid_expand(
                coarser_motion[self.current_scale - 1], channel_axis=0
            ).reshape(-1)
            if self.previous_frame_best_estimation is not None:
                motion0 = (
                    motion0 + self.previous_frame_best_estimation[self.current_scale].reshape(-1)
                ) / 2
            # if self.slv_config["patch"]["initialize"] == "random":
            # motion0 += (np.random.rand(motion0.shape[0]).astype(np.float64) - 0.5) * motion0 / 2
            # elif self.slv_config["patch"]["initialize"] == "optuna-sampling":
            motion0 = self.initialize_guess_from_optuna_sampling(
                events.clone().detach().cpu().numpy(), motion0
            )
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
                    best_guess = best_guess.detach().cpu().numpy()
                motion0 = np.tile(best_guess[None], (self.n_patch, 1)).T.reshape(-1)
            elif self.slv_config["patch"]["initialize"] == "grid-best":
                logger.info("sampling initialization")
                best_guess = self.initialize_guess_from_patch(
                    events, patch_index=self.n_patch // 2 - 1
                )
                if isinstance(best_guess, torch.Tensor):
                    best_guess = best_guess.detach().cpu().numpy()
                motion0 = np.tile(best_guess[None], (self.n_patch, 1)).T.reshape(-1)
            elif self.slv_config["patch"]["initialize"] == "optuna-sampling":
                logger.info("Optuna intelligent sampling initialization")
                motion0 = self.initialize_guess_from_optuna_sampling(events)
        self.cost_func.enable_history_register()

        result = scipy_autograd.minimize(
            lambda x: self.objective_scipy(x, events, coarser_motion),
            motion0,
            method=self.opt_method,
            options={
                "gtol": 1e-5,
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

    def initialize_guess_from_optuna_sampling(self, events: np.ndarray, motion0):
        # Using Optuna sampler, get best guess.
        motion1 = np.zeros((self.motion_vector_size, self.n_patch))
        for i in range(self.n_patch):
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=min(10, self.opt_config["n_iter"] // 5)
            )
            filtered_events = utils.crop_event(
                events,
                self.patches[i].x_min,
                self.patches[i].x_max,
                self.patches[i].y_min,
                self.patches[i].y_max,
            )
            filtered_events = utils.set_event_origin_to_zero(
                np.copy(filtered_events), self.patches[i].x_min, self.patches[i].y_min, 0
            )
            if len(filtered_events) > 10:
                opt_result = optuna.create_study(
                    direction="minimize",
                    sampler=sampler,
                )
                opt_result.optimize(
                    lambda trial: self.objective_initial(
                        trial, filtered_events, motion0.reshape(2, -1)[..., i]
                    ),
                    n_trials=self.opt_config["n_iter"]
                    / (
                        self.current_scale - self.coarest_scale
                    ),  # assume always current_scale > 1 here
                    # n_jobs=-1,
                )
                motion1[:, i] = np.array(
                    [
                        opt_result.best_params["trans_x"],
                        opt_result.best_params["trans_y"],
                    ]
                )
            else:
                motion1[:, i] = motion0.reshape(2, -1)[..., i]
        logger.debug(f"Initial value: {motion1 = }")
        return motion1

    def objective_initial(self, trial, events: np.ndarray, motion0):
        # Parameters setting
        params = {k: self.sampling_initial(trial, k, motion0) for k in self.motion_model_keys}
        motion_array = np.array([params["trans_x"], params["trans_y"]])
        if self.normalize_t_in_batch:
            t_scale = np.max(events[:, 2]) - np.min(events[:, 2])
            motion_array *= t_scale
        loss = self.calculate_cost_for_small_patch(
            events,
            motion_array,
            "2d-translation",
        )
        logger.debug(f"{trial.number = } / {loss = }")
        if np.isnan(loss):
            return 0.0
        return loss

    def calculate_cost_for_small_patch(
        self,
        events: np.ndarray,
        warp,
        motion_model: str,
    ):
        warper = self.scaled_warper[self.current_scale]
        imager = self.scaled_imager[self.current_scale]
        middle_events, _ = warper.warp_event(events, warp, motion_model, direction="middle")
        arg_cost = {"omit_boundary": False, "clip": True}
        orig_iwe = imager.create_iwe(
            events,
            self.iwe_config["method"],
            self.iwe_config["blur_sigma"],
        )
        arg_cost.update({"orig_iwe": orig_iwe})
        middle_iwe = imager.create_iwe(
            middle_events,
            self.iwe_config["method"],
            self.iwe_config["blur_sigma"],
        )
        arg_cost.update({"iwe": middle_iwe})

        # Only for Zhu Average Timestamp
        # arg_cost.update({"events": events})
        # forward_events, _ = warper.warp_event(events, warp, motion_model, direction="first")
        # backward_events, _ = warper.warp_event(events, warp, motion_model, direction="last")
        # arg_cost.update({"forward_warp": forward_events})
        # arg_cost.update({"backward_warp": backward_events})

        loss = self.loss_func_for_small_patch.calculate(arg_cost)
        if isinstance(loss, np.ndarray):
            if np.isnan(loss):
                logger.warning(f"Loss is nan")
                return 0.0
        return loss

    def sampling_initial(self, trial, key: str, motion0):
        abs_range = 10  # secrets paper
        # abs_range = 1
        if key == "trans_x":
            motion_range = np.array(
                [0.8 * motion0[0], motion0[0] - abs_range, 1.2 * motion0[0], motion0[0] + abs_range]
            )
        else:
            motion_range = np.array(
                [0.8 * motion0[1], motion0[1] - abs_range, 1.2 * motion0[1], motion0[1] + abs_range]
            )
        return trial.suggest_uniform(key, motion_range.min(), motion_range.max())

    def objective_scipy(
        self,
        motion_array: np.ndarray,
        events: np.ndarray,
        coarser_motion: dict,
        suppress_log: bool = False,
    ):
        """
        Args:
            motion_array (np.ndarray): [2 * n_patches] array. n_patches size depends on current_scale.

        Returns:
            [type]: [description]
        """
        if self.normalize_t_in_batch:
            t_scale = events[:, 2].max() - events[:, 2].min()
        else:
            t_scale = 1.0
        assert self.current_scale not in coarser_motion.keys()
        pyramidal_motion = coarser_motion.copy()
        pyramidal_motion.update({self.current_scale: motion_array})

        dense_flow = self.motion_to_dense_flow(pyramidal_motion, t_scale) * t_scale
        loss = self.calculate_cost(
            events,
            dense_flow,
            self.motion_model_for_dense_warp,
            motion_array.reshape((self.motion_vector_size,) + self.patch_image_size),
        )

        if not suppress_log:
            logger.info(f"{loss = }")
        return loss

    def motion_to_dense_flow(
        self,
        pyramidal_motion: Dict[int, types.NUMPY_TORCH],
        t_scale: float = 1.0,
    ) -> types.NUMPY_TORCH:
        """Returns dense flow for the pyramid.

        Args:
            pyramidal_motion (Dict[int, types.NUMPY_TORCH]): Dictionary holds each scale motion, [2 x h_patch x w_patch] array.

        Returns:
            types.NUMPY_TORCH: [2 x H x W]
        """
        finest_motion = pyramidal_motion[self.current_scale]
        if isinstance(finest_motion, torch.Tensor):
            dense_flow = self.interpolate_dense_flow_from_patch_tensor(finest_motion)
        elif isinstance(finest_motion, np.ndarray):
            dense_flow = self.interpolate_dense_flow_from_patch_numpy(finest_motion)
        else:
            e = f"Unsupported type: {type(finest_motion)}"
            raise TypeError(e)

        if not self.is_time_aware:
            return dense_flow

        if self.scale_later:
            scale = dense_flow.max()
        else:
            scale = 1.0

        if isinstance(dense_flow, np.ndarray):
            dense_flow_voxel = (
                utils.construct_dense_flow_voxel_numpy(
                    dense_flow * t_scale / scale,
                    self.time_bin,
                    self.flow_interpolation,
                    t0_location=self.t0_flow_location,
                )
                * scale
                / t_scale
            )
        elif isinstance(dense_flow, torch.Tensor):
            dense_flow_voxel = (
                utils.construct_dense_flow_voxel_torch(
                    dense_flow * t_scale / scale,
                    self.time_bin,
                    self.flow_interpolation,
                    t0_location=self.t0_flow_location,
                )
                * scale
                / t_scale
            )
        return dense_flow_voxel

    def visualize_one_batch_warp(self, events: np.ndarray, warp: Optional[dict] = None):
        if self.visualizer is None:
            return
        if warp is not None:
            flow = self.motion_to_dense_flow(warp)
            if self.normalize_t_in_batch:
                t_scale = np.max(events[:, 2]) - np.min(events[:, 2])
                flow *= t_scale
            events, _ = self.warper.warp_event(events, flow, self.motion_model_for_dense_warp)
            if self.is_time_aware:
                flow = self.get_original_flow_from_time_aware_flow_voxel(flow)

        clipped_iwe = self.create_clipped_iwe_for_visualization(
            events, max_scale=self.iwe_visualize_max_scale
        )
        self.visualizer.visualize_image(clipped_iwe)
        if warp is not None:
            self.visualizer.visualize_optical_flow_on_event_mask(flow, events)
            self.visualizer.visualize_overlay_optical_flow_on_event(flow, clipped_iwe)

    def visualize_pred_sequential(self, events: np.ndarray, warp: np.ndarray):
        """
        Args:
            events (np.ndarray): [description]
            pred_motion (np.ndarray)
        """
        if self.normalize_t_in_batch:
            t_scale = np.max(events[:, 2]) - np.min(events[:, 2])
        else:
            t_scale = 1.0
        flow = self.motion_to_dense_flow(warp, t_scale) * t_scale
        events, _ = self.warper.warp_event(
            events, flow, self.motion_model_for_dense_warp, direction="middle"
        )

        clipped_iwe = self.create_clipped_iwe_for_visualization(
            events, max_scale=self.iwe_visualize_max_scale
        )
        if self.is_time_aware:
            flow = self.get_original_flow_from_time_aware_flow_voxel(flow)
        self._pred_sequential(clipped_iwe, flow, events_for_mask=events)

    def calculate_flow_error(
        self,
        motion: np.ndarray,
        gt_flow: np.ndarray,
        timescale: float = 1.0,
        events: Optional[np.ndarray] = None,
    ) -> dict:
        """Calculate optical flow error based on GT.

        Args:
            motion (np.ndarray): Motion matrix, will be converted into dense flow. [pix/sec].
            gt_flow (np.ndarray): [H, W, 2]. Pixel displacement.
            timescale (float): To convert flow (pix/s) to displacement.

        Returns:
            dict: flow error dict.
        """
        gt_flow = np.transpose(gt_flow, (2, 0, 1))  # 2, H, W

        pred_flow = self.motion_to_dense_flow(motion, timescale) * timescale
        if self.is_time_aware:
            pred_flow = self.get_original_flow_from_time_aware_flow_voxel(pred_flow)[
                None
            ]  # [1, 2, H, W]
        else:
            pred_flow = pred_flow[None]
        if events is not None:
            event_mask = self.imager.create_eventmask(events)
            if self.padding:
                event_mask = event_mask[
                    ..., self.padding : -self.padding, self.padding : -self.padding
                ]
            fwl = self.calculate_fwl(motion, gt_flow, timescale, events)
        else:
            event_mask = None
            fwl = {}
        flow_error = utils.calculate_flow_error_numpy(gt_flow[None], pred_flow, event_mask=event_mask)  # type: ignore
        flow_error.update(fwl)
        logger.info(f"{flow_error = } for time period {timescale} sec.")
        return flow_error

    def calculate_fwl(
        self,
        motion: np.ndarray,
        gt_flow: np.ndarray,
        timescale: float,
        events: np.ndarray,
    ) -> dict:
        """Calculate FWL (from Stoffregen 2020)

        Args:
            motion (np.ndarray): Motion matrix, will be converted into dense flow. [pix/sec].
            gt_flow (np.ndarray): [2, H, W]. Pixel displacement.
            timescale (float): To convert flow (pix/s) to displacement.
            events (np.ndarray): [n, 4]

        Returns:
            dict: flow error dict.
        """
        orig_iwe = self.imager.create_iwe(events)
        gt_warper = warp.Warp(self.image_shape, normalize_t=True)
        gt_warp, _ = gt_warper.warp_event(events, gt_flow, "dense-flow")
        gt_iwe = self.imager.create_iwe(gt_warp)
        gt_fwl = costs.NormalizedImageVariance().calculate(
            {"orig_iwe": orig_iwe, "iwe": gt_iwe, "omit_boundary": False}
        )
        fwl = {"GT_FWL": gt_fwl}
        pred_fwl = self.calculate_fwl_pred(motion, events, timescale)
        fwl.update(pred_fwl)
        return fwl

    def calculate_fwl_pred(
        self,
        motion: np.ndarray,
        events: np.ndarray,
        timescale: float = 1.0,
    ) -> dict:
        """Calculate FWL (from Stoffregen 2020)
        ATTENTION this returns Var(IWE_orig) / Var(IWE) , Less than 1 is better.

        Args:
            motion (np.ndarray): Motion matrix, will be converted into dense flow. [pix/sec].
            gt_flow (np.ndarray): [2, H, W]. Pixel displacement.
            timescale (float): To convert flow (pix/s) to displacement.
            events (np.ndarray): [n, 4]

        Returns:
            dict: flow error dict.
        """
        orig_iwe = self.imager.create_iwe(events)
        pred_flow = self.motion_to_dense_flow(motion, timescale) * timescale
        pred_warp, _ = self.warper.warp_event(events, pred_flow, self.motion_model_for_dense_warp)
        pred_iwe = self.imager.create_iwe(pred_warp)
        pred_fwl = costs.NormalizedImageVariance().calculate(
            {"orig_iwe": orig_iwe, "iwe": pred_iwe, "omit_boundary": False}
        )
        fwl = {"PRED_FWL": pred_fwl}
        return fwl
