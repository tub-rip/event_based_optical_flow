import logging
import math
import os
import shutil
from typing import List, Optional

import cv2
import numpy as np
import optuna
import scipy
import torch

from .. import costs, event_image_converter, utils, visualizer, warp
from ..types import NUMPY_TORCH
from . import scipy_autograd

logger = logging.getLogger(__name__)


# List of scipy optimizers supported
SCIPY_OPTIMIZERS = [
    "Nelder-Mead",
    "Powell",
    "CG",
    "BFGS",
    "Newton-CG",
    "L-BFGS-B",
    "TNC",
    "COBYLA",
    "SLSQP",
    "trust-constr",
    "dogleg",  # requires positive semi definite hessian
    "trust-ncg",
    "trust-exact",  # requires hessian
    "trust-krylov",
]

TORCH_OPTIMIZERS = [
    "Adadelta",
    "Adagrad",
    "Adam",
    "AdamW",
    "SparseAdam",
    "Adamax",
    "ASGD",
    "LBFGS",
    "NAdam",
    "RAdam",
    "RMSprop",
    "Rprop",
    "SGD",
]


class SingleThreadInMemoryStorage(optuna.storages.InMemoryStorage):
    """This is faster version of in-memory storage only when the study n_jobs = 1 (single thread).

    Args:
        optuna ([type]): [description]
    """

    def set_trial_param(
        self,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: optuna.distributions.BaseDistribution,
    ) -> None:
        with self._lock:
            trial = self._get_trial(trial_id)
            self.check_trial_is_updatable(trial_id, trial.state)

            study_id = self._trial_id_to_study_id_and_number[trial_id][0]
            # Check param distribution compatibility with previous trial(s).
            if param_name in self._studies[study_id].param_distribution:
                optuna.distributions.check_distribution_compatibility(
                    self._studies[study_id].param_distribution[param_name], distribution
                )
            # Set param distribution.
            self._studies[study_id].param_distribution[param_name] = distribution

            # Set param.
            trial.params[param_name] = distribution.to_external_repr(param_value_internal)
            trial.distributions[param_name] = distribution


class SolverBase(object):
    """Base class for solver.

    Params:
        image_shape (tuple) ... (H, W)
        calibration_parameter (dict) ... dictionary of the calibration parameter
        solver_config (dict) ... solver configuration
        optimizer_config (dict) ... optimizer configuration
        output_config (dict) ... Output configuration
        visualize_module ... visualizer.Visualizer
    """

    def __init__(
        self,
        image_shape: tuple,
        calibration_parameter: dict = {},
        solver_config: dict = {},
        optimizer_config: dict = {},
        output_config: dict = {},
        visualize_module: Optional[visualizer.Visualizer] = None,
    ):
        self.image_shape = image_shape
        self.padding = (
            solver_config["outer_padding"] if "outer_padding" in solver_config.keys() else 0
        )
        self.pad_image_shape = (image_shape[0] + self.padding, image_shape[1] + self.padding)
        self.calib_param = calibration_parameter
        self.opt_config = optimizer_config
        self.opt_method = optimizer_config["method"]
        if self.opt_method == "optuna":
            if "sampler" in self.opt_config.keys():
                self.sampling_method = optimizer_config["sampler"]
            else:
                self.sampling_method = "TPE"
            logger.info(f"Sampler is {self.sampling_method}")
        self.slv_config = solver_config
        self.out_config = output_config
        self.iwe_config = solver_config["iwe"]
        self.visualizer = visualize_module

        # Cuda utilization
        self._cuda_available = torch.cuda.is_available()
        if self._cuda_available:
            logger.info("Use cuda!")
            self._device = "cuda"
        else:
            self._device = "cpu"

        self.setup_cost_func()

        # Motion model and parameter configuration
        self.normalize_t_in_batch = True
        self.imager = event_image_converter.EventImageConverter(
            self.image_shape, outer_padding=self.padding
        )
        self.warper = warp.Warp(
            self.image_shape,
            calculate_feature=True,
            normalize_t=self.normalize_t_in_batch,
            calib_param=self.calib_param,
        )
        self.warp_direction = (
            self.slv_config["warp_direction"]
            if "warp_direction" in self.slv_config.keys()
            else "first"
        )

        self.previous_frame_best_estimation = None
        self.motion_model = self.slv_config["motion_model"]
        self.motion_model_keys = self.warper.get_key_names(self.motion_model)
        self.motion_vector_size = self.warper.get_motion_vector_size(self.motion_model)
        self.param_keys = self.slv_config["parameters"]
        self.setup_time_aware()

        self.iwe_visualize_max_scale = (
            50 if not "max_scale" in self.slv_config.keys() else self.slv_config["max_scale"]
        )

        logger.info(f"Configuration: \n    {self.normalize_t_in_batch = }")
        logger.info(f"Configuration: \n    {self.opt_config} \n    {self.slv_config}")

    def init_calib_param(self, calibration_parameter: dict):
        self.calib_param = calibration_parameter
        self.warper = warp.Warp(
            self.image_shape,
            calculate_feature=True,
            normalize_t=self.normalize_t_in_batch,
            calib_param=self.calib_param,
        )

    def setup_cost_func(self):
        # Cost function configuration
        percentile = 1.0
        precision = "64"
        try:
            if self.slv_config["cost"] == "hybrid":
                logger.info(f"Load hybrid cost")
                self.cost_weight = self.slv_config["cost_with_weight"]
                self.cost_func = costs.HybridCost(
                    direction="minimize",
                    cost_with_weight=self.cost_weight,
                    store_history=True,
                    image_size=self.pad_image_shape,
                    percentile=percentile,
                    precision=precision,
                    cuda_available=self._cuda_available,
                )
            else:
                logger.info(f"Load {costs.functions[self.slv_config['cost']]}")
                self.cost_weight = None
                self.cost_func = costs.functions[self.slv_config["cost"]](
                    direction="minimize",
                    store_history=True,
                    image_size=self.pad_image_shape,
                    percentile=percentile,
                    precision=precision,
                    cuda_available=self._cuda_available,
                )
        except KeyError as e:
            logger.error(
                f"Your cost function {self.slv_config['cost']} is not supported. \n Supported functions are {costs.functions}"
            )
            raise e

    def setup_time_aware(self):
        if utils.check_key_and_bool(self.slv_config, "time_aware"):
            logger.info("Setup time-aware parameters")
            self.is_time_aware = True
            self.motion_model_for_dense_warp = "dense-flow-voxel"
            self.time_bin = self.slv_config["time_bin"]
            self.flow_interpolation = self.slv_config["flow_interpolation"]
            self.t0_flow_location = self.slv_config["t0_flow_location"]
            if utils.check_key_and_bool(self.slv_config, "scale_later"):
                logger.info("Scaling before upwind")
                self.scale_later = True
            else:
                logger.info("No scaling before upwind")
                self.scale_later = False
        else:
            logger.info("Setup time-ignorant parameters")
            self.is_time_aware = False
            self.motion_model_for_dense_warp = "dense-flow"

    def get_original_flow_from_time_aware_flow_voxel(self, flow_voxel: NUMPY_TORCH) -> NUMPY_TORCH:
        """Get original (not interpolated) flow slice from voxel.

        Args:
            flow_voxel (NUMPY_TORCH): [(b, ) time_bin, 2, H, W]

        Returns:
            NUMPY_TORCH: [(b, ) 2, H, W]
        """
        if len(flow_voxel.shape) == 4:  # time_bin, 2, H, W
            flow_voxel = flow_voxel[None]  # batch, t, 2, H, W
        if self.t0_flow_location == "first":
            orig_ind = 0
        elif self.t0_flow_location == "middle":
            orig_ind = flow_voxel.shape[1] // 2

        if isinstance(flow_voxel, torch.Tensor):
            return flow_voxel[:, orig_ind].squeeze()
        elif isinstance(flow_voxel, np.ndarray):
            return np.squeeze(flow_voxel[:, orig_ind])
        raise NotImplementedError

    def motion_model_to_motion(self, params: dict) -> np.ndarray:
        """Returns 2D or 3D motion."""
        return self.warper.motion_model_to_motion(self.motion_model, params)

    def motion_to_motion_model(self, motion: np.ndarray) -> np.ndarray:
        """Returns 2D or 3D motion."""
        return self.warper.motion_model_from_motion(motion, self.motion_model)

    def motion_to_dense_flow(self, motion: np.ndarray) -> np.ndarray:
        """Convert motion to dense flow.
        Args:
            motion (np.ndarray): [description]

        Returns:
            np.ndarray: 2 x H x W
        """
        return self.warper.get_flow_from_motion(motion, self.motion_model)

    # Visualizations
    # Helper function
    def create_clipped_iwe_for_visualization(self, events: np.ndarray, max_scale=50):
        """Creeate IWE for visualization.

        Args:
            events (_type_): _description_
            max_scale (int, optional): _description_. Defaults to 50.

        Returns:
            _type_: np.ndarray
        """
        assert events.shape[-1] <= 4, "this function is for events"
        if isinstance(events, torch.Tensor):
            events = events.clone().detach().cpu().numpy()
        im = self.imager.create_image_from_events_numpy(
            events, method=self.iwe_config["method"], sigma=0
        )
        clipped_iwe = 255 - np.clip(max_scale * im, 0, 255).astype(np.uint8)
        if self.padding > 0:
            clipped_iwe = clipped_iwe[self.padding : -self.padding, self.padding : -self.padding]
        return clipped_iwe

    def visualize_one_batch_warp(self, events: np.ndarray, warp: Optional[np.ndarray] = None):
        if self.visualizer is None:
            return
        if warp is not None:
            if isinstance(warp, torch.Tensor):
                warp = warp.clone().detach().cpu().numpy()
            else:
                warp = np.copy(warp)
            if self.normalize_t_in_batch:
                t_scale = np.max(events[:, 2]) - np.min(events[:, 2])
                warp *= t_scale
            events, _ = self.warper.warp_event(events, warp, self.motion_model)
            flow = self.motion_to_dense_flow(warp)
        clipped_iwe = self.create_clipped_iwe_for_visualization(
            events, max_scale=self.iwe_visualize_max_scale
        )
        self.visualizer.visualize_image(clipped_iwe)
        if warp is not None:
            self.visualizer.visualize_optical_flow_on_event_mask(flow, events)

    def visualize_one_batch_warp_gt(
        self, events: np.ndarray, gt_warp: np.ndarray, motion_model: str = "dense-flow"
    ):
        """
        Args:
            events (np.ndarray): [description]
            gt_warp (np.ndarray): If flow, [H, W, 2]. If other, [motion_dim].
            motion_model (str): motion model, defaults to 'dense-flow'
        """
        if motion_model == "dense-flow":
            gt_warp = np.transpose(gt_warp, (2, 0, 1))  # [2, H, W]
        events, _ = self.warper.warp_event(events, gt_warp, motion_model=motion_model)
        clipped_iwe = self.create_clipped_iwe_for_visualization(
            events, max_scale=self.iwe_visualize_max_scale
        )
        self.visualizer.visualize_image(clipped_iwe)  # type: ignore
        if motion_model == "dense-flow":
            self.visualizer.visualize_overlay_optical_flow_on_event(gt_warp, clipped_iwe)  # type: ignore

    def visualize_original_sequential(self, events: np.ndarray):
        """Visualize sequential, original image
        Args:
            events (np.ndarray): [description]
            pred_motion (np.ndarray)
        """
        clipped_iwe = self.create_clipped_iwe_for_visualization(
            events, max_scale=self.iwe_visualize_max_scale
        )
        self.visualizer.visualize_image(clipped_iwe, file_prefix="original")  # type: ignore

    def visualize_pred_sequential(self, events: np.ndarray, warp: np.ndarray):
        """Visualize sequential, prediction
        Args:
            events (np.ndarray): [description]
            pred_motion (np.ndarray)
        """
        if self.normalize_t_in_batch:
            warp = np.copy(warp)
            t_scale = np.max(events[:, 2]) - np.min(events[:, 2])
            warp *= t_scale
        # TODO giev config param of paper
        # events, _ = self.warper.warp_event(events, warp, self.motion_model, direction="middle")  # for ECCV secret paper
        events, feat = self.warper.warp_event(
            events, warp, self.motion_model, direction="first"
        )  # for collapse paper
        flow = self.motion_to_dense_flow(warp)
        clipped_iwe = self.create_clipped_iwe_for_visualization(
            events, max_scale=self.iwe_visualize_max_scale
        )

        self._pred_sequential(clipped_iwe, flow)

    def _pred_sequential(
        self,
        iwe: np.ndarray,
        flow: np.ndarray,
        events_for_mask: Optional[np.ndarray] = None,
    ):
        """
        Args:
            iwe (np.ndarray): [description]
            flow (np.ndarray): [description]
            with_grid (bool, optional): [description]. Defaults to False.
            events_for_mask (Optional[np.ndarray], optional): [description]. Defaults to None.
        """
        if self.visualizer is None:
            return
        self.visualizer.visualize_image(iwe, file_prefix="pred_warp")  # type: ignore

        if events_for_mask is not None:
            self.visualizer.visualize_optical_flow_on_event_mask(flow, events_for_mask, file_prefix="pred_masked")  # type: ignore

    def visualize_gt_sequential(
        self, events: np.ndarray, gt_warp: np.ndarray, gt_type: str = "flow"
    ):
        """Visualize sequential, GT
        Args:
            events (np.ndarray): [description]
            gt_warp (np.ndarray): if flow, [H, W, 2]; otherwise [n-dim]
        """
        if self.visualizer is None:
            return
        if gt_type == "flow":
            motion_model = "dense-flow"
            gt_warp = np.transpose(gt_warp, (2, 0, 1))  # [2, H, W]
        else:
            motion_model = self.motion_model

        # events, _ = self.warper.warp_event(events, gt_warp, motion_model, direction="middle") # for ECCV secret paper
        events, feat = self.warper.warp_event(
            events, gt_warp, motion_model, direction="first"
        )  # for collapse paper
        clipped_iwe = self.create_clipped_iwe_for_visualization(
            events, max_scale=self.iwe_visualize_max_scale
        )
        self.visualizer.visualize_image(clipped_iwe, file_prefix="gt_warp")  # type: ignore

        # Flow
        if motion_model != "dense-flow":
            gt_flow = self.motion_to_dense_flow(gt_warp)
        else:
            gt_flow = gt_warp
        self.visualizer.visualize_optical_flow(
            gt_flow[0],
            gt_flow[1],
            visualize_color_wheel=False,
            file_prefix="gt_flow",
        )

    def visualize_test_sequential(
        self,
        events: np.ndarray,
        test_flow: np.ndarray,
        submission_index: int,
    ):
        """Visualize sequential, GT
        Args:
            events (np.ndarray): [description]
            test_flow (np.ndarray): it's displacement
            submission_index (int) ... submission int ID
        """
        if self.visualizer is None:
            return
        # events, _ = self.warper.warp_event(events, test_flow, "dense-flow", direction="middle")  # for ECCV secret paper
        events, _ = self.warper.warp_event(
            events, test_flow, "dense-flow", direction="first"
        )  # for collapse paper
        clipped_iwe = self.create_clipped_iwe_for_visualization(
            events, max_scale=self.iwe_visualize_max_scale
        )
        submission_format = "{:06d}".format(submission_index)
        self.visualizer.visualize_image(clipped_iwe, file_prefix=f"test_warp_{submission_format}")  # type: ignore
        self.visualizer.visualize_optical_flow(
            test_flow[0],
            test_flow[1],
            visualize_color_wheel=False,
            file_prefix=f"test_flow_{submission_format}",
            save_flow=False,
        )
        self.visualizer.visualize_overlay_optical_flow_on_event(
            test_flow, clipped_iwe, file_prefix=f"test_overlay_{submission_format}"
        )  # type: ignore
        self.visualizer.visualize_optical_flow_on_event_mask(
            test_flow, events, file_prefix=f"test_masked_{submission_format}"
        )  # type: ignore

    def visualize_flows(
        self,
        motion: np.ndarray,
        gt_flow: np.ndarray,
        timescale: float = 1.0,
    ) -> None:
        """Visualize the comparison between predicted motion and GT optical flow.

        Args:
            motion (np.ndarray): Motion matrix, will be converted into dense flow. [pix/sec].
            gt_flow (np.ndarray): [H, W, 2]. Pixel displacement.
            timescale (float): To convert flow (pix/s) to displacement.
        """
        pred_flow = self.motion_to_dense_flow(motion * timescale)  # [2, H, W]
        if self.is_time_aware:
            pred_flow = self.get_original_flow_from_time_aware_flow_voxel(pred_flow)
        gt_flow = np.transpose(gt_flow, (2, 0, 1))  # [2, H, W]
        self.visualizer.visualize_optical_flow_pred_and_gt(  # type: ignore
            pred_flow,
            gt_flow,
            pred_file_prefix="flow_comparison_pred",
            gt_file_prefix="flow_comparison_gt",
        )

    def calculate_pose_error(
        self,
        motion: np.ndarray,
        gt_motion: np.ndarray,
        events: np.ndarray,
        timescale: float = 1.0,
        motion_model: str = "3d-rotation",
    ) -> dict:
        """Calculate motion array error based on GT.

        Args:
            motion (np.ndarray): Motion matrix, will be converted into dense flow. [pix/sec].
            gt_flow (np.ndarray): [H, W, 2]. Pixel displacement.
            timescale (float): To convert flow (pix/s) to displacement.

        Returns:
            dict: flow error dict.
        """
        if isinstance(motion, torch.Tensor):
            pred_speed = motion.clone().detach().cpu().numpy()
        else:
            pred_speed = motion
        gt_speed = gt_motion / timescale
        # gt_speed = gt_motion
        l1_error = gt_speed - pred_speed
        l1_deg = np.rad2deg(l1_error)
        logger.info(f"{pred_speed = } / sec.")
        logger.info(f"{gt_speed = } / sec.")
        logger.info(f"{l1_error = } ({l1_deg} deg) / sec.")
        pose_error = {
            "L1-rad/x": l1_error[0],
            "L1-rad/y": l1_error[1],
            "L1-rad/z": l1_error[2],
            "L1-deg/x": l1_deg[0],
            "L1-deg/y": l1_deg[1],
            "L1-deg/z": l1_deg[2],
            "pred_speed/x": pred_speed[0],
            "pred_speed/y": pred_speed[1],
            "pred_speed/z": pred_speed[2],
            "gt_speed/x": gt_speed[0],
            "gt_speed/y": gt_speed[1],
            "gt_speed/z": gt_speed[2],
        }
        fwl = self.calculate_fwl(
            pred_speed, gt_motion, timescale, events, motion_model=motion_model
        )
        pose_error.update(fwl)
        return pose_error

    def save_pose_error_as_text(
        self, nth_frame: int, pose_error_dict: dict, fname: str = "pose_error_per_frame.txt"
    ):
        if self.visualizer is not None:
            save_file_name = os.path.join(self.visualizer.save_dir, fname)
        else:
            save_file_name = fname

        with open(save_file_name, "a") as f:
            f.write(f"frame {nth_frame}::" + str(pose_error_dict) + "\n")

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
        pred_flow = self.motion_to_dense_flow(motion * timescale)
        if self.is_time_aware:
            pred_flow = self.get_original_flow_from_time_aware_flow_voxel(pred_flow)
        pred_flow = pred_flow[None]  # [1, 2, H, W]

        if events is not None:
            event_mask = self.imager.create_eventmask(events)
            fwl = self.calculate_fwl(motion, gt_flow, timescale, events)
            if self.padding > 0:
                event_mask = event_mask[
                    ..., self.padding : -self.padding, self.padding : -self.padding
                ]
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
        gt_motion: np.ndarray,
        timescale: float,
        events: np.ndarray,
        motion_model: Optional[str] = None,
    ) -> dict:
        """Calculate FWL (from Stoffregen 2020)
        ATTENTION this returns Var(IWE_orig) / Var(IWE) , **Less than 1 is better alignment.**

        Args:
            motion (np.ndarray): Motion matrix, will be converted into dense flow. [pix/sec].
            gt_flow (np.ndarray): [2, H, W]. Pixel displacement.
            timescale (float): To convert flow (pix/s) to displacement.
            events (np.ndarray): [n, 4]

        Returns:
            dict: flow error dict.
        """
        orig_iwe = self.imager.create_iwe(events)
        gt_warper = warp.Warp(self.image_shape, normalize_t=True, calib_param=self.calib_param)
        if motion_model is None:
            gt_warp, _ = gt_warper.warp_event(events, gt_motion, "dense-flow")
        else:
            gt_warp, _ = gt_warper.warp_event(events, gt_motion, motion_model)
        gt_iwe = self.imager.create_iwe(gt_warp)
        gt_fwl = costs.NormalizedImageVariance().calculate(
            {"orig_iwe": orig_iwe, "iwe": gt_iwe, "omit_boundary": False}
        )
        fwl = {"GT_FWL": gt_fwl}
        pred_fwl = self.calculate_fwl_pred(motion, events, timescale, motion_model)
        fwl.update(pred_fwl)
        return fwl

    def calculate_fwl_pred(
        self,
        motion: np.ndarray,
        events: np.ndarray,
        timescale: float = 1.0,
        motion_model: Optional[str] = None,
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
        if motion_model is None:
            pred_flow = self.motion_to_dense_flow(motion * timescale)
            pred_warp, _ = self.warper.warp_event(
                events, pred_flow, self.motion_model_for_dense_warp
            )
        else:
            pred_warp, _ = self.warper.warp_event(events, motion * timescale, motion_model)
        pred_iwe = self.imager.create_iwe(pred_warp)
        # pred_fwl = costs.NormalizedImageVariance(direction='maximize').calculate(
        pred_fwl = costs.NormalizedImageVariance().calculate(
            {"orig_iwe": orig_iwe, "iwe": pred_iwe, "omit_boundary": False}
        )
        fwl = {"PRED_FWL": pred_fwl}
        return fwl

    def save_flow_error_as_text(
        self, nth_frame: int, flow_error_dict: dict, fname: str = "flow_error_per_frame.txt"
    ):
        if self.visualizer is not None:
            save_file_name = os.path.join(self.visualizer.save_dir, fname)
        else:
            save_file_name = fname

        with open(save_file_name, "a") as f:
            f.write(f"frame {nth_frame}::" + str(flow_error_dict) + "\n")

    def set_previous_frame_best_estimation(self, previous_best: np.ndarray):
        if isinstance(previous_best, np.ndarray):
            self.previous_frame_best_estimation = np.copy(previous_best)
        elif isinstance(previous_best, torch.Tensor):
            self.previous_frame_best_estimation = torch.clone(previous_best)
        elif isinstance(previous_best, dict):
            self.previous_frame_best_estimation = previous_best.copy()

    def update_time_scale_for_previous_frame_best_estimation(self, scale: float):
        if isinstance(self.previous_frame_best_estimation, dict):
            self.previous_frame_best_estimation = {
                k: v * scale for (k, v) in self.previous_frame_best_estimation.items()
            }
        else:
            self.previous_frame_best_estimation *= scale

    # Main optimization function
    def optimize(self, events: np.ndarray) -> np.ndarray:
        """Run optimization.

        Inputs:
            events (np.ndarray) ... [n_events x 4] event array. Should be (x, y, t, p).

        Returns:
            (np.ndarray) ... Best motion array.
        """
        # Preprocessings
        logger.info("Start optimization.")
        time_period = events[:, 2].max() - events[:, 2].min()
        logger.info(f"Event stats: {len(events)} events, in {time_period} sec.")

        if self.opt_method == "optuna":
            study = self.run_optuna(events)  # Main function
            logger.info(
                f"End optimization.\n Best parameters: {study.best_params}, Cost: {study.best_value}"
            )
            logger.info("Profile file saved.")
            if self.visualizer:
                shutil.copy("optimize.prof", self.visualizer.save_dir)
                self.visualizer.visualize_optuna_history(study)
                self.visualizer.visualize_optuna_study(study, params=self.param_keys)
            best_motion = self.motion_model_to_motion(study.best_params)
        elif self.opt_method in SCIPY_OPTIMIZERS:
            opt_result = self.run_scipy(events)
            logger.info(f"End optimization.\n Best parameters: {opt_result}")
            if self.visualizer:
                shutil.copy("optimize.prof", self.visualizer.save_dir)
            logger.info("Profile file saved.")
            best_motion = opt_result.x
        elif self.opt_method in TORCH_OPTIMIZERS:
            opt_result = self.run_torch(events)
            logger.info(f"End optimization.\n Best parameters: {opt_result}")
            if self.visualizer:
                shutil.copy("optimize.prof", self.visualizer.save_dir)
            logger.info("Profile file saved.")
            best_motion = opt_result["param"]
        else:
            e = f"Optimize algorithm {self.opt_method} is not supported"
            logger.error(e)
            raise NotImplementedError(e)

        if self.visualizer:
            shutil.copy("optimize.prof", self.visualizer.save_dir)
            if self.opt_method == "optuna":
                self.visualizer.visualize_optuna_history(study)
            elif self.opt_method in SCIPY_OPTIMIZERS or self.opt_method in TORCH_OPTIMIZERS:
                self.visualizer.visualize_scipy_history(self.cost_func.get_history())

        logger.info(f"Best: {best_motion}")
        self.cost_func.clear_history()
        return best_motion

    # Optuna solver functions
    @utils.profile(
        output_file="optimize.prof", sort_by="cumulative", lines_to_print=300, strip_dirs=True
    )
    def run_optuna(self, events: np.ndarray) -> optuna.study.Study:
        if self.sampling_method == "TPE":
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=max(10, self.opt_config["n_iter"] // 10)
            )
        elif self.sampling_method == "random":
            sampler = optuna.samplers.RandomSampler()
        elif self.sampling_method in ["grid", "uniform"]:
            sampler = self.uniform_sampling()
        else:
            e = f"Sampling method {self.sampling_method} is not supported"
            logger.error(e)
            raise NotImplementedError(e)

        study = optuna.create_study(
            direction="minimize", sampler=sampler, storage=SingleThreadInMemoryStorage()
        )
        study.optimize(
            lambda trial: self.objective(trial, events), n_trials=self.opt_config["n_iter"]
        )
        return study

    def uniform_sampling(self):
        if self.previous_frame_best_estimation is not None:
            previous_best = self.motion_to_motion_model(self.previous_frame_best_estimation)
            min_max = {
                k: [
                    min(previous_best[k] * 0.5, previous_best[k] * 1.5) - 1,
                    max(previous_best[k] * 0.5, previous_best[k] * 1.5) + 1,
                ]
                for k in self.motion_model_keys
            }
        else:
            min_max = {
                k: [
                    self.opt_config["parameters"][k]["min"],
                    self.opt_config["parameters"][k]["max"],
                ]
                for k in self.motion_model_keys
            }
        search_space = {
            k: np.arange(
                min_max[k][0],
                min_max[k][1],
                (min_max[k][1] - min_max[k][0]) / self.opt_config["n_iter"],  # type: ignore
            )
            for k in self.motion_model_keys
        }
        sampler = optuna.samplers.GridSampler(search_space)
        return sampler

    def sampling(self, trial, key: str):
        if self.previous_frame_best_estimation is not None:
            previous_best = self.motion_to_motion_model(self.previous_frame_best_estimation)
            min_val = min(previous_best[key] * 0.5, previous_best[key] * 1.5) - 1
            max_val = max(previous_best[key] * 0.5, previous_best[key] * 1.5) + 1
            return trial.suggest_uniform(key, min_val, max_val)
        else:
            return trial.suggest_uniform(
                key,
                self.opt_config["parameters"][key]["min"],
                self.opt_config["parameters"][key]["max"],
            )

    def objective(self, trial, events: np.ndarray):
        raise NotImplementedError

    # Scipy solver functions
    def objective_scipy(self, motion: np.ndarray):
        raise NotImplementedError

    @utils.profile(
        output_file="optimize.prof", sort_by="cumulative", lines_to_print=300, strip_dirs=True
    )
    def run_scipy(self, events: np.ndarray) -> scipy.optimize.OptimizeResult:
        self.events = events  # use this variable because currently not supported as passing arg
        if self.previous_frame_best_estimation is not None:
            x0 = np.copy(self.previous_frame_best_estimation)
        else:
            if self.slv_config["initialize"] == "random":
                x0 = self.initialize_random()
            elif self.slv_config["initialize"] == "zero":
                x0 = self.initialize_zeros()
            else:
                e = f"Initilization not implemented"
                logger.error(e)
                raise NotImplementedError(e)
        logger.info(f"Initial value: {x0}")

        result = scipy_autograd.minimize(
            self.objective_scipy,
            x0,
            method=self.opt_method,
            options={"gtol": 1e-8, "disp": True},
            # TODO support bounds
            # bounds=[(-300, 300), (-300, 300)]
        )
        return result

    @utils.profile(
        output_file="optimize.prof", sort_by="cumulative", lines_to_print=300, strip_dirs=True
    )
    def run_torch(self, events: np.ndarray) -> dict:
        self.events = events  # use this variable because currently not supported as passing arg
        if self.previous_frame_best_estimation is not None:
            x0 = np.copy(self.previous_frame_best_estimation)
        else:
            if self.slv_config["initialize"] == "random":
                x0 = self.initialize_random()
            elif self.slv_config["initialize"] == "zero":
                x0 = self.initialize_zeros()
            else:
                e = f"Initilization not implemented"
                logger.error(e)
                raise NotImplementedError(e)
        logger.info(f"Initial value: {x0}")

        poses = torch.from_numpy(x0.copy()).float().to(self._device)
        poses.requires_grad = True
        lr_step = iters = self.opt_config["n_iter"]
        lr = 0.05
        lr_decay = 0.1

        optimizer = torch.optim.__dict__[self.opt_method]([poses], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_step, lr_decay)
        min_loss = math.inf
        best_poses = x0
        best_it = 0
        # optimization process
        for it in range(iters):
            optimizer.zero_grad()
            loss = self.objective_scipy(poses)
            if loss < min_loss:
                best_poses = poses
                min_loss = loss.item()
                best_it = it
            try:
                loss.backward()
            except Exception as e:
                logger.error(e)
                break
            optimizer.step()
            scheduler.step()
        return {"param": best_poses.detach().cpu().numpy(), "loss": min_loss, "best_iter": best_it}

    def initialize_random(self):
        logger.info("random initialization")
        x0 = np.random.rand(self.motion_vector_size).astype(np.float64) * 0.01 - 0.005
        return x0

    def initialize_zeros(self):
        logger.info("zero initialization")
        x0 = np.zeros(self.motion_vector_size).astype(np.float64)
        return x0

    def undistort_image(self, image: np.ndarray):
        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(
            self.calib_param["K"],
            self.calib_param["D"],
            self.image_shape,
            1,
            self.image_shape,
        )
        undistorted_image = cv2.undistort(
            image,
            self.calib_param["K"],
            self.calib_param["D"],
            None,
            newcameramtx,
        )
        return undistorted_image

    def setup_single_training(self, *args, **kwargs):
        pass  # Empty, only for compatibility with nnbase

    def train(self, *args, **kwargs):
        pass
