import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import optuna
import torch
from torchvision import transforms

from .. import solver, types, utils, visualizer, warp
from .base import SingleThreadInMemoryStorage

logger = logging.getLogger(__name__)


class PatchContrastMaximization(solver.SolverBase):
    """Patch-based CMax, parent class.

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
        # Placeholders
        self.patch_shift = (0, 0)
        self.patch_image_size = (0, 0)
        self.n_patch = 0
        self.patches = {}
        super().__init__(
            image_shape,
            calibration_parameter,
            solver_config,
            optimizer_config,
            output_config,
            visualize_module,
        )
        self.filter_type = self.slv_config["patch"]["filter_type"]

    def set_patch_size_and_sliding_window(self):
        if isinstance(self.slv_config["patch"]["size"], int):
            self.patch_size = (self.slv_config["patch"]["size"], self.slv_config["patch"]["size"])
        elif isinstance(self.slv_config["patch"]["size"], list):
            self.patch_size = tuple(self.slv_config["patch"]["size"])
        else:
            e = "Unsupported type for patch."
            logger.error(e)
            raise TypeError(e)

        if isinstance(self.slv_config["patch"]["sliding_window"], int):
            self.sliding_window = (
                self.slv_config["patch"]["sliding_window"],
                self.slv_config["patch"]["sliding_window"],
            )
        elif isinstance(self.slv_config["patch"]["sliding_window"], list):
            self.sliding_window = tuple(self.slv_config["patch"]["sliding_window"])
        else:
            e = "Unsupported type for sliding_window."
            logger.error(e)
            raise TypeError(e)

    def prepare_patch(
        self, image_size: tuple, patch_size: tuple, sliding_window: tuple
    ) -> Tuple[Dict[int, types.FlowPatch], tuple]:
        """Get list of patches.

        Args:
            image_size (tuple): (H, W)
            patch_size (tuple): (H, W)
            sliding_window (tuple): (H, W)

        Returns:
            [type]: [description]
        """
        image_h, image_w = image_size
        patch_h, patch_w = patch_size
        slide_h, slide_w = sliding_window
        center_x = np.arange(0, image_h - patch_h + slide_h, slide_h) + patch_h / 2
        center_y = np.arange(0, image_w - patch_w + slide_w, slide_w) + patch_w / 2
        xx, yy = np.meshgrid(center_x, center_y)
        patch_shape = xx.T.shape
        xx = xx.T.reshape(-1)
        yy = yy.T.reshape(-1)
        patches = {
            i: types.FlowPatch(
                x=xx[i],
                y=yy[i],
                shape=patch_size,
                u=0.0,
                v=0.0,
            )
            for i in range(0, len(xx))
        }
        return patches, patch_shape

    # Get initial value
    def initialize_random(self):
        logger.info("random initialization")
        # scale = 10  # old
        x0 = np.random.rand(self.motion_vector_size, self.n_patch).astype(np.float64)  # [0, 1]
        xmin = self.opt_config["parameters"]["trans_x"]["min"]
        xmax = self.opt_config["parameters"]["trans_x"]["max"]
        ymin = self.opt_config["parameters"]["trans_y"]["min"]
        ymax = self.opt_config["parameters"]["trans_y"]["max"]
        x0[0] = x0[0] * (xmax - xmin) + xmin
        x0[1] = x0[1] * (ymax - ymin) + ymin
        # x0 *= scale  # old
        return x0

    def initialize_zeros(self):
        logger.info("zero initialization")
        x0 = np.zeros((self.motion_vector_size, self.n_patch)).astype(np.float64)
        return x0

    def initialize_guess_from_patch(self, events: np.ndarray, patch_index: int = 1):
        # sampling_field = np.arange(-300, 300, 50)
        sampling_field = np.arange(-150, 150, 30)
        best_guess = np.zeros(self.motion_vector_size)
        best_loss = np.inf
        for i in range(len(sampling_field)):
            for j in range(len(sampling_field)):
                guess = (
                    torch.from_numpy(np.array([sampling_field[i], sampling_field[j]]))
                    .double()
                    .requires_grad_()
                    .to(self._device)
                )
                self.events = (
                    torch.from_numpy(
                        utils.crop_event(
                            events,
                            self.patches[patch_index].x_min,
                            self.patches[patch_index].x_max,
                            self.patches[patch_index].y_min,
                            self.patches[patch_index].y_max,
                        )
                    )
                    .double()
                    .requires_grad_()
                    .to(self._device)
                )

                loss = self.objective_scipy_for_patch(guess, suppress_log=True)
                logger.info(f"Loss is {loss} for x: {sampling_field[i]} and y: {sampling_field[j]}")
                if loss < best_loss:
                    best_guess = guess
                    best_loss = loss
        logger.info(f"Initial value: {best_guess = }")
        # if isinstance(best_guess, torch.Tensor):
        #     best_guess = best_guess.cpu().detach().numpy()
        return best_guess

    def initialize_guess_from_whole_image(self, events: np.ndarray):
        # sampling_field = np.arange(-300, 300, 50)
        sampling_field = np.arange(-150, 150, 10)
        best_guess = np.zeros(self.motion_vector_size)
        best_loss = np.inf
        for i in range(len(sampling_field)):
            for j in range(len(sampling_field)):
                guess = (
                    torch.from_numpy(np.array([sampling_field[i], sampling_field[j]]))
                    .double()
                    .requires_grad_()
                    .to(self._device)
                )
                self.events = torch.from_numpy(events).double().requires_grad_().to(self._device)

                loss = self.objective_scipy_for_patch(guess, suppress_log=True)
                logger.info(f"Loss is {loss} for x: {sampling_field[i]} and y: {sampling_field[j]}")
                if loss < best_loss:
                    best_guess = guess
                    best_loss = loss
        logger.info(f"Initial value: {best_guess = }")
        # if isinstance(best_guess, torch.Tensor):
        #     best_guess = best_guess.cpu().detach().numpy()
        return best_guess

    def initialize_guess_from_optuna_sampling(self, events: np.ndarray, n_split=4):
        # Using Optuna sampler, get best guess.
        motion0 = np.zeros((self.motion_vector_size, self.n_patch))
        for i in range(self.n_patch):
            sampler = optuna.samplers.TPESampler()
            filtered_events = utils.crop_event(
                events,
                self.patches[i].x_min,
                self.patches[i].x_max,
                self.patches[i].y_min,
                self.patches[i].y_max,
            )
            if len(filtered_events) > 2:
                opt_result = optuna.create_study(
                    direction="minimize", sampler=sampler, storage=SingleThreadInMemoryStorage()
                )
                opt_result.optimize(
                    lambda trial: self.objective_initial(trial, filtered_events),
                    n_trials=self.opt_config["n_iter"],
                )
                motion0[:, i] = np.array(
                    [
                        opt_result.best_params["trans_x"],
                        opt_result.best_params["trans_y"],
                    ]
                )
            else:
                motion0[:, i] = np.array([0, 0])
        logger.info(f"Initial value: {motion0 = }")
        return motion0

    def objective_initial(self, trial, events: np.ndarray):
        # Parameters setting
        params = {k: self.sampling_initial(trial, k) for k in self.motion_model_keys}
        motion_array = np.array([params["trans_x"], params["trans_y"]])
        if self.normalize_t_in_batch:
            t_scale = np.max(events[:, 2]) - np.min(events[:, 2])
            motion_array *= t_scale
        loss = self.calculate_cost(
            events,
            motion_array,
            "2d-translation",
            np.zeros((2, 20, 20)),
            save_intermediate_result=False,
        )
        logger.info(f"{trial.number = } / {loss = }")
        return loss

    def sampling_initial(self, trial, key: str):
        return trial.suggest_uniform(
            key,
            self.opt_config["parameters"][key]["min"],
            self.opt_config["parameters"][key]["max"],
        )

    def objective_scipy_for_patch(self, motion: np.ndarray, suppress_log: bool = False):
        """
        Args:
            motion (np.ndarray): [2,]

        Returns:
            [type]: [description]
        """
        events = self.events.clone()
        if len(events) == 0:  # For smaller patch, it is possible to have no events
            logger.warning(f"No events in the patch.")
            return torch.sum(motion - motion)  # return 0
        if self.normalize_t_in_batch:
            t_scale = events[:, 2].max() - events[:, 2].min()
        else:
            t_scale = 1.0

        loss = self.calculate_cost(
            events,
            motion * t_scale,
            self.slv_config["motion_model"],
            motion.reshape((self.motion_vector_size, 1, 1)),
            save_intermediate_result=not suppress_log,
        )
        if not suppress_log:
            logger.info(f"{loss = }")
        return loss

    # Generic function to calculate cost.
    def calculate_cost(
        self,
        events: np.ndarray,
        warp,
        motion_model: str,
        coarse_flow: Optional[np.ndarray] = None,
        save_intermediate_result: bool = True,
    ):
        arg_cost = self.get_arg_for_cost(events, warp, motion_model, coarse_flow)
        loss = self.cost_func.calculate(arg_cost)
        if isinstance(loss, np.ndarray):
            if np.isnan(loss):
                logger.warning(f"Loss is nan")
                loss = 0.0
        return loss

    def get_arg_for_cost(self, events, warp, motion_model, coarse_flow=None):
        arg_cost = {"omit_boundary": True, "clip": True}

        if "events" in self.cost_func.required_keys:
            arg_cost.update({"events": events})

        if "orig_iwe" in self.cost_func.required_keys:
            orig_iwe = self.imager.create_iwe(
                events,
                self.iwe_config["method"],
                self.iwe_config["blur_sigma"],
            )
            arg_cost.update({"orig_iwe": orig_iwe})

        if (
            "iwe" in self.cost_func.required_keys
            or "backward_iwe" in self.cost_func.required_keys
            or "backward_warp" in self.cost_func.required_keys
        ):
            backward_events, backward_feat = self.warper.warp_event(
                events, warp, motion_model, direction="first"
            )
            backward_iwe = self.imager.create_iwe(
                backward_events,
                self.iwe_config["method"],
                self.iwe_config["blur_sigma"],
            )
            arg_cost.update(
                {
                    "iwe": backward_iwe,
                    "backward_iwe": backward_iwe,
                    "backward_warp": backward_events,
                }
            )

        if (
            "forward_iwe" in self.cost_func.required_keys
            or "forward_warp" in self.cost_func.required_keys
        ):
            forward_events, forward_feat = self.warper.warp_event(
                events, warp, motion_model, direction="last"
            )
            forward_iwe = self.imager.create_iwe(
                forward_events,
                self.iwe_config["method"],
                self.iwe_config["blur_sigma"],
            )
            arg_cost.update({"forward_iwe": forward_iwe, "forward_warp": forward_events})

        if "middle_iwe" in self.cost_func.required_keys:
            middle_events, middle_feat = self.warper.warp_event(
                events, warp, motion_model, direction="middle"
            )
            middle_iwe = self.imager.create_iwe(
                middle_events,
                self.iwe_config["method"],
                self.iwe_config["blur_sigma"],
            )
            arg_cost.update({"middle_iwe": middle_iwe})

        if "flow" in self.cost_func.required_keys:
            arg_cost.update({"flow": coarse_flow})

        return arg_cost

    # Function for both types
    def visualize_one_batch_warp(self, events: np.ndarray, warp: Optional[np.ndarray] = None):
        if self.visualizer is None:
            return
        if warp is not None:
            if self.normalize_t_in_batch:
                t_scale = np.max(events[:, 2]) - np.min(events[:, 2])
                warp *= t_scale

            flow = self.motion_to_dense_flow(warp)
            # optimization is based on 2d trans, so we need to flip the sign for dense-flow visualization.
            orig_events = np.copy(events)
            events, feat = self.warper.warp_event(events, flow, self.motion_model_for_dense_warp)
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
            warp *= t_scale
        flow = self.motion_to_dense_flow(warp)
        # optimization is based on 2d trans, so we need to flip the sign for dense-flow visualization.
        events, _ = self.warper.warp_event(
            events, flow, self.motion_model_for_dense_warp, direction="middle"
        )
        clipped_iwe = self.create_clipped_iwe_for_visualization(
            events, max_scale=self.iwe_visualize_max_scale
        )
        if self.is_time_aware:
            flow = self.get_original_flow_from_time_aware_flow_voxel(flow)
        self._pred_sequential(clipped_iwe, flow, with_grid=True, events_for_mask=events)

    def motion_to_dense_flow(self, motion_array):
        if isinstance(motion_array, np.ndarray):
            return self.interpolate_dense_flow_from_patch_numpy(motion_array)
        elif isinstance(motion_array, torch.Tensor):
            return self.interpolate_dense_flow_from_patch_tensor(motion_array)
        e = f"Unsupported type: {type(motion_array)}"
        raise TypeError(e)

    def interpolate_dense_flow_from_patch_numpy(self, motion_array: np.ndarray) -> np.ndarray:
        """
        Interpolate dense flow from patch.
        Args:
            flow_array (np.ndarray): [2 x h_patch x w_patch]

        Returns:
            np.ndarray: [2 x H x W]
        """
        # pad_h = int(self.patch_size[0] / 2 // self.sliding_window[0]) + 1
        pad_h = (
            int(self.patch_size[0] / 2 // self.sliding_window[0])
            + self.patch_shift[0] // self.sliding_window[0]
            + 1
        )
        pad_w = (
            int(self.patch_size[1] / 2 // self.sliding_window[1])
            + self.patch_shift[1] // self.sliding_window[1]
            + 1
        )
        # pad_w = int(self.patch_size[1] / 2 // self.sliding_window[1]) + 1
        flow_array = np.pad(
            -motion_array.reshape((self.motion_vector_size,) + self.patch_image_size),
            ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode="edge",
        )

        if self.filter_type == "bilinear":
            interp = cv2.INTER_LINEAR
        elif self.filter_type == "nearest":
            interp = cv2.INTER_NEAREST
        upscaled_u = cv2.resize(
            flow_array[0],
            None,
            None,
            fx=self.sliding_window[1],
            fy=self.sliding_window[0],
            interpolation=interp,
        )
        upscaled_v = cv2.resize(
            flow_array[1],
            None,
            None,
            fx=self.sliding_window[1],
            fy=self.sliding_window[0],
            interpolation=interp,
        )
        dense_flow = np.concatenate([upscaled_u[None, ...], upscaled_v[None, ...]], axis=0)
        cx, cy = dense_flow.shape[1] // 2, dense_flow.shape[2] // 2
        h1 = cx - self.image_shape[0] // 2
        w1 = cy - self.image_shape[1] // 2
        h2 = h1 + self.image_shape[0]
        w2 = w1 + self.image_shape[1]
        return dense_flow[..., h1:h2, w1:w2]

    def interpolate_dense_flow_from_patch_tensor(self, motion_array: torch.Tensor) -> torch.Tensor:
        """
        Args:
            motion_array (np.ndarray): 1-d array, [2 x h_patch x w_patch]

        Returns:
            np.ndarray: [2 x H x W]
        """
        pad_h = (
            int(self.patch_size[0] / 2 // self.sliding_window[0])
            + self.patch_shift[0] // self.sliding_window[0]
            + 1
        )
        pad_w = (
            int(self.patch_size[1] / 2 // self.sliding_window[1])
            + self.patch_shift[1] // self.sliding_window[1]
            + 1
        )
        flow_array = torch.nn.functional.pad(
            -motion_array.reshape(
                (
                    1,
                    self.motion_vector_size,
                )
                + self.patch_image_size
            ),
            # (pad_h, pad_h, pad_w, pad_w),
            (pad_w, pad_w, pad_h, pad_h),
            mode="replicate",
        )[0]
        if self.filter_type == "bilinear":
            interp = transforms.InterpolationMode.BILINEAR
        elif self.filter_type == "nearest":
            interp = transforms.InterpolationMode.NEAREST
        size = [
            flow_array.shape[1] * self.sliding_window[0],
            flow_array.shape[2] * self.sliding_window[1],
        ]
        dense_flow = transforms.functional.resize(flow_array, size, interpolation=interp)
        cx, cy = dense_flow.shape[1] // 2, dense_flow.shape[2] // 2
        h1 = cx - self.image_shape[0] // 2
        w1 = cy - self.image_shape[1] // 2
        h2 = h1 + self.image_shape[0]
        w2 = w1 + self.image_shape[1]
        return dense_flow[..., h1:h2, w1:w2]

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
        if self.visualizer is None:
            return
        pred_flow = self.motion_to_dense_flow(motion * timescale)  # [2, H, W]
        gt_flow = np.transpose(gt_flow, (2, 0, 1))  # [2, H, W]
        self.visualizer.visualize_optical_flow_pred_and_gt(
            pred_flow, gt_flow, file_prefix="flow_comparison"
        )
