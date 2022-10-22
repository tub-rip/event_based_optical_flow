import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .. import solver, types, utils, visualizer

logger = logging.getLogger(__name__)


class TimeAwarePatchContrastMaximization(solver.MixedPatchContrastMaximization):
    """Time-aware patch-based CMax.

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
        assert self.is_time_aware

    def motion_to_dense_flow(self, motion_array: types.NUMPY_TORCH) -> types.NUMPY_TORCH:
        """Returns dense flow at quantized time voxel.
        TODO eventually I should be able to remove this entire class!

        Args:
            motion_array (types.NUMPY_TORCH): [2 x h_patch x w_patch] Flow array.

        Returns:
            types.NUMPY_TORCH: [time_bin x 2 x H x W]
        """
        if self.scale_later:
            scale = motion_array.max()
        else:
            scale = 1.0
        if isinstance(motion_array, np.ndarray):
            dense_flow_t0 = self.interpolate_dense_flow_from_patch_numpy(motion_array)
            return (
                utils.construct_dense_flow_voxel_numpy(
                    dense_flow_t0 / scale,
                    self.time_bin,
                    self.flow_interpolation,
                    t0_location=self.t0_flow_location,
                )
                * scale
            )
        elif isinstance(motion_array, torch.Tensor):
            dense_flow_t0 = self.interpolate_dense_flow_from_patch_tensor(motion_array)
            return (
                utils.construct_dense_flow_voxel_torch(
                    dense_flow_t0 / scale,
                    self.time_bin,
                    self.flow_interpolation,
                    t0_location=self.t0_flow_location,
                )
                * scale
            )
        e = f"Unsupported type: {type(motion_array)}"
        raise TypeError(e)
