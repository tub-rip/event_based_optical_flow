import logging
from typing import Optional

import cv2
import numpy as np
import scipy
import torch
from torch.nn import functional

logger = logging.getLogger(__name__)

try:
    import torch_scatter
except ImportError:
    e = "Torch scatter needs to run some special interpolation."
    logger.warning(e)


# Generation
def generate_dense_optical_flow(image_size: tuple, max_val: int = 30) -> np.ndarray:
    """Generate random optical flow.

    Args:
        image_size (tuple) ... (H, W)

    Returns:
        flow (np.ndarray) ... [2 x H x W] array.
    """
    flow = np.random.uniform(-max_val, max_val, (2,) + image_size)
    return flow


# Time-aware flow: voxelization
def construct_dense_flow_voxel_numpy(
    dense_flow: np.ndarray,
    time_bin: int,
    scheme: str = "upwind",
    t0_location: str = "middle",
    clamp: Optional[int] = None,
) -> np.ndarray:
    """Construct dense flow voxel from given dense flow array at t0.

    Args:
        dense_flow (np.ndarray): [2 x H x W] flow, at t=0.
        time_bin (int) ... how many bins to create the voxel.
        scheme (str) ... 'upwind', 'max', 'same', 'zero'
        t0_location (str) ... 'first', 'middle'. Where the `dense_flow` is.
        clamp (Optional[int]) ... If given, output voxel is clamped.

    Returns:
        np.ndarray: [time_bin x 2 x H x W] flow.
            If t0_location is 'first', index 0 means flow at t=0, and index `time_bin - 1` means flow at t = 1.
            If t0_location is 'middle', index 0 means flow at t=-0.5, and index `time_bin - 1` means flow at t = 0.5.
    """
    if t0_location not in ["first", "middle"]:
        e = f"{t0_location =} not supported"
        logger.error(e)
        raise NotImplementedError(e)
    if len(dense_flow.shape) == 3:
        dense_flow = dense_flow[None]  # 1, 2, H, W
        is_single_data = True
    else:
        is_single_data = False
    n_batch = dense_flow.shape[0]
    dense_flow_voxel = np.zeros((n_batch, time_bin) + dense_flow.shape[1:])

    if scheme in ["upwind", "burgers"]:
        dt = 1.0 / time_bin
        if t0_location == "first":
            t0_index = 0
        elif t0_location == "middle":
            t0_index = time_bin // 2
        dense_flow_voxel[:, t0_index] = np.copy(dense_flow)
        if scheme == "burgers":
            for i in range(t0_index, 0, -1):
                dense_flow_voxel[:, i - 1] = inviscid_burger_flow_to_voxel_numpy(dense_flow_voxel[:, i], -dt, dx=1, dy=1)  # type: ignore
            for i in range(t0_index, time_bin - 1):
                dense_flow_voxel[:, i + 1] = inviscid_burger_flow_to_voxel_numpy(dense_flow_voxel[:, i], dt, dx=1, dy=1)  # type: ignore
        else:
            for i in range(t0_index, 0, -1):
                dense_flow_voxel[:, i - 1] = upwind_flow_to_voxel_numpy(dense_flow_voxel[:, i], -dt, dx=1, dy=1)  # type: ignore
            for i in range(t0_index, time_bin - 1):
                dense_flow_voxel[:, i + 1] = upwind_flow_to_voxel_numpy(dense_flow_voxel[:, i], dt, dx=1, dy=1)  # type: ignore
    else:
        if t0_location == "first":
            time_bin_array = (np.arange(0, time_bin)) / time_bin
        elif t0_location == "middle":
            time_bin_array = (np.arange(0, time_bin) - time_bin // 2) / time_bin
        for i in range(0, time_bin):
            dt = time_bin_array[i]
            dense_flow_voxel[:, i] = propagate_flow_to_voxel_numpy(dense_flow, dt, scheme)
    if clamp is not None:
        dense_flow_voxel = np.clip(dense_flow_voxel, -clamp, clamp)
    if is_single_data:
        return dense_flow_voxel[0]
    return dense_flow_voxel


def construct_dense_flow_voxel_torch(
    dense_flow: torch.Tensor,
    time_bin: int,
    scheme: str = "upwind",
    t0_location: str = "middle",
    clamp: Optional[int] = None,
) -> torch.Tensor:
    """Construct dense flow voxel from given dense flow array at t0.

    Args:
        dense_flow (torch.Tensor): [(batch x) 2 x H x W] flow, at t=0.
        time_bin (int) ... how many bins to create the voxel.
        scheme (str) ... 'upwind', 'max', 'same', 'zero'
        t0_location (str) ... 'first', 'middle'. Where the `dense_flow` is.
        clamp (Optional[int]) ... If given, output voxel is clamped.

    Returns:
        np.ndarray: [batch x time_bin x 2 x H x W] flow.
            If t0_location is 'first', index 0 means flow at t=0, and index `time_bin - 1` means flow at t = 1.
            If t0_location is 'middle', index 0 means flow at t=-0.5, and index `time_bin - 1` means flow at t = 0.5.
    """
    if t0_location not in ["first", "middle"]:
        e = f"{t0_location =} not supported"
        logger.error(e)
        raise NotImplementedError(e)
    if len(dense_flow.shape) == 3:
        dense_flow = dense_flow[None]  # 1, 2, H, W
        is_single_data = True
    else:
        is_single_data = False
    n_batch = dense_flow.shape[0]
    dense_flow_voxel = dense_flow.new_zeros((n_batch, time_bin) + dense_flow.shape[1:])

    if scheme in ["upwind", "burgers"]:
        dt = 1.0 / time_bin
        if t0_location == "first":
            t0_index = 0
        elif t0_location == "middle":
            t0_index = time_bin // 2
        dense_flow_voxel[:, t0_index] = torch.clone(dense_flow)
        if scheme == "burgers":
            for i in range(t0_index, -1, -1):
                dense_flow_voxel[:, i - 1] = inviscid_burger_flow_to_voxel_torch(torch.clone(dense_flow_voxel[:, i]), -dt, dx=1, dy=1)  # type: ignore
            for i in range(t0_index, time_bin - 1):
                dense_flow_voxel[:, i + 1] = inviscid_burger_flow_to_voxel_torch(dense_flow_voxel[:, i], dt, dx=1, dy=1)  # type: ignore
        else:
            for i in range(t0_index, 0, -1):
                dense_flow_voxel[:, i - 1] = upwind_flow_to_voxel_torch(dense_flow_voxel[:, i], -dt, dx=1, dy=1)  # type: ignore
            for i in range(t0_index, time_bin - 1):
                dense_flow_voxel[:, i + 1] = upwind_flow_to_voxel_torch(dense_flow_voxel[:, i], dt, dx=1, dy=1)  # type: ignore
    else:
        if t0_location == "first":
            time_bin_array = np.arange(0, time_bin) / time_bin
        elif t0_location == "middle":
            time_bin_array = (np.arange(0, time_bin) - time_bin // 2) / time_bin
        for i in range(0, time_bin):
            dt = time_bin_array[i]
            dense_flow_voxel[:, i] = propagate_flow_to_voxel_torch(dense_flow, dt, scheme)
    if clamp is not None:
        dense_flow_voxel = torch.clamp(dense_flow_voxel, -clamp, clamp)
    if is_single_data:
        return dense_flow_voxel[0]
    return dense_flow_voxel


def propagate_flow_to_voxel_numpy(
    flow_0: np.ndarray, dt: float, method: str = "nearest"
) -> np.ndarray:
    """Propagate flow into time voxel.

    Args:
        flow_0 (np.ndarray): 2 x H x W
        dt (float): [description] d
        method (str, optional): [description]. Defaults to "nearest".

    Raises:
        NotImplementedError: [description]

    Returns:
        np.ndarray: [description]
    """
    flow_0_flatten = flow_0.reshape(2, -1)
    flow_t_flatten = np.zeros_like(flow_0_flatten)

    _, h, w = flow_0.shape
    coord_x = np.arange(0, h)
    coord_y = np.arange(0, w)
    xx, yy = np.meshgrid(coord_x, coord_y, indexing="ij")
    flow_t_inds_x = (flow_0[0, xx, yy] * dt + xx).flatten()
    flow_t_inds_y = (flow_0[1, xx, yy] * dt + yy).flatten()

    if method in ["bilinear", "max"]:
        x1 = np.floor(flow_t_inds_x + 1e-8)
        y1 = np.floor(flow_t_inds_y + 1e-8)
        floor_to_x = flow_t_inds_x - x1
        floor_to_y = flow_t_inds_y - y1
        inds = np.concatenate(
            [
                y1 + x1 * w,
                y1 + (x1 + 1) * w,
                (y1 + 1) + x1 * w,
                (y1 + 1) + (x1 + 1) * w,
            ],
            axis=-1,
        )
        inds_mask = np.concatenate(
            [
                (0 <= y1) * (y1 < w) * (0 <= x1) * (x1 < h),
                (0 <= y1) * (y1 < w) * (0 <= x1 + 1) * (x1 + 1 < h),
                (0 <= y1 + 1) * (y1 + 1 < w) * (0 <= x1) * (x1 < h),
                (0 <= y1 + 1) * (y1 + 1 < w) * (0 <= x1 + 1) * (x1 + 1 < h),
            ],
            axis=-1,
        )
        if method == "bilinear":  # padding zeros to non-existing data. this is bilinear vote.
            w0_pos0 = (1 - floor_to_x) * (1 - floor_to_y) * flow_0_flatten[0]
            w0_pos1 = (1 - floor_to_x) * floor_to_y * flow_0_flatten[0]
            w0_pos2 = floor_to_x * (1 - floor_to_y) * flow_0_flatten[0]
            w0_pos3 = floor_to_x * floor_to_y * flow_0_flatten[0]
            w1_pos0 = (1 - floor_to_x) * (1 - floor_to_y) * flow_0_flatten[1]
            w1_pos1 = (1 - floor_to_x) * floor_to_y * flow_0_flatten[1]
            w1_pos2 = floor_to_x * (1 - floor_to_y) * flow_0_flatten[1]
            w1_pos3 = floor_to_x * floor_to_y * flow_0_flatten[1]

            vals0 = np.concatenate([w0_pos0, w0_pos1, w0_pos2, w0_pos3], axis=-1)
            vals1 = np.concatenate([w1_pos0, w1_pos1, w1_pos2, w1_pos3], axis=-1)
            inds = (inds * inds_mask).astype(np.int64)

            # This is just add. no interpolation.
            vals0 = vals0 * inds_mask
            vals1 = vals1 * inds_mask
            np.add.at(flow_t_flatten[0], inds, vals0)
            np.add.at(flow_t_flatten[1], inds, vals1)
        elif method == "max":
            w0_pos0 = np.copy(flow_0_flatten[0])
            w1_pos0 = np.copy(flow_0_flatten[1])
            vals0 = np.concatenate([w0_pos0, w0_pos0, w0_pos0, w0_pos0], axis=-1)
            vals1 = np.concatenate([w1_pos0, w1_pos0, w1_pos0, w1_pos0], axis=-1)
            inds = (inds * inds_mask).astype(np.int64)

            # This is just add. no interpolation.
            vals0 = vals0 * inds_mask
            vals1 = vals1 * inds_mask
            # np.max.at(flow_t_flatten[0], inds, vals0)
            # np.max.at(flow_t_flatten[1], inds, vals1)

            # TODO rewrite with numpy!
            # np.maximum.at(flow_t_flatten[0], inds, vals0)
            # np.maximum.at(flow_t_flatten[1], inds, vals1)
            vals0 = torch.from_numpy(vals0)
            vals1 = torch.from_numpy(vals1)
            inds = torch.from_numpy(inds)
            abs_val = torch.abs(vals0) + torch.abs(vals1)
            _, max_arg = torch_scatter.scatter_max(abs_val, inds, dim=0)

            max0 = torch.cat((vals0, torch.zeros(1)))[max_arg]
            max1 = torch.cat((vals1, torch.zeros(1)))[max_arg]
            flow_t_flatten[0, : len(max0)] = max0.numpy()
            flow_t_flatten[1, : len(max1)] = max1.numpy()
            # max0, _ = np.maximum.at(vals0, inds)
            # max1, _ = np.maximum.at(vals1, inds)
            # flow_t_flatten[0, :len(max0)] = max0
            # flow_t_flatten[1, :len(max1)] = max1

    elif method in ["nearest", "linear", "cubic"]:
        # use scipy.interpolate.griddata
        flow_t_inds_xy = np.vstack([flow_t_inds_x, flow_t_inds_y]).T  # [h x w, 2]
        dest_inds = np.vstack([xx.flatten(), yy.flatten()]).T  # [h x w, 2]
        flow_t_flatten[0] = scipy.interpolate.griddata(
            flow_t_inds_xy, flow_0_flatten[0], dest_inds, method=method
        )
        flow_t_flatten[1] = scipy.interpolate.griddata(
            flow_t_inds_xy, flow_0_flatten[1], dest_inds, method=method
        )
    elif method == "same":
        flow_t_flatten = np.copy(flow_0_flatten)
    else:
        e = f"{method = } is not supported."
        logger.error(e)
        raise NotImplementedError(e)
    return flow_t_flatten.reshape((2, h, w)).squeeze()


def propagate_flow_to_voxel_torch(
    flow_0: torch.Tensor, dt: float, method: str = "nearest"
) -> torch.Tensor:
    flow_0_flatten = flow_0.reshape(2, -1)
    flow_t_flatten = torch.zeros_like(flow_0_flatten)

    _, h, w = flow_0.shape
    coord_x = torch.arange(0, h, device=flow_0.device)
    coord_y = torch.arange(0, w, device=flow_0.device)
    xx, yy = torch.meshgrid(coord_x, coord_y)  # defaults to indexing='ij')
    flow_t_inds_x = (flow_0[0, xx, yy] * dt + xx).flatten()
    flow_t_inds_y = (flow_0[1, xx, yy] * dt + yy).flatten()

    if method in ["bilinear", "max"]:
        x1 = torch.floor(flow_t_inds_x + 1e-8)
        y1 = torch.floor(flow_t_inds_y + 1e-8)
        floor_to_x = flow_t_inds_x - x1
        floor_to_y = flow_t_inds_y - y1
        inds = torch.cat(
            [
                y1 + x1 * w,
                y1 + (x1 + 1) * w,
                (y1 + 1) + x1 * w,
                (y1 + 1) + (x1 + 1) * w,
            ],
            dim=-1,
        )
        inds_mask = torch.cat(
            [
                (0 <= y1) * (y1 < w) * (0 <= x1) * (x1 < h),
                (0 <= y1) * (y1 < w) * (0 <= x1 + 1) * (x1 + 1 < h),
                (0 <= y1 + 1) * (y1 + 1 < w) * (0 <= x1) * (x1 < h),
                (0 <= y1 + 1) * (y1 + 1 < w) * (0 <= x1 + 1) * (x1 + 1 < h),
            ],
            dim=-1,
        )
        if method == "bilinear":  # padding zeros to non-existing data.
            w0_pos0 = (1 - floor_to_x) * (1 - floor_to_y) * flow_0_flatten[0]
            w0_pos1 = (1 - floor_to_x) * floor_to_y * flow_0_flatten[0]
            w0_pos2 = floor_to_x * (1 - floor_to_y) * flow_0_flatten[0]
            w0_pos3 = floor_to_x * floor_to_y * flow_0_flatten[0]

            w1_pos0 = (1 - floor_to_x) * (1 - floor_to_y) * flow_0_flatten[1]
            w1_pos1 = (1 - floor_to_x) * floor_to_y * flow_0_flatten[1]
            w1_pos2 = floor_to_x * (1 - floor_to_y) * flow_0_flatten[1]
            w1_pos3 = floor_to_x * floor_to_y * flow_0_flatten[1]

            vals0 = torch.cat([w0_pos0, w0_pos1, w0_pos2, w0_pos3], dim=-1)
            vals1 = torch.cat([w1_pos0, w1_pos1, w1_pos2, w1_pos3], dim=-1)
            inds = (inds * inds_mask).long()

            vals0 = vals0 * inds_mask
            vals1 = vals1 * inds_mask

            flow_t_flatten[0].scatter_add_(0, inds, vals0)
            flow_t_flatten[1].scatter_add_(0, inds, vals1)
        elif method == "max":  # padding zeros to non-existing data.
            w0_pos0 = torch.clone(flow_0_flatten[0])
            w0_pos1 = torch.clone(flow_0_flatten[0])
            w0_pos2 = torch.clone(flow_0_flatten[0])
            w0_pos3 = torch.clone(flow_0_flatten[0])
            w1_pos0 = torch.clone(flow_0_flatten[1])
            w1_pos1 = torch.clone(flow_0_flatten[1])
            w1_pos2 = torch.clone(flow_0_flatten[1])
            w1_pos3 = torch.clone(flow_0_flatten[1])

            vals0 = torch.cat([w0_pos0, w0_pos1, w0_pos2, w0_pos3], dim=-1)
            vals1 = torch.cat([w1_pos0, w1_pos1, w1_pos2, w1_pos3], dim=-1)
            inds = (inds * inds_mask).long()

            vals0 = vals0 * inds_mask
            vals1 = vals1 * inds_mask
            abs_val = torch.abs(vals0) + torch.abs(vals1)
            _, max_arg = torch_scatter.scatter_max(abs_val, inds, dim=0)
            max0 = torch.cat((vals0, vals0.new_zeros(1)))[max_arg]
            max1 = torch.cat((vals1, vals1.new_zeros(1)))[max_arg]
            flow_t_flatten[0, : len(max0)] = max0
            flow_t_flatten[1, : len(max1)] = max1

            # max0, _ = torch_scatter.scatter_max(vals0, inds, dim=0)
            # max1, _ = torch_scatter.scatter_max(vals1, inds, dim=0)
            # flow_t_flatten[0, : len(max0)] = max0
            # flow_t_flatten[1, : len(max1)] = max1
    elif method in ["nearest", "linear", "cubic"]:
        # use scipy.interpolate.griddata
        flow_t_inds_xy = np.vstack([flow_t_inds_x, flow_t_inds_y]).T  # [h x w, 2]
        dest_inds = np.vstack([xx.flatten(), yy.flatten()]).T  # [h x w, 2]
        flow_t_flatten[0] = scipy.interpolate.griddata(
            flow_t_inds_xy, flow_0_flatten[0], dest_inds, method=method
        )
        flow_t_flatten[1] = scipy.interpolate.griddata(
            flow_t_inds_xy, flow_0_flatten[1], dest_inds, method=method
        )
    elif method == "same":
        flow_t_flatten = torch.clone(flow_0_flatten)
    else:
        e = f"{method = } is not supported."
        logger.error(e)
        raise NotImplementedError(e)
    return flow_t_flatten.reshape((2, h, w)).squeeze()


def upwind_flow_to_voxel_numpy(flow: np.ndarray, dt: float, dx: int = 1, dy: int = 1) -> np.ndarray:
    """1st-order Upwind scheme to propagate flow.
    For stability, `dt < flow.max()`. See https://en.wikipedia.org/wiki/Upwind_scheme

    Args:
        flow (np.ndarray): [(b, ) 2, W, H] flow at t0
        dt (float): When it is negative, it will return "backward" upwind result.
        dx (int, optional): [description]. Defaults to 1.
        dy (int, optional): [description]. Defaults to 1.

    Returns:
        np.ndarray: [(b, )2, W, H] flow at t=dt
    """
    if dt == 0:
        return flow
    if len(flow.shape) == 3:
        flow = flow[None]  # 1, 2, H, W
    # If dt is negative (backward upwind), we can swap the flow sign,
    dt_sign = np.sign(dt)
    dt = np.abs(dt)
    flow = flow * dt_sign

    u_dx = np.diff(flow[:, [0]], axis=-2)  # b, 1, H-1, W
    u_dy = np.diff(flow[:, [0]], axis=-1)  # b, 1, H, W-1
    v_dx = np.diff(flow[:, [1]], axis=-2)  # b, 1, H-1, W
    v_dy = np.diff(flow[:, [1]], axis=-1)  # b, 1, H, W-1

    pad_h_back = ((0, 0), (0, 0), (1, 0), (0, 0))
    pad_h_next = ((0, 0), (0, 0), (0, 1), (0, 0))
    pad_w_back = ((0, 0), (0, 0), (0, 0), (1, 0))
    pad_w_next = ((0, 0), (0, 0), (0, 0), (0, 1))

    u_dx_back = np.pad(u_dx, pad_h_back, mode="constant", constant_values=0) / dx  # i - (i - 1)
    u_dx_forw = np.pad(u_dx, pad_h_next, mode="constant", constant_values=0) / dx  # (i + 1) - i
    u_dy_back = np.pad(u_dy, pad_w_back, mode="constant", constant_values=0) / dx  # i - (i - 1)
    u_dy_forw = np.pad(u_dy, pad_w_next, mode="constant", constant_values=0) / dx  # (i + 1) - i
    v_dx_back = np.pad(v_dx, pad_h_back, mode="constant", constant_values=0) / dy  # i - (i - 1)
    v_dx_forw = np.pad(v_dx, pad_h_next, mode="constant", constant_values=0) / dy  # (i + 1) - i
    v_dy_back = np.pad(v_dy, pad_w_back, mode="constant", constant_values=0) / dy  # i - (i - 1)
    v_dy_forw = np.pad(v_dy, pad_w_next, mode="constant", constant_values=0) / dy  # (i + 1) - i

    # Fx(n+1) = ... Fx * dFx/dx + Fy * dFx / dy
    # Fy(n+1) = ... Fx * dFy/dx + Fy * dFy / dy
    # --> concat for compact form
    # [Fx(n+1), Fy(n+1)] = [Fx * dFx/dx, Fx * dFy/dx] + [Fy * dFx/dy, Fy * dFy/dy]
    # maximum / minimum is condition of which to use (i+1) - i or i - (i-1)
    flow_t = flow - dt * (
        np.maximum(flow[:, [0]], 0) * np.concatenate([u_dx_back, v_dx_back], axis=1)
        + np.minimum(flow[:, [0]], 0) * np.concatenate([u_dx_forw, v_dx_forw], axis=1)
        + np.maximum(flow[:, [1]], 0) * np.concatenate([u_dy_back, v_dy_back], axis=1)
        + np.minimum(flow[:, [1]], 0) * np.concatenate([u_dy_forw, v_dy_forw], axis=1)
    )
    return np.squeeze(flow_t) * dt_sign


def upwind_flow_to_voxel_torch(
    flow: torch.Tensor, dt: float, dx: int = 1, dy: int = 1
) -> torch.Tensor:
    """1st-order Upwind scheme to propagate flow.
    For stability, `dt < flow.max()`. See https://en.wikipedia.org/wiki/Upwind_scheme

    Args:
        flow (torch.Tensor): [(b, ) 2, W, H] flow at t0
        dt (float): When it is negative, it will return "backward" upwind result.
        dx (int, optional): [description]. Defaults to 1.
        dy (int, optional): [description]. Defaults to 1.

    Returns:
        torch.Tensor: [(b, )2, W, H] flow at t=dt
    """
    if dt == 0:
        return flow
    if len(flow.shape) == 3:
        flow = flow[None]  # 1, 2, H, W
    # If dt is negative (backward upwind), we can swap the flow sign,
    dt_sign = np.sign(dt)
    dt = np.abs(dt)
    flow = flow * dt_sign

    u_dx = torch.diff(flow[:, [0]], dim=-2)  # b, 1, H-1, W
    u_dy = torch.diff(flow[:, [0]], dim=-1)  # b, 1, H, W-1
    v_dx = torch.diff(flow[:, [1]], dim=-2)  # b, 1, H-1, W
    v_dy = torch.diff(flow[:, [1]], dim=-1)  # b, 1, H, W-1

    pad_h_back = (0, 0, 1, 0)
    pad_h_next = (0, 0, 0, 1)
    pad_w_back = (1, 0, 0, 0)
    pad_w_next = (0, 1, 0, 0)

    u_dx_back = functional.pad(u_dx, pad_h_back, mode="constant", value=0) / dx  # i - (i - 1)
    u_dx_forw = functional.pad(u_dx, pad_h_next, mode="constant", value=0) / dx  # (i + 1) - i
    u_dy_back = functional.pad(u_dy, pad_w_back, mode="constant", value=0) / dx  # i - (i - 1)
    u_dy_forw = functional.pad(u_dy, pad_w_next, mode="constant", value=0) / dx  # (i + 1) - i
    v_dx_back = functional.pad(v_dx, pad_h_back, mode="constant", value=0) / dy  # i - (i - 1)
    v_dx_forw = functional.pad(v_dx, pad_h_next, mode="constant", value=0) / dy  # (i + 1) - i
    v_dy_back = functional.pad(v_dy, pad_w_back, mode="constant", value=0) / dy  # i - (i - 1)
    v_dy_forw = functional.pad(v_dy, pad_w_next, mode="constant", value=0) / dy  # (i + 1) - i

    # see `upwind_flow_to_voxel_numpy` for comments
    flow_t = flow - dt * (
        torch.maximum(flow[:, [0]], torch.zeros_like(flow[:, [0]]))
        * torch.cat([u_dx_back, v_dx_back], dim=1)
        + torch.minimum(flow[:, [0]], torch.zeros_like(flow[:, [0]]))
        * torch.cat([u_dx_forw, v_dx_forw], dim=1)
        + torch.maximum(flow[:, [1]], torch.zeros_like(flow[:, [0]]))
        * torch.cat([u_dy_back, v_dy_back], dim=1)
        + torch.minimum(flow[:, [1]], torch.zeros_like(flow[:, [0]]))
        * torch.cat([u_dy_forw, v_dy_forw], dim=1)
    )
    return flow_t.squeeze() * dt_sign


def inviscid_burger_flow_to_voxel_numpy(
    flow: np.ndarray, dt: float, dx: int = 1, dy: int = 1
) -> np.ndarray:
    """Inviscid Burgers equation to propagate flow.
    For stability, `dt < flow.max()`. https://en.wikipedia.org/wiki/Burgers%27_equation

    Args:
        flow (np.ndarray): [description]
        dt (float): [description]
        dx (int, optional): [description]. Defaults to 1.
        dy (int, optional): [description]. Defaults to 1.

    Returns:
        np.ndarray: [description]
    """
    if dt == 0:
        return flow
    if len(flow.shape) == 3:
        flow = flow[None]  # 1, 2, H, W
    # If dt is negative (backward upwind), we can swap the flow sign,
    dt_sign = np.sign(dt)
    dt = np.abs(dt)
    flow = flow * dt_sign

    pad_h_back = ((0, 0), (0, 0), (1, 0), (0, 0))
    pad_h_next = ((0, 0), (0, 0), (0, 1), (0, 0))
    pad_w_back = ((0, 0), (0, 0), (0, 0), (1, 0))
    pad_w_next = ((0, 0), (0, 0), (0, 0), (0, 1))

    # Inviscid Burgers equation for dFx/dx and dFy/dy
    pow_flow = flow**2 * np.sign(flow)
    u_forw = np.pad(flow[:, [0]], pad_h_next, mode="edge")[..., 1:, :]  # b x 1 x H x W
    u_back = np.pad(flow[:, [0]], pad_h_back, mode="edge")[..., :-1, :]
    v_forw = np.pad(flow[:, [1]], pad_w_next, mode="edge")[..., 1:]  # b x 1 x H x W
    v_back = np.pad(flow[:, [1]], pad_w_back, mode="edge")[..., :-1]
    u_dx_forw = u_forw * u_forw
    u_dx_back = -u_back * u_back
    v_dy_forw = v_forw * v_forw
    v_dy_back = -v_back * v_back

    flow_back = np.concatenate([u_back, v_back], axis=1)  # b x 2 x H x W
    flow_forw = np.concatenate([u_forw, v_forw], axis=1)
    d_back = np.concatenate([u_dx_back, v_dy_back], axis=1)
    d_forw = np.concatenate([u_dx_forw, v_dy_forw], axis=1)
    burgers_factor = (
        pow_flow
        + np.maximum(np.sign(flow_back), 0) * d_back  # be +1 * d_back when flow_back > 0, else 0
        - np.minimum(np.sign(flow_forw), 0)
        * d_forw  # be - (-1 * d_forw) when flow_forw < 0, else 0
    ) / 2.0

    # For dFx/dy, dFy/dx, same as upwind scheme
    u_dy = np.diff(flow[:, [0]], axis=-1)  # b, 1, H, W-1
    v_dx = np.diff(flow[:, [1]], axis=-2)  # b, 1, H-1, W

    u_dy_back = np.pad(u_dy, pad_w_back, mode="constant", constant_values=0) / dx  # i - (i - 1)
    u_dy_forw = np.pad(u_dy, pad_w_next, mode="constant", constant_values=0) / dx  # (i + 1) - i
    v_dx_back = np.pad(v_dx, pad_h_back, mode="constant", constant_values=0) / dy  # i - (i - 1)
    v_dx_forw = np.pad(v_dx, pad_h_next, mode="constant", constant_values=0) / dy  # (i + 1) - i

    # see `upwind_flow_to_voxel_numpy` for comments
    flow_t = flow - dt * (
        np.maximum(flow[:, [0]], 0) * np.concatenate([np.zeros_like(v_dx_back), v_dx_back], axis=1)  # type: ignore
        + np.minimum(flow[:, [0]], 0) * np.concatenate([np.zeros_like(v_dx_back), v_dx_forw], axis=1)  # type: ignore
        + np.maximum(flow[:, [1]], 0) * np.concatenate([u_dy_back, np.zeros_like(u_dy_back)], axis=1)  # type: ignore
        + np.minimum(flow[:, [1]], 0) * np.concatenate([u_dy_forw, np.zeros_like(u_dy_back)], axis=1)  # type: ignore
        + burgers_factor
    )
    return np.squeeze(flow_t) * dt_sign


def inviscid_burger_flow_to_voxel_torch(
    flow: torch.Tensor, dt: float, dx: int = 1, dy: int = 1
) -> torch.Tensor:
    """Inviscid Burgers equation to propagate flow.
    For stability, `dt < flow.max()`. https://en.wikipedia.org/wiki/Burgers%27_equation

    Args:
        flow (torch.Tensor): [(b, ) 2, W, H] flow at t0
        dt (float): When it is negative, it will return "backward" upwind result.
        dx (int, optional): [description]. Defaults to 1.
        dy (int, optional): [description]. Defaults to 1.

    Returns:
        torch.Tensor: [(b, )2, W, H] flow at t=dt
    """
    if dt == 0:
        return flow
    if len(flow.shape) == 3:
        flow = flow[None]  # 1, 2, H, W
    # If dt is negative (backward upwind), we can swap the flow sign,
    dt_sign = np.sign(dt)
    dt = np.abs(dt)
    flow = flow * dt_sign

    pad_h_back = (0, 0, 1, 0)
    pad_h_next = (0, 0, 0, 1)
    pad_w_back = (1, 0, 0, 0)
    pad_w_next = (0, 1, 0, 0)

    # Inviscid Burgers equation for dFx/dx and dFy/dy
    pow_flow = flow**2 * torch.sign(flow)
    u_forw = functional.pad(flow[:, [0]], pad_h_next, mode="replicate")[..., 1:, :]  # b x 1 x H x W
    u_back = functional.pad(flow[:, [0]], pad_h_back, mode="replicate")[..., :-1, :]
    v_forw = functional.pad(flow[:, [1]], pad_w_next, mode="replicate")[..., 1:]  # b x 1 x H x W
    v_back = functional.pad(flow[:, [1]], pad_w_back, mode="replicate")[..., :-1]
    u_dx_forw = u_forw * u_forw
    u_dx_back = -u_back * u_back
    v_dy_forw = v_forw * v_forw
    v_dy_back = -v_back * v_back

    flow_back = torch.cat([u_back, v_back], dim=1)
    flow_forw = torch.cat([u_forw, v_forw], dim=1)
    d_back = torch.cat([u_dx_back, v_dy_back], dim=1)
    d_forw = torch.cat([u_dx_forw, v_dy_forw], dim=1)
    burgers_factor = (
        pow_flow
        + torch.maximum(torch.sign(flow_back), torch.zeros_like(flow_back)) * d_back
        - torch.minimum(torch.sign(flow_forw), torch.zeros_like(flow_forw)) * d_forw
    ) / 2.0

    # For dFx/dy, dFy/dx, same as upwind scheme
    u_dy = torch.diff(flow[:, [0]], dim=-1)  # b, 1, H, W-1
    v_dx = torch.diff(flow[:, [1]], dim=-2)  # b, 1, H-1, W

    u_dy_back = functional.pad(u_dy, pad_w_back, mode="constant", value=0) / dx  # i - (i - 1)
    u_dy_forw = functional.pad(u_dy, pad_w_next, mode="constant", value=0) / dx  # (i + 1) - i
    v_dx_back = functional.pad(v_dx, pad_h_back, mode="constant", value=0) / dy  # i - (i - 1)
    v_dx_forw = functional.pad(v_dx, pad_h_next, mode="constant", value=0) / dy  # (i + 1) - i

    # see `upwind_flow_to_voxel_numpy` for comments
    zero_shape = flow[:, [0]].shape
    flow_t = flow - dt * (
        torch.maximum(flow[:, [0]], flow.new_zeros(zero_shape))
        * torch.cat([flow.new_zeros(zero_shape), v_dx_back], dim=1)
        + torch.minimum(flow[:, [0]], flow.new_zeros(zero_shape))
        * torch.cat([flow.new_zeros(zero_shape), v_dx_forw], dim=1)
        + torch.maximum(flow[:, [1]], flow.new_zeros(zero_shape))
        * torch.cat([u_dy_back, flow.new_zeros(zero_shape)], dim=1)
        + torch.minimum(flow[:, [1]], flow.new_zeros(zero_shape))
        * torch.cat([u_dy_forw, flow.new_zeros(zero_shape)], dim=1)
        + burgers_factor
    )
    return flow_t.squeeze() * dt_sign


# Evaluation metrics
def calculate_flow_error_tensor(
    flow_gt: torch.Tensor,
    flow_pred: torch.Tensor,
    event_mask: Optional[torch.Tensor] = None,
    time_scale: Optional[torch.Tensor] = None,
) -> dict:
    """Calculate flow error.
    Args:
        flow_gt (torch.Tensor) ... [B x 2 x H x W]
        flow_pred (torch.Tensor) ... [B x 2 x H x W]
        event_mask (torch.Tensor) ... [B x 1 x W x H]. Optional.
        time_scale (torch.Tensor) ... [B x 1]. Optional. This will be multiplied.
            If you want to get error in 0.05 ms, time_scale should be
            `0.05 / actual_time_period`.

    Retuns:
        errors (dict) ... Key containrs 'AE', 'EPE', '1/2/3PE'. all float.

    """
    # Only compute error over points that are valid in the GT (not inf or 0).
    flow_mask = torch.logical_and(
        torch.logical_and(~torch.isinf(flow_gt[:, [0], ...]), ~torch.isinf(flow_gt[:, [1], ...])),
        torch.logical_and(torch.abs(flow_gt[:, [0], ...]) > 0, torch.abs(flow_gt[:, [1], ...]) > 0),
    )  # B, H, W
    if event_mask is None:
        total_mask = flow_mask
    else:
        total_mask = torch.logical_and(event_mask, flow_mask)
    gt_masked = flow_gt * total_mask  # b, 2, H, W
    pred_masked = flow_pred * total_mask
    n_points = torch.sum(total_mask, dim=(1, 2, 3)) + 1e-5  # B, 1

    errors = {}
    # Average endpoint error.
    if time_scale is not None:
        time_scale = time_scale.reshape(len(gt_masked), 1, 1, 1)
        gt_masked = gt_masked * time_scale
        pred_masked = pred_masked * time_scale
    endpoint_error = torch.linalg.norm(gt_masked - pred_masked, dim=1)
    errors["EPE"] = torch.mean(torch.sum(endpoint_error, dim=(1, 2)) / n_points)
    errors["1PE"] = torch.mean(torch.sum(endpoint_error > 1, dim=(1, 2)) / n_points)
    errors["2PE"] = torch.mean(torch.sum(endpoint_error > 2, dim=(1, 2)) / n_points)
    errors["3PE"] = torch.mean(torch.sum(endpoint_error > 3, dim=(1, 2)) / n_points)
    errors["5PE"] = torch.mean(torch.sum(endpoint_error > 5, dim=(1, 2)) / n_points)
    errors["10PE"] = torch.mean(torch.sum(endpoint_error > 10, dim=(1, 2)) / n_points)
    errors["20PE"] = torch.mean(torch.sum(endpoint_error > 20, dim=(1, 2)) / n_points)

    # Angular error
    u, v = pred_masked[:, 0, ...], pred_masked[:, 1, ...]
    u_gt, v_gt = gt_masked[:, 0, ...], gt_masked[:, 1, ...]
    errors["AE"] = torch.mean(
        torch.sum(
            torch.acos(
                (1.0 + u * u_gt + v * v_gt)
                / (torch.sqrt(1 + u * u + v * v) * torch.sqrt(1 + u_gt * u_gt + v_gt * v_gt))
            ),
            dim=(1, 2),
        )
        / n_points
    )
    return errors


def calculate_flow_error_numpy(
    flow_gt: np.ndarray, flow_pred: np.ndarray, event_mask: Optional[np.ndarray] = None
) -> dict:
    """Calculate flow error.
    Args:
        flow_gt (np.ndarray) ... [B x 2 x H x W]
        flow_pred (np.ndarray) ... [B x 2 x H x W]
        event_mask (np.ndarray) ... [B x 1 x W x H]. Optional.

    Retuns:
        errors (dict) ... Key containrs 'AE', 'EPE', '1/2/3PE'. all float.

    """
    assert len(flow_gt.shape) == len(flow_pred.shape) == 4
    # Only compute error over points that are valid in the GT (not inf or 0).
    flow_mask = np.logical_and(
        np.logical_and(~np.isinf(flow_gt[:, [0], ...]), ~np.isinf(flow_gt[:, [1], ...])),
        np.logical_and(np.abs(flow_gt[:, [0], ...]) > 0, np.abs(flow_gt[:, [1], ...]) > 0),
    )
    if event_mask is None:
        total_mask = flow_mask
    else:
        total_mask = np.logical_and(event_mask, flow_mask)
    gt_masked = flow_gt * total_mask  # b, 2, H, W
    pred_masked = flow_pred * total_mask
    n_points = np.sum(total_mask, axis=(1, 2, 3)) + 1e-5  # B, 1

    errors = {}

    # Average endpoint error.
    endpoint_error = np.linalg.norm(gt_masked - pred_masked, axis=1)
    errors["EPE"] = np.mean(np.sum(endpoint_error, axis=(1, 2)) / n_points)
    errors["1PE"] = np.mean(np.sum(endpoint_error > 1, axis=(1, 2)) / n_points)
    errors["2PE"] = np.mean(np.sum(endpoint_error > 2, axis=(1, 2)) / n_points)
    errors["3PE"] = np.mean(np.sum(endpoint_error > 3, axis=(1, 2)) / n_points)
    errors["5PE"] = np.mean(np.sum(endpoint_error > 5, axis=(1, 2)) / n_points)
    errors["10PE"] = np.mean(np.sum(endpoint_error > 10, axis=(1, 2)) / n_points)
    errors["20PE"] = np.mean(np.sum(endpoint_error > 20, axis=(1, 2)) / n_points)

    # Angular error
    u, v = pred_masked[:, 0, ...], pred_masked[:, 1, ...]
    u_gt, v_gt = gt_masked[:, 0, ...], gt_masked[:, 1, ...]
    errors["AE"] = np.mean(
        np.sum(
            np.arccos(
                (1.0 + u * u_gt + v * v_gt)
                / (np.sqrt(1 + u * u + v * v) * np.sqrt(1 + u_gt * u_gt + v_gt * v_gt))
            ),
            axis=(1, 2),
        )
        / n_points
    )
    return errors


# The below code is coming from Zhu. et al, EV-FlowNet
# Optical flow loader
def estimate_corresponding_gt_flow(x_flow_in, y_flow_in, gt_timestamps, start_time, end_time):
    """Code obtained from https://github.com/daniilidis-group/EV-FlowNet

    The ground truth flow maps are not time synchronized with the grayscale images. Therefore, we
    need to propagate the ground truth flow over the time between two images.
    This function assumes that the ground truth flow is in terms of pixel displacement, not velocity.
    Pseudo code for this process is as follows:
    x_orig = range(cols)
    y_orig = range(rows)
    x_prop = x_orig
    y_prop = y_orig
    Find all GT flows that fit in [image_timestamp, image_timestamp+image_dt].
    for all of these flows:
    x_prop = x_prop + gt_flow_x(x_prop, y_prop)
    y_prop = y_prop + gt_flow_y(x_prop, y_prop)
    The final flow, then, is x_prop - x-orig, y_prop - y_orig.
    Note that this is flow in terms of pixel displacement, with units of pixels, not pixel velocity.

    Args:
        x_flow_in, y_flow_in - list of numpy arrays, each array corresponds to per pixel flow at
            each timestamp.
        gt_timestamps - timestamp for each flow array.
        start_time, end_time - gt flow will be estimated between start_time and end time.
    Returns:
        (x_disp, y_disp) ... Each displacement of x and y.
    """
    # Each gt flow at timestamp gt_timestamps[gt_iter] represents the displacement between
    # gt_iter and gt_iter+1.
    gt_iter = np.searchsorted(gt_timestamps, start_time, side="right") - 1
    gt_dt = gt_timestamps[gt_iter + 1] - gt_timestamps[gt_iter]
    x_flow = np.squeeze(x_flow_in[gt_iter, ...])
    y_flow = np.squeeze(y_flow_in[gt_iter, ...])
    dt = end_time - start_time

    # No need to propagate if the desired dt is shorter than the time between gt timestamps.
    if gt_dt >= dt:
        return x_flow * dt / gt_dt, y_flow * dt / gt_dt

    x_indices, y_indices = np.meshgrid(np.arange(x_flow.shape[1]), np.arange(x_flow.shape[0]))
    x_indices = x_indices.astype(np.float32)
    y_indices = y_indices.astype(np.float32)

    orig_x_indices = np.copy(x_indices)
    orig_y_indices = np.copy(y_indices)

    # Mask keeps track of the points that leave the image, and zeros out the flow afterwards.
    x_mask = np.ones(x_indices.shape, dtype=bool)
    y_mask = np.ones(y_indices.shape, dtype=bool)

    scale_factor = (gt_timestamps[gt_iter + 1] - start_time) / gt_dt
    total_dt = gt_timestamps[gt_iter + 1] - start_time

    prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor=scale_factor)

    gt_iter += 1
    while gt_timestamps[gt_iter + 1] < end_time:
        x_flow = np.squeeze(x_flow_in[gt_iter, ...])
        y_flow = np.squeeze(y_flow_in[gt_iter, ...])

        prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask)
        total_dt += gt_timestamps[gt_iter + 1] - gt_timestamps[gt_iter]
        gt_iter += 1

    final_dt = end_time - gt_timestamps[gt_iter]
    total_dt += final_dt

    final_gt_dt = gt_timestamps[gt_iter + 1] - gt_timestamps[gt_iter]
    x_flow = np.squeeze(x_flow_in[gt_iter, ...])
    y_flow = np.squeeze(y_flow_in[gt_iter, ...])
    scale_factor = final_dt / final_gt_dt

    prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor)
    x_shift = x_indices - orig_x_indices
    y_shift = y_indices - orig_y_indices
    x_shift[~x_mask] = 0
    y_shift[~y_mask] = 0
    return x_shift, y_shift


def prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor=1.0):
    """Code obtained from https://github.com/daniilidis-group/EV-FlowNet

    Propagates x_indices and y_indices by their flow, as defined in x_flow, y_flow.
    x_mask and y_mask are zeroed out at each pixel where the indices leave the image.
    The optional scale_factor will scale the final displacement.

    In-place operation.
    """
    flow_x_interp = cv2.remap(x_flow, x_indices, y_indices, cv2.INTER_NEAREST)
    flow_y_interp = cv2.remap(y_flow, x_indices, y_indices, cv2.INTER_NEAREST)
    x_mask[flow_x_interp == 0] = False
    y_mask[flow_y_interp == 0] = False

    x_indices += flow_x_interp * scale_factor
    y_indices += flow_y_interp * scale_factor
