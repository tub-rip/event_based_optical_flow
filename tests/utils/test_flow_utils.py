import numpy as np
import pytest
import torch

from src import utils


def test_flow_error_numpy_torch():
    imsize = (100, 200)
    flow_pred_np = utils.generate_dense_optical_flow(imsize, max_val=20).astype(np.float32)
    flow_pred_np = flow_pred_np[None, ...]
    flow_gt_np = utils.generate_dense_optical_flow(imsize, max_val=20).astype(np.float32)
    flow_gt_np = flow_gt_np[None, ...]

    flow_pred_th = torch.from_numpy(flow_pred_np)
    flow_gt_th = torch.from_numpy(flow_gt_np)

    error_np = utils.calculate_flow_error_numpy(flow_gt_np, flow_pred_np)
    error_th = utils.calculate_flow_error_tensor(flow_gt_th, flow_pred_th)

    for k in error_np.keys():
        np.testing.assert_almost_equal(error_np[k], error_th[k].numpy(), decimal=5)


def test_flow_error_different_batch_size():
    # Test with bsize=1 vs. bsize=8
    imsize = (100, 200)
    flow_pred = utils.generate_dense_optical_flow(imsize, max_val=20).astype(np.float32)
    flow_gt = utils.generate_dense_optical_flow(imsize, max_val=20).astype(np.float32)
    flow_pred = np.tile(flow_pred, (8, 1, 1, 1))
    flow_gt = np.tile(flow_gt, (8, 1, 1, 1))
    mask = np.random.rand(8, 1, imsize[0], imsize[1]) > 0.1

    # Numpy
    error_batch = utils.calculate_flow_error_numpy(flow_gt, flow_pred, mask)
    error_one = utils.calculate_flow_error_numpy(flow_gt[[0]], flow_pred[[0]])

    for k in error_one.keys():
        np.testing.assert_almost_equal(error_one[k], error_batch[k], decimal=1)

    # Torch
    flow_gt = torch.from_numpy(flow_gt)
    flow_pred = torch.from_numpy(flow_pred)
    mask = torch.from_numpy(mask)
    error_batch = utils.calculate_flow_error_tensor(flow_gt, flow_pred, mask)
    error_one = utils.calculate_flow_error_tensor(flow_gt[[0]], flow_pred[[0]])

    for k in error_one.keys():
        np.testing.assert_almost_equal(error_one[k].numpy(), error_batch[k].numpy(), decimal=1)


@pytest.mark.parametrize("scheme", ["upwind", "burgers"])
def test_construct_dense_flow_voxel_numpy(scheme):
    imsize = (100, 200)
    n_bin = 60
    flow = utils.generate_dense_optical_flow(imsize, max_val=20).astype(np.float64)
    voxel_flow = utils.construct_dense_flow_voxel_numpy(flow, 1)
    np.testing.assert_almost_equal(voxel_flow[0], flow, decimal=8)

    voxel_flow = utils.construct_dense_flow_voxel_numpy(flow, n_bin, scheme, t0_location="middle")
    np.testing.assert_almost_equal(voxel_flow[n_bin // 2], flow, decimal=8)
    voxel_flow = utils.construct_dense_flow_voxel_numpy(flow, n_bin, scheme, t0_location="first")
    np.testing.assert_almost_equal(voxel_flow[0], flow, decimal=8)

    voxel_flow = utils.construct_dense_flow_voxel_numpy(flow, n_bin, scheme, t0_location="middle")
    np.testing.assert_almost_equal(voxel_flow[n_bin // 2], flow, decimal=8)
    voxel_flow = utils.construct_dense_flow_voxel_numpy(flow, n_bin, scheme, t0_location="first")
    np.testing.assert_almost_equal(voxel_flow[0], flow, decimal=8)


@pytest.mark.parametrize("scheme", ["upwind", "burgers"])
def test_construct_dense_flow_voxel_upwind_torch(scheme):
    imsize = (100, 200)
    n_bin = 60
    flow = utils.generate_dense_optical_flow(imsize, max_val=20).astype(np.float64)

    flow_torch = torch.from_numpy(flow).double()
    voxel_flow_torch = utils.construct_dense_flow_voxel_torch(
        flow_torch, n_bin, scheme, t0_location="middle"
    )
    np.testing.assert_almost_equal(
        voxel_flow_torch.numpy()[n_bin // 2], flow_torch.numpy(), decimal=8
    )
    voxel_flow_torch = utils.construct_dense_flow_voxel_torch(
        flow_torch, n_bin, scheme, t0_location="first"
    )
    np.testing.assert_almost_equal(voxel_flow_torch.numpy()[0], flow_torch.numpy(), decimal=8)


@pytest.mark.parametrize(
    "scheme,t0_location",
    [["upwind", "middle"], ["upwind", "first"], ["burgers", "middle"], ["burgers", "first"]],
)
def test_construct_dense_flow_voxel_numerical(scheme, t0_location):
    imsize = (100, 200)
    n_bin = 100
    flow = utils.generate_dense_optical_flow(imsize, max_val=10).astype(np.float64)

    flow_torch = torch.from_numpy(flow).double()

    voxel_flow = utils.construct_dense_flow_voxel_numpy(flow, n_bin, scheme, t0_location)
    voxel_flow_torch = utils.construct_dense_flow_voxel_torch(
        flow_torch, n_bin, scheme, t0_location
    )
    np.testing.assert_almost_equal(voxel_flow_torch.numpy(), voxel_flow, decimal=6)


def test_inviscid_burger_flow_to_voxel_numerical():
    imsize = (100, 200)
    dt = 0.01
    flow = utils.generate_dense_optical_flow(imsize, max_val=1).astype(np.float64)
    flow_torch = torch.from_numpy(flow).double()

    flow1 = utils.inviscid_burger_flow_to_voxel_numpy(flow, dt, 1, 1)
    flow1_torch = utils.inviscid_burger_flow_to_voxel_torch(flow_torch, dt, 1, 1)
    np.testing.assert_almost_equal(flow1_torch.numpy(), flow1, decimal=6)

    flow1 = utils.inviscid_burger_flow_to_voxel_numpy(flow, -dt, 1, 1)
    flow1_torch = utils.inviscid_burger_flow_to_voxel_torch(flow_torch, -dt, 1, 1)
    np.testing.assert_almost_equal(flow1_torch.numpy(), flow1, decimal=6)
