import numpy as np
import pytest
import torch

from src import utils, warp


@pytest.mark.parametrize(
    "model,size",
    [["2d-translation", 2]],
)
def test_get_motion_vector_size(model, size):
    warper = warp.Warp((100, 200), normalize_t=True)
    assert size == warper.get_motion_vector_size(model)


def test_calculate_dt_normalize():
    image_size = (100, 200)
    warper = warp.Warp(image_size, normalize_t=True)

    events = utils.generate_events(300, image_size[0], image_size[1], tmin=1, tmax=2)
    dt = warper.calculate_dt(events, 1.0)
    np.testing.assert_allclose(dt.min(), 0.0, rtol=1e-2, atol=0.1)
    np.testing.assert_allclose(dt.max(), 1.0, rtol=1e-2, atol=0.1)

    events = utils.generate_events(300, image_size[0], image_size[1], tmin=0, tmax=0.5)
    dt = warper.calculate_dt(events, 0)
    np.testing.assert_allclose(dt.min(), 0.0, rtol=1e-2, atol=0.1)
    np.testing.assert_allclose(dt.max(), 1.0, rtol=1e-2, atol=0.1)

    events = utils.generate_events(300, image_size[0], image_size[1], tmin=-1, tmax=1)
    dt = warper.calculate_dt(events, 0)
    np.testing.assert_allclose(dt.min(), -0.5, rtol=1e-2, atol=0.1)
    np.testing.assert_allclose(dt.max(), 0.5, rtol=1e-2, atol=0.1)

    events = utils.generate_events(300, image_size[0], image_size[1], tmin=-1, tmax=1)
    dt = warper.calculate_dt(events, -1)
    np.testing.assert_allclose(dt.min(), 0.0, rtol=1e-2, atol=0.1)
    np.testing.assert_allclose(dt.max(), 1.0, rtol=1e-2, atol=0.1)


def test_calculate_dt_non_normalize():
    image_size = (10, 20)
    warper = warp.Warp(image_size, normalize_t=False)
    events = utils.generate_events(300, image_size[0], image_size[1], tmin=1, tmax=2)
    dt = warper.calculate_dt(events, 1.0)
    np.testing.assert_allclose(dt.max(), 1.0, rtol=1e-2, atol=0.1)
    np.testing.assert_allclose(dt.min(), 0.0, rtol=1e-2, atol=0.1)

    events = utils.generate_events(300, image_size[0], image_size[1], tmin=0, tmax=0.5)
    dt = warper.calculate_dt(events, 0)
    np.testing.assert_allclose(dt.max(), 0.5, rtol=1e-2, atol=0.1)
    np.testing.assert_allclose(dt.min(), 0.0, rtol=1e-2, atol=0.1)

    events = utils.generate_events(300, image_size[0], image_size[1], tmin=-1, tmax=1)
    dt = warper.calculate_dt(events, 0)
    np.testing.assert_allclose(dt.max(), 1.0, rtol=1e-2, atol=0.1)
    np.testing.assert_allclose(dt.min(), -1.0, rtol=1e-2, atol=0.1)

    events = utils.generate_events(300, image_size[0], image_size[1], tmin=-1, tmax=1)
    dt = warper.calculate_dt(events, -1)
    np.testing.assert_allclose(dt.max(), 2.0, rtol=1e-2, atol=0.1)
    np.testing.assert_allclose(dt.min(), 0.0, rtol=1e-2, atol=0.1)


def test_calculate_dt_numpy_batch():
    image_size = (10, 20)
    warper = warp.Warp(image_size, normalize_t=True)
    events = np.array(
        [
            utils.generate_events(300, image_size[0], image_size[1], tmin=1, tmax=i + 2)
            for i in range(4)
        ]
    )
    dt = warper.calculate_dt(events, 1.0)
    np.testing.assert_allclose(dt.max(axis=-1), 1.0, rtol=1e-2, atol=0.1)
    np.testing.assert_allclose(dt.min(axis=-1), 0.0, rtol=1e-2, atol=0.1)
    assert dt.shape == (4, 300)


def test_calculate_dt_torch_batch():
    image_size = (10, 20)
    warper = warp.Warp(image_size, normalize_t=True)
    events = np.array(
        [
            utils.generate_events(300, image_size[0], image_size[1], tmin=1, tmax=i + 2)
            for i in range(4)
        ]
    )
    dt = warper.calculate_dt(torch.from_numpy(events), 1.0).numpy()
    np.testing.assert_allclose(dt.max(axis=-1), 1.0, rtol=1e-2, atol=0.1)
    np.testing.assert_allclose(dt.min(axis=-1), 0.0, rtol=1e-2, atol=0.1)
    assert dt.shape == (4, 300)


def test_warp_event_dense_flow():
    image_size = (3, 4)
    warper = warp.Warp(image_size, normalize_t=True)

    events = np.array(
        [
            [1, 2, 0],
            [2, 3, 0.2],
            [0, 1, 0.6],
            [1, 0, 1.0],
        ]
    )
    flow = np.array(
        [
            [
                [1.0, -0.5, 2, 8],
                [-2, 0, 2.0, 0],
                [2, 1, -2, 0],
            ],
            [
                [-10, 1.0, 3, 2],
                [0, 2, -0.9, 0],
                [0, 10, -3, 0],
            ],
        ]
    )

    expected = np.array(
        [
            [1.0, 2.0, 0],
            [2.0, 3.0, 0.2],
            [0.3, 0.4, 0.6],
            [3, 0, 1.0],
        ]
    )
    # NUmpy
    warped, _ = warper.warp_event(events, flow, "dense-flow")
    np.testing.assert_allclose(warped, expected)

    # Torch
    warped_torch, _ = warper.warp_event(
        torch.from_numpy(events), torch.from_numpy(flow), "dense-flow"
    )
    assert torch.allclose(warped_torch, torch.from_numpy(expected))


def test_warp_event_dense_flow_batch():
    image_size = (3, 4)
    warper = warp.Warp(image_size, normalize_t=True)

    events = np.array(
        [
            [[1, 2, 0], [2, 3, 0.2]],
            [[0, 1, 0.6], [1, 0, 1.2]],
        ]
    )
    flow = np.array(
        [
            [
                [
                    [1.0, -0.5, 2, 8],
                    [-2, 0, 2.0, 0],
                    [2, 1, -2, 0],
                ],
                [
                    [-10, 1.0, 3, 2],
                    [0, 2, -0.9, 0],
                    [0, 10, -3, 0],
                ],
            ],
            [
                [
                    [1.0, -0.5, 2, 8],
                    [-2, 0, 2.0, 0],
                    [2, 1, -2, 0],
                ],
                [
                    [-10, 1.0, 3, 2],
                    [0, 2, -0.9, 0],
                    [0, 10, -3, 0],
                ],
            ],
        ]
    )

    expected = np.array(
        [
            [[1.0, 2.0, 0], [2, 3, 1.0]],
            [[0, 1, 0], [3, 0, 1.0]],
        ]
    )
    # NUmpy
    warped, _ = warper.warp_event(events, flow, "dense-flow")
    np.testing.assert_allclose(warped, expected)

    # Torch
    warped_torch, _ = warper.warp_event(
        torch.from_numpy(events), torch.from_numpy(flow), "dense-flow"
    )
    assert torch.allclose(warped_torch, torch.from_numpy(expected))


def test_warp_event_dense_flow_accuracy():
    image_size = (10, 20)
    warper = warp.Warp(image_size, normalize_t=True)
    events = np.array(
        [
            utils.generate_events(300, image_size[0], image_size[1], tmin=1, tmax=i + 2)
            for i in range(4)
        ]
    ).astype(np.float32)
    flow = np.array(
        [utils.generate_dense_optical_flow(image_size, max_val=10) for i in range(4)]
    ).astype(np.float32)
    warped, _ = warper.warp_event(events, flow, "dense-flow")
    warped_torch, _ = warper.warp_event(
        torch.from_numpy(events), torch.from_numpy(flow), "dense-flow"
    )
    np.testing.assert_allclose(warped, warped_torch.numpy(), rtol=1e-5, atol=1e-5)
