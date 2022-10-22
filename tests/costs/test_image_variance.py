import numpy as np
import pytest
import torch

from src import utils
from src.costs import ImageVariance
from src.event_image_converter import EventImageConverter
from src.warp import Warp


# minimum is different
def test_calculate_store_history():
    size = (260, 346)
    events = utils.generate_events(1000, size[0], size[1], tmin=0.1, tmax=0.9)
    events_torch = torch.from_numpy(events)
    imager = EventImageConverter(size)

    cost = ImageVariance(direction="minimize", store_history=True)
    # Calculate numpy
    count = imager.create_image_from_events_numpy(events, "bilinear_vote", weight=1.0, sigma=0)
    _ = cost.calculate({"iwe": count, "omit_boundary": True})

    # Calculate torch
    count = imager.create_image_from_events_tensor(events_torch, "bilinear_vote", weight=1.0)
    _ = cost.calculate({"iwe": count, "omit_boundary": True})

    history = cost.get_history()
    assert len(history["loss"]) == 2
    np.testing.assert_allclose(history["loss"][0], history["loss"][1], rtol=1e-5, atol=1e-5)


def test_calculate_not_store_history():
    size = (260, 346)
    events = utils.generate_events(1000, size[0], size[1], tmin=0.1, tmax=0.9)
    events_torch = torch.from_numpy(events)
    imager = EventImageConverter(size)

    cost = ImageVariance(direction="minimize", store_history=False)
    # Calculate numpy
    count = imager.create_image_from_events_numpy(events, "bilinear_vote", weight=1.0, sigma=0)
    _ = cost.calculate({"iwe": count, "omit_boundary": True})

    # Calculate torch
    count = imager.create_image_from_events_tensor(events_torch, "bilinear_vote", weight=1.0)
    _ = cost.calculate({"iwe": count, "omit_boundary": True})

    history = cost.get_history()
    assert len(history["loss"]) == 0


@pytest.mark.parametrize(
    "direction,is_small",
    [["natural", True], ["minimize", False], ["maximize", True]],
)
def test_calculate_blur_is_small(direction, is_small):
    size = (10, 40)
    imager = EventImageConverter(size)
    cost = ImageVariance(direction=direction, store_history=False)

    events = np.array(
        [
            [5.0, 10.0],
            [8.0, 3.0],
            [2.0, 2.0],
        ]
    )
    var_blur = cost.calculate({"iwe": imager.create_iwe(events), "omit_boundary": False})
    events = np.array(
        [
            [5.0, 10.0],
            [5.0, 10.0],
            [2.0, 2.0],
        ]
    )
    var_sharp = cost.calculate({"iwe": imager.create_iwe(events), "omit_boundary": False})

    assert (var_blur < var_sharp) == is_small


def test_calculate_np_torch():
    size = (10, 20)
    imager = EventImageConverter(size)
    cost = ImageVariance(direction="natural", store_history=False)

    events = np.array(
        [
            [12.0, 10.0],
            [8.0, 3.0],
            [2.0, 2.0],
        ]
    ).astype(np.float64)
    events_torch = torch.from_numpy(events)
    var_blur_numpy = cost.calculate(
        {"iwe": imager.create_iwe(events, sigma=0), "omit_boundary": True}
    )
    var_blur_torch = cost.calculate(
        {"iwe": imager.create_iwe(events_torch, sigma=0), "omit_boundary": True}
    )

    np.testing.assert_allclose(var_blur_torch.item(), var_blur_numpy, rtol=1e-2)
