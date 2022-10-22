import numpy as np
import pytest
import torch
from PIL.Image import Image

from src import utils
from src.costs import HybridCost, ImageVariance
from src.event_image_converter import EventImageConverter
from src.warp import Warp


# minimum is different
def test_hybrid_cost_store_history():
    size = (20, 34)
    imager = EventImageConverter(size)
    cost_with_weight = {"image_variance": 1.0, "gradient_magnitude": 2.4}

    cost = HybridCost(direction="minimize", cost_with_weight=cost_with_weight, store_history=True)
    variance = ImageVariance(store_history=True)

    # Calculate numpy
    events = utils.generate_events(1000, size[0], size[1])
    count = imager.create_image_from_events_numpy(events, sigma=0)
    _ = cost.calculate({"iwe": count, "omit_boundary": True})
    _ = variance.calculate({"iwe": count, "omit_boundary": True})
    events = utils.generate_events(1000, size[0], size[1])
    count = imager.create_image_from_events_numpy(events, sigma=0)
    _ = cost.calculate({"iwe": count, "omit_boundary": True})
    _ = variance.calculate({"iwe": count, "omit_boundary": True})
    events = utils.generate_events(1000, size[0], size[1])
    count = imager.create_image_from_events_numpy(events, sigma=0)
    _ = cost.calculate({"iwe": count, "omit_boundary": True})
    _ = variance.calculate({"iwe": count, "omit_boundary": True})

    history = cost.get_history()
    keys = ["loss", "image_variance", "gradient_magnitude"]
    assert history.keys() == set(keys)
    for k in keys:
        assert len(history[k]) == 3
    np.testing.assert_allclose(
        history["image_variance"], variance.get_history()["loss"], rtol=1e-5, atol=1e-5
    )


def test_hybrid_cost_without_store_history():
    size = (20, 34)
    imager = EventImageConverter(size)
    cost_with_weight = {"image_variance": 1.0, "gradient_magnitude": 2.4}

    cost = HybridCost(direction="minimize", cost_with_weight=cost_with_weight, store_history=False)

    # Calculate numpy
    events = utils.generate_events(1000, size[0], size[1])
    count = imager.create_image_from_events_numpy(events, "bilinear_vote", sigma=0)
    _ = cost.calculate({"iwe": count, "omit_boundary": True})
    events = utils.generate_events(1000, size[0], size[1])
    count = imager.create_image_from_events_numpy(events, "bilinear_vote", sigma=0)
    _ = cost.calculate({"iwe": count, "omit_boundary": True})
    events = utils.generate_events(1000, size[0], size[1])
    count = imager.create_image_from_events_numpy(events, "bilinear_vote", sigma=0)
    _ = cost.calculate({"iwe": count, "omit_boundary": True})

    history = cost.get_history()
    keys = ["loss", "image_variance", "gradient_magnitude"]
    assert history.keys() == set(keys)
    for k in keys:
        assert len(history[k]) == 0
