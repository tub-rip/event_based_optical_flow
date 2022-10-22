import numpy as np
import torch

from src import utils


def test_crop_events():
    events = utils.generate_events(100, 30, 20)
    cropped = utils.crop_event(events, -10, 40, -20, 30)
    assert len(events) == len(cropped)

    cropped_torch = utils.crop_event(torch.from_numpy(events), -10, 40, -20, 30)
    assert len(events) == len(cropped) == len(cropped_torch)

    cropped = utils.crop_event(events, 5, 29, 2, 11)
    cropped_torch = utils.crop_event(torch.from_numpy(events), 5, 29, 2, 11)
    assert len(cropped) == len(cropped_torch)
