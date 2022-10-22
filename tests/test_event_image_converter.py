import numpy as np
import torch

from src import event_image_converter, utils


def test_create_iwe():
    image_shape = (100, 200)
    imager = event_image_converter.EventImageConverter(image_shape)
    events = np.array(
        [utils.generate_events(100, image_shape[0] - 1, image_shape[1] - 1) for _ in range(4)]
    )
    iwe = imager.create_iwe(events)
    assert iwe.shape == (4, 100, 200)


def test_bilinear_vote_integer():
    image_shape = (3, 4)
    imager = event_image_converter.EventImageConverter(image_shape)

    events = np.array(
        [
            [1.0, 2],
            [0, 1],
            [1, 0],
        ]
    )
    weights = np.array([1, 2, 0.8])
    img = imager.bilinear_vote_numpy(events, weight=weights)
    expected = np.array(
        [
            [0, 2, 0, 0],
            [0.8, 0, 1, 0],
            [0, 0, 0, 0],
        ]
    )
    # NUmpy
    np.testing.assert_array_equal(img, expected)

    # Torch
    img = imager.bilinear_vote_tensor(torch.from_numpy(events), weight=torch.from_numpy(weights))
    assert torch.allclose(img, torch.from_numpy(expected))


def test_bilinear_vote_float():
    image_shape = (3, 4)
    imager = event_image_converter.EventImageConverter(image_shape)
    events = np.array(
        [
            [1.2, 2],
            [0, 1.9],
            [0.5, 0.6],
        ]
    )
    weights = np.array([-1.0, 1.0, 1.5])
    img = imager.bilinear_vote_numpy(events, weight=weights)
    expected = np.array(
        [
            [0.3, 0.55, 0.9, 0],
            [0.3, 0.45, -0.8, 0],
            [0, 0, -0.2, 0],
        ]
    )
    # numpy
    np.testing.assert_allclose(img, expected)

    # torch
    img = imager.bilinear_vote_tensor(torch.from_numpy(events), weight=torch.from_numpy(weights))
    assert torch.allclose(img, torch.from_numpy(expected))


def test_bilinear_vote_batch():
    image_shape = (3, 4)
    imager = event_image_converter.EventImageConverter(image_shape)
    events = np.array(
        [
            [
                [1, 2],
                [0, 1],
                [1, 0],
            ],
            [
                [1.2, 2],
                [0, 1.9],
                [0.5, 0.6],
            ],
        ]
    )
    weights = np.array([[1.0, 2.0, 0.8], [-1.0, 1.0, 1.5]])
    img = imager.bilinear_vote_numpy(events, weight=weights)
    expected = np.array(
        [
            [
                [0, 2, 0, 0],
                [0.8, 0, 1, 0],
                [0, 0, 0, 0],
            ],
            [
                [0.3, 0.55, 0.9, 0],
                [0.3, 0.45, -0.8, 0],
                [0, 0, -0.2, 0],
            ],
        ]
    )
    # numpy
    np.testing.assert_allclose(img, expected)

    # torch
    img = imager.bilinear_vote_tensor(torch.from_numpy(events), weight=torch.from_numpy(weights))
    assert torch.allclose(img, torch.from_numpy(expected))


def test_bilinear_vote_accuracy():
    image_shape = (10, 20)
    imager = event_image_converter.EventImageConverter(image_shape)
    events = utils.generate_events(100, image_shape[0] - 1, image_shape[1] - 1)
    events += np.random.rand(100)[..., None]
    events = events.astype(np.float32)
    img = imager.bilinear_vote_numpy(events, weight=1.0)

    img_t = imager.bilinear_vote_tensor(torch.from_numpy(events), weight=1.0)
    np.testing.assert_allclose(img, img_t.numpy(), rtol=1e-5)
