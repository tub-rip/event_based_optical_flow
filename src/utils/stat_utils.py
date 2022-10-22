import logging
import math

import numpy as np
import scipy
import scipy.fftpack
import torch
from torch import nn

logger = logging.getLogger(__name__)


class SobelTorch(nn.Module):
    """Sobel operator for pytorch, for divergence calculation.
        This is equivalent implementation of
        ```
        sobelx = cv2.Sobel(flow[0], cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(flow[1], cv2.CV_64F, 0, 1, ksize=3)
        dxy = (sobelx + sobely) / 8.0
        ```
    Args:
        ksize (int) ... Kernel size of the convolution operation.
        in_channels (int) ... In channles.
        cuda_available (bool) ... True if cuda is available.
    """

    def __init__(
        self, ksize: int = 3, in_channels: int = 2, cuda_available: bool = False, precision="32"
    ):
        super().__init__()
        self.cuda_available = cuda_available
        self.in_channels = in_channels
        self.filter_dx = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=ksize,
            stride=1,
            padding=1,
            bias=False,
        )
        self.filter_dy = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=ksize,
            stride=1,
            padding=1,
            bias=False,
        )
        # x in height direction
        if precision == "64":
            Gx = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).double()
            Gy = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).double()
        else:
            Gx = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
            Gy = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])

        if self.cuda_available:
            Gx = Gx.cuda()
            Gy = Gy.cuda()

        self.filter_dx.weight = nn.Parameter(Gx.unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.filter_dy.weight = nn.Parameter(Gy.unsqueeze(0).unsqueeze(0), requires_grad=False)

    def forward(self, img):
        """
        Args:
            img (torch.Tensor) ... [b x (2 or 1) x H x W]. The 2 ch is [h, w] direction.

        Returns:
            sobel (torch.Tensor) ... [b x (4 or 2) x (H - 2) x (W - 2)].
                4ch means Sobel_x on xdim, Sobel_y on ydim, Sobel_x on ydim, and Sobel_y on xdim.
                To make it divergence, run `(sobel[:, 0] + sobel[:, 1]) / 8.0`.
        """
        if self.in_channels == 2:
            dxx = self.filter_dx(img[..., [0], :, :])
            dyy = self.filter_dy(img[..., [1], :, :])
            dyx = self.filter_dx(img[..., [1], :, :])
            dxy = self.filter_dy(img[..., [0], :, :])
            return torch.cat([dxx, dyy, dyx, dxy], dim=1)
        elif self.in_channels == 1:
            dx = self.filter_dx(img[..., [0], :, :])
            dy = self.filter_dy(img[..., [0], :, :])
            return torch.cat([dx, dy], dim=1)
