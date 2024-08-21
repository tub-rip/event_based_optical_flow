import torch
from torch import nn


class build_resnet_block(nn.Module):
    """
    a resnet block which includes two general_conv2d
    """

    def __init__(self, channels, layers=2, do_batch_norm=False):
        super().__init__()
        self._channels = channels
        self._layers = layers

        self.res_block = nn.Sequential(
            *[
                general_conv2d(
                    in_channels=self._channels,
                    out_channels=self._channels,
                    strides=1,
                    do_batch_norm=do_batch_norm,
                )
                for i in range(self._layers)
            ]
        )

    def forward(self, input_res):
        inputs = input_res.clone()
        input_res = self.res_block(input_res)
        return input_res + inputs


class upsample_conv2d_and_predict_flow_or_depth(nn.Module):
    """
    an upsample convolution layer which includes a nearest interpolate and a general_conv2d
    """

    def __init__(
        self, in_channels, out_channels, ksize=3, do_batch_norm=False, type="flow", scale=256.0
    ):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._ksize = ksize
        self._do_batch_norm = do_batch_norm
        self._flow_or_depth = type
        self._scale = scale

        self.general_conv2d = general_conv2d(
            in_channels=self._in_channels,
            out_channels=self._out_channels,
            ksize=self._ksize,
            strides=1,
            do_batch_norm=self._do_batch_norm,
            padding=0,
        )

        self.pad = nn.ReflectionPad2d(
            padding=(
                int((self._ksize - 1) / 2),
                int((self._ksize - 1) / 2),
                int((self._ksize - 1) / 2),
                int((self._ksize - 1) / 2),
            )
        )

        if self._flow_or_depth == "flow":
            self.predict = general_conv2d(
                in_channels=self._out_channels,
                out_channels=2,
                ksize=1,
                strides=1,
                padding=0,
                activation="tanh",
            )
        elif self._flow_or_depth == "depth":
            self.predict = general_conv2d(
                in_channels=self._out_channels,
                out_channels=1,
                ksize=1,
                strides=1,
                padding=0,
                activation="sigmoid",
            )
        else:
            raise NotImplementedError("flow or depth?")

    def forward(self, conv):
        """
        Returns:
            feature
            pred (tensor) ... [N, ch, H, W]; 2 ch (flow) or 1 ch (depth).
        """
        shape = conv.shape
        conv = nn.functional.interpolate(
            # conv, size=[shape[2] * 2, shape[3] * 2], mode="nearest"
            conv,
            size=[shape[2] * 2, shape[3] * 2],
            mode="bilinear",
        )
        conv = self.pad(conv)
        conv = self.general_conv2d(conv)
        pred = self.predict(conv) * self._scale
        return torch.cat([conv, pred.clone()], dim=1), pred


def general_conv2d(
    in_channels, out_channels, ksize=3, strides=2, padding=1, do_batch_norm=False, activation="relu"
):
    """
    a general convolution layer which includes a conv2d, a relu and a batch_normalize
    """
    if activation == "relu":
        if do_batch_norm:
            conv2d = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=ksize,
                    stride=strides,
                    padding=padding,
                ),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.99),
            )
        else:
            conv2d = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=ksize,
                    stride=strides,
                    padding=padding,
                ),
                nn.ReLU(inplace=True),
            )
    elif activation == "tanh":
        if do_batch_norm:
            conv2d = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=ksize,
                    stride=strides,
                    padding=padding,
                ),
                nn.Tanh(),
                nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.99),
            )
        else:
            conv2d = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=ksize,
                    stride=strides,
                    padding=padding,
                ),
                nn.Tanh(),
            )
    elif activation == "sigmoid":
        if do_batch_norm:
            conv2d = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=ksize,
                    stride=strides,
                    padding=padding,
                ),
                nn.Sigmoid(),
                nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.99),
            )
        else:
            conv2d = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=ksize,
                    stride=strides,
                    padding=padding,
                ),
                nn.Sigmoid(),
            )
    return conv2d


import torch
import torch.nn as nn
import torch.nn.functional as f


class ConvLayer(nn.Module):
    """
    Convolutional layer.
    Default: bias, ReLU, no downsampling, no batch norm.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        activation="relu",
        norm=None,
        BN_momentum=0.1,
    ):
        super(ConvLayer, self).__init__()

        bias = False if norm == "BN" else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class TransposedConvLayer(nn.Module):
    """
    Transposed convolutional layer to increase spatial resolution (x2) in a decoder.
    Default: bias, ReLU, no downsampling, no batch norm.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        activation="relu",
        norm=None,
    ):
        super(TransposedConvLayer, self).__init__()

        bias = False if norm == "BN" else True
        self.transposed_conv2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=2,
            padding=padding,
            output_padding=1,
            bias=bias,
        )

        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.transposed_conv2d(x)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class UpsampleConvLayer(nn.Module):
    """
    Upsampling layer (bilinear interpolation + Conv2d) to increase spatial resolution (x2) in a decoder.
    Default: bias, ReLU, no downsampling, no batch norm.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        activation="relu",
        norm=None,
    ):
        super(UpsampleConvLayer, self).__init__()

        bias = False if norm == "BN" else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        x_upsampled = f.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.conv2d(x_upsampled)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class ResidualBlock(nn.Module):
    """
    Residual block as in "Deep residual learning for image recognition", He et al. 2016.
    Default: bias, ReLU, no downsampling, no batch norm, ConvLSTM.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        downsample=None,
        norm=None,
        BN_momentum=0.1,
    ):
        super(ResidualBlock, self).__init__()
        bias = False if norm == "BN" else True
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
        )
        self.norm = norm
        if norm == "BN":
            self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
            self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == "IN":
            self.bn1 = nn.InstanceNorm2d(out_channels)
            self.bn2 = nn.InstanceNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.norm in ["BN", "IN"]:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.norm in ["BN", "IN"]:
            out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


def skip_concat(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    padding = nn.ZeroPad2d((diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
    x1 = padding(x1)
    return torch.cat([x1, x2], dim=1)


def skip_sum(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    padding = nn.ZeroPad2d((diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
    x1 = padding(x1)
    return x1 + x2
