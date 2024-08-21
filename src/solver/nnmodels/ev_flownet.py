import torch
from torch import nn

from ... import utils
from . import basic_layers

_BASE_CHANNELS = 64


class EVFlowNet(nn.Module):
    """EV-FlowNet definition
    Code is obtained from https://github.com/CyrilSterling/EVFlowNet-pytorch
    Thanks to the author @CyrilSterling (and @alexzhu for the original paper!)
    """
    def __init__(self, nn_param: dict = {}):
        super().__init__()
        self.no_batch_norm = nn_param["no_batch_norm"]

        # Parameters for event voxel input.
        self.n_channel = nn_param["n_bin"] if "n_bin" in nn_param.keys() else 4
        self.scale_bin_time = nn_param["scale_time"] if "scale_time" in nn_param.keys() else 128.0

        self.encoder1 = basic_layers.general_conv2d(
            in_channels=self.n_channel,
            out_channels=_BASE_CHANNELS,
            do_batch_norm=not self.no_batch_norm,
        )
        self.encoder2 = basic_layers.general_conv2d(
            in_channels=_BASE_CHANNELS,
            out_channels=2 * _BASE_CHANNELS,
            do_batch_norm=not self.no_batch_norm,
        )
        self.encoder3 = basic_layers.general_conv2d(
            in_channels=2 * _BASE_CHANNELS,
            out_channels=4 * _BASE_CHANNELS,
            do_batch_norm=not self.no_batch_norm,
        )
        self.encoder4 = basic_layers.general_conv2d(
            in_channels=4 * _BASE_CHANNELS,
            out_channels=8 * _BASE_CHANNELS,
            do_batch_norm=not self.no_batch_norm,
        )

        self.resnet_block = nn.Sequential(
            *[
                basic_layers.build_resnet_block(
                    8 * _BASE_CHANNELS, do_batch_norm=not self.no_batch_norm
                )
                for i in range(2)
            ]
        )

        self.decoder1 = basic_layers.upsample_conv2d_and_predict_flow_or_depth(
            in_channels=16 * _BASE_CHANNELS,
            out_channels=4 * _BASE_CHANNELS,
            do_batch_norm=not self.no_batch_norm,
            type="flow",
            scale=self.scale_bin_time,
        )

        self.decoder2 = basic_layers.upsample_conv2d_and_predict_flow_or_depth(
            in_channels=8 * _BASE_CHANNELS + 2,
            out_channels=2 * _BASE_CHANNELS,
            do_batch_norm=not self.no_batch_norm,
            type="flow",
            scale=self.scale_bin_time,
        )

        self.decoder3 = basic_layers.upsample_conv2d_and_predict_flow_or_depth(
            in_channels=4 * _BASE_CHANNELS + 2,
            out_channels=_BASE_CHANNELS,
            do_batch_norm=not self.no_batch_norm,
            type="flow",
            scale=self.scale_bin_time,
        )

        self.decoder4 = basic_layers.upsample_conv2d_and_predict_flow_or_depth(
            in_channels=2 * _BASE_CHANNELS + 2,
            out_channels=int(_BASE_CHANNELS / 2),
            do_batch_norm=not self.no_batch_norm,
            type="flow",
            scale=self.scale_bin_time,
        )

    @utils.profile(
        output_file="dnn_forward.prof", sort_by="cumulative", lines_to_print=300, strip_dirs=True
    )
    def forward(self, inputs: torch.Tensor) -> dict:
        """
        Args:
            inputs (torch.Tensor) ... [n_batch, n_bin, height, width]

        Returns
            flow_dict (dict) ... "flow3": [n_batch, 1, height, width]
                "flow0": [n_batch, 1, height // 2**3, width // 2**3]
        """
        # encoder
        skip_connections = {}
        inputs = self.encoder1(inputs)
        skip_connections["skip0"] = inputs.clone()
        inputs = self.encoder2(inputs)
        skip_connections["skip1"] = inputs.clone()
        inputs = self.encoder3(inputs)
        skip_connections["skip2"] = inputs.clone()
        inputs = self.encoder4(inputs)
        skip_connections["skip3"] = inputs.clone()

        # transition
        inputs = self.resnet_block(inputs)

        # decoder
        flow_dict = {}
        inputs = torch.cat([inputs, skip_connections["skip3"]], dim=1)
        inputs, flow = self.decoder1(inputs)
        flow_dict["flow0"] = flow.clone()

        inputs = torch.cat([inputs, skip_connections["skip2"]], dim=1)
        inputs, flow = self.decoder2(inputs)
        flow_dict["flow1"] = flow.clone()

        inputs = torch.cat([inputs, skip_connections["skip1"]], dim=1)
        inputs, flow = self.decoder3(inputs)
        flow_dict["flow2"] = flow.clone()

        inputs = torch.cat([inputs, skip_connections["skip0"]], dim=1)
        inputs, flow = self.decoder4(inputs)
        flow_dict["flow3"] = flow.clone()

        return flow_dict
