import torch
import torch.nn as nn
import torch.nn.functional as F
from .EventSurrealLayers import Encoder, Decoder, ConvLSTM
from eseg.utils.functions import eventstovoxel
from typing import List

__all__ = ["EConvlstm"]


class EConvlstm(nn.Module):
    """Event-based ConvLSTM depth estimation network.

    Pipeline:
        events -> voxelization -> MobileNetV2 encoder (+ optional skip ConvLSTMs)
        -> bottleneck ConvLSTM(s) -> UNet-style decoder -> sigmoid depth map.

    Args:
        model_type: String flag controlling presence of LSTM layers (contains "LSTM")
        width/height: Sensor spatial resolution
        skip_lstm: If True, apply ConvLSTM to skip feature maps
    """

    def __init__(
        self,
        model_type: str = "CONVLSTM",
        width: int = 346,
        height: int = 260,
        skip_lstm: bool = True,
    ):
        super().__init__()
        self.bins = 5
        self.width = width
        self.height = height
        self.model_type = model_type
        self.skip_lstm = skip_lstm
        self.method = "add"  # skip fusion strategy ("add" or "concatenate")

        self.encoder = Encoder(2 * self.bins)
        self.encoder_channels = [32, 24, 32, 64, 1280]
        self.mheight = 9
        self.mwidth = 11
        self.voxel_bn = torch.nn.GroupNorm(num_groups=1, num_channels=2 * self.bins)

        if "LSTM" in self.model_type:
            if self.skip_lstm:
                self.skip_convlstms = nn.ModuleList(
                    [
                        ConvLSTM(input_dim=ch, hidden_dims=[ch], kernel_size=3, num_layers=1)
                        for ch in self.encoder_channels[:-1]
                    ]
                )
            self.convlstm = ConvLSTM(
                input_dim=self.encoder_channels[4],
                hidden_dims=[128, 128],
                kernel_size=3,
                num_layers=2,
            )
            self.encoder_channels = self.encoder_channels[:-1] + [128]
        else:
            self.estimated_depth = None

        self.decoder = Decoder(self.encoder_channels, self.method)
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.encoder_channels[0], 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),  # Output depth in [0,1]
        )

    def reset_states(self):
        if "LSTM" in self.model_type:
            self.convlstm.reset_hidden()
            if self.skip_lstm:
                for skip_lstm in self.skip_convlstms:
                    skip_lstm.reset_hidden()
        else:
            self.estimated_depth = None

    def detach_states(self):
        if "LSTM" in self.model_type:
            self.convlstm.detach_hidden()
            if self.skip_lstm:
                for skip_lstm in self.skip_convlstms:
                    skip_lstm.detach_hidden()
        elif self.estimated_depth is not None:
            self.estimated_depth = self.estimated_depth.detach()

    def forward(self, event_sequence: List[torch.Tensor], training: bool = False, hotpixel: bool = False):
        """Forward pass over a list of event tensors.

        Args:
            event_sequence: list of length T, each [B, N, 4] raw events or [B, C, H, W] voxel
            training: Whether to apply training-time augmentations
            hotpixel: Inject synthetic hot pixels during voxelization

        Returns:
            outputs: [B, T, H, W]
            encodings: Bottleneck ConvLSTM feature sequence [B, T, C, h, w]
            seq_events: List of per-timestep voxel tensors used by the encoder
        """
        B = event_sequence[0].shape[0]
        T = len(event_sequence)
        seq_events: List[torch.Tensor] = []

        # Preprocess: normalize timestamps & voxelize if raw events provided
        with torch.no_grad():
            if event_sequence[0].shape[-1] == 4:  # raw events path
                for events in event_sequence:
                    min_t = events[:, :, 0].min(dim=1, keepdim=True)[0]
                    max_t = events[:, :, 0].max(dim=1, keepdim=True)[0]
                    denom = (max_t - min_t).clamp(min=1e-8)
                    events[:, :, 0] = (events[:, :, 0] - min_t) / denom
                    events[:, :, 1] = events[:, :, 1].clamp(0, self.width - 1)
                    events[:, :, 2] = events[:, :, 2].clamp(0, self.height - 1)
                    hist_events = eventstovoxel(
                        events,
                        self.height,
                        self.width,
                        bins=self.bins,
                        training=training,
                        hotpixel=hotpixel,
                    ).float()
                    seq_events.append(hist_events)
            else:  # already voxelized
                seq_events = event_sequence

        all_events = torch.stack(seq_events, dim=0)  # [T, B, C, H, W]
        all_events = all_events.view(T * B, *all_events.shape[2:])  # [T*B, C, H, W]
        all_events = self.voxel_bn(all_events)
        enc_top, all_feats = self.encoder(all_events)

        # Reshape encoder outputs back to time-major
        enc_top = enc_top.view(T, B, *enc_top.shape[1:])
        all_feats = [feat.view(T, B, *feat.shape[1:]) for feat in all_feats]

        # Prepare bottleneck sequence for ConvLSTM
        lstm_inputs = []
        for t in range(T):
            interpolated = F.interpolate(
                enc_top[t], size=(self.mheight, self.mwidth), mode="bilinear", align_corners=False
            )
            lstm_inputs.append(interpolated)
        lstm_inputs = torch.stack(lstm_inputs, dim=1)  # [B, T, C, mH, mW]

        skip_outputs: List[torch.Tensor] = []
        if self.skip_lstm:
            for i, skip_lstm in enumerate(self.skip_convlstms):
                skip_feats = all_feats[i].permute(1, 0, 2, 3, 4)  # [B, T, C, H, W]
                skip_out = skip_lstm(skip_feats)
                skip_outputs.append(skip_out)

        encodings = self.convlstm(lstm_inputs)  # [B, T, C, mH, mW]

        # Decoder expects flattened time*batch dimension
        flatten_encodings = encodings.view(B * T, *encodings.shape[2:])
        if skip_outputs:
            flatten_skip = [skip.view(B * T, *skip.shape[2:]) for skip in skip_outputs]
        else:
            flatten_skip = [
                feat.permute(1, 0, 2, 3, 4).contiguous().view(B * T, *feat.shape[2:])
                for feat in all_feats[:-1]
            ]
        x = self.decoder(flatten_encodings, flatten_skip)
        outputs = self.final_conv(x).view(B, T, 1, *x.shape[-2:])
        return outputs.squeeze(2), encodings.detach(), seq_events
