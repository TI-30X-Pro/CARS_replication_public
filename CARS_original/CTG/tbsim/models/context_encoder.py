from typing import List

import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor

from tbsim.models import base_models
from tbsim.models.base_models import ConvBlock, Up


class ContextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.map_encoder = MapEncoder(
            model_arch=config["MapEncoder"]["model_arch"],
            input_image_shape=config["MapEncoder"]["input_image_shape"],
            global_feature_dim=config["MapEncoder"]["global_feature_dim"],
            grid_feature_dim=config["MapEncoder"]["grid_feature_dim"],
        )

        self.center_hist = AgentHistoryEncoder(
            num_steps=config["AgentHistoryEncoder"]["num_steps"],
            out_dim=config["AgentHistoryEncoder"]["out_dim"],
            norm_info=config["norm_info_center"],
        )

        self.center_state = MLP(
            in_dim=config["StateEncoder"]["in_dim"],
            out_dim=config["StateEncoder"]["out_dim"],
            hidden_dims=config["StateEncoder"]["hidden_dims"],
        )

        self.neighbor_hist = NeighborHistoryEncoder(
            num_steps=config["NeighborHistoryEncoder"]["num_steps"],
            out_dim=config["NeighborHistoryEncoder"]["out_dim"],
            norm_info=config["norm_info_neighbor"],
        )

        combined_input = (
            config["MapEncoder"]["global_feature_dim"]
            + config["AgentHistoryEncoder"]["out_dim"]
            + config["StateEncoder"]["out_dim"]
            + config["NeighborHistoryEncoder"]["out_dim"]
        )

        self.context_fusion_mlp = MLP(
            in_dim=combined_input,
            out_dim=config["ContextFusion"]["out_dim"],
            hidden_dims=config["ContextFusion"]["hidden_dims"],
        )

    def forward(self, batch):
        maps = batch["maps"].contiguous()
        global_feat, grid_feat = self.map_encoder(maps)
        center_hist = self.center_hist(
            batch["center_hist_positions"],
            batch["center_hist_speeds"],
            batch["center_hist_yaws"],
            batch["center_hist_acc_lons"],
            batch["center_hist_yaw_rates"],
            batch["extent"],
            batch["center_hist_availabilities"],
        )
        center_state = self.center_state(
            torch.cat(
                [
                    batch["center_curr_positions"],
                    batch["center_curr_speeds"].unsqueeze(-1),
                    batch["center_curr_yaws"].unsqueeze(-1),
                ],
                -1,
            )
        )

        neigh_hist = self.neighbor_hist(
            batch["neigh_hist_positions"],
            batch["neigh_hist_speeds"],
            batch["neigh_hist_yaws"],
            batch["neigh_hist_acc_lons"],
            batch["neigh_hist_yaw_rates"],
            batch["neigh_extent"][..., :2],
            batch["neigh_hist_availabilities"],
        )
        cond_feat = self.context_fusion_mlp(
            torch.cat([global_feat, center_hist, center_state, neigh_hist], -1)
        )
        return cond_feat, grid_feat


class AgentHistoryEncoder(nn.Module):
    def __init__(self, num_steps, out_dim, norm_info):
        super().__init__()
        self.mean = torch.tensor(norm_info[0])
        self.std = torch.tensor(norm_info[1])
        self.state_dim = 10
        input_dim = num_steps * self.state_dim
        layer_dims = (input_dim, input_dim, out_dim, out_dim)
        self.traj_mlp = base_models.MLP(input_dim, out_dim, layer_dims)

    def forward(self, pos, speed, yaw, acc, yaw_rate, extent, avail):
        B, T, _ = pos.size()
        device = pos.device
        mean = self.mean.to(device)
        std = self.std.to(device)

        pos = torch.where(avail.unsqueeze(-1), pos, torch.zeros_like(pos))
        speed = torch.where(avail, speed, torch.zeros_like(speed))
        yaw = torch.where(avail, yaw, torch.zeros_like(yaw))
        acc = torch.where(avail, acc, torch.zeros_like(acc))
        yaw_rate = torch.where(avail, yaw_rate, torch.zeros_like(yaw_rate))

        heading_vec = torch.cat(
            [torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)],
            -1,
        )

        pos = (pos - mean[:2]) / std[:2]
        speed = (speed.unsqueeze(-1) - mean[2]) / std[2]
        acc = (acc.unsqueeze(-1) - mean[4]) / std[4]
        yaw_rate = (yaw_rate.unsqueeze(-1) - mean[5]) / std[5]

        extent_bt = extent[..., :2].unsqueeze(1).expand(-1, T, -1)
        extent_bt = (extent_bt - mean[6:8]) / std[6:8]

        avail_f = avail.unsqueeze(-1).to(pos.dtype)

        hist_in = torch.cat(
            [pos, heading_vec, speed, acc, yaw_rate, extent_bt, avail_f],
            -1,
        )
        hist_in = torch.where(avail_f.bool(), hist_in, torch.zeros_like(hist_in))

        hist_in = torch.nan_to_num(hist_in, nan=0.0, posinf=0.0, neginf=0.0)

        hist_in = hist_in.reshape(B, -1)
        output = self.traj_mlp(hist_in)
        return output


class NeighborHistoryEncoder(nn.Module):
    def __init__(self, num_steps, out_dim=128, norm_info=None):
        super().__init__()
        self.neigh_encode = AgentHistoryEncoder(num_steps, out_dim, norm_info)

    def forward(self, pos, speed, yaw, acc, yaw_rate, extent, avail):
        B, N, T, _ = pos.size()
        neigh_hist_enc = self.neigh_encode(
            pos.reshape(B * N, T, -1),
            speed.reshape(B * N, T),
            yaw.reshape(B * N, T),
            acc.reshape(B * N, T),
            yaw_rate.reshape(B * N, T),
            extent.reshape(B * N, -1),
            avail.reshape(B * N, T),
        )
        neigh_hist_enc = neigh_hist_enc.view(B, N, -1)

        no_data = avail.sum(dim=-1) == 0
        neigh_hist_enc = torch.where(
            no_data.unsqueeze(-1),
            torch.tensor(-float("inf"), device=neigh_hist_enc.device),
            neigh_hist_enc,
        )

        neighbor_pool = torch.amax(neigh_hist_enc, dim=1)

        neighbor_pool = torch.where(
            torch.isinf(neighbor_pool),
            torch.zeros_like(neighbor_pool),
            neighbor_pool,
        )
        return neighbor_pool


class Conv1DBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class MLP(nn.Module):
    """A simple multi-layer perceptron."""

    def __init__(self, in_dim, out_dim, hidden_dims, activation=nn.ReLU):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MapEncoder(nn.Module):
    """Encodes map, may output a global feature, feature map, or both."""

    def __init__(
        self,
        model_arch: str,
        input_image_shape: tuple = (3, 224, 224),
        global_feature_dim=64,
        grid_feature_dim=64,
    ) -> None:
        super().__init__()
        self.return_global_feat = global_feature_dim is not None
        self.return_grid_feat = grid_feature_dim is not None
        encoder = base_models.RasterizedMapEncoder(
            model_arch=model_arch,
            input_image_shape=input_image_shape,
            feature_dim=global_feature_dim,
        )
        self.input_image_shape = input_image_shape
        feat_nodes = {
            "map_model.layer1": "layer1",
            "map_model.layer2": "layer2",
            "map_model.layer3": "layer3",
            "map_model.layer4": "layer4",
            "map_model.fc": "fc",
        }
        self.encoder_heads = create_feature_extractor(encoder, feat_nodes)
        if self.return_grid_feat:
            encoder_channels = list(encoder.feature_channels().values())
            input_shape_scale = encoder.feature_scales()["layer4"]
            self.decoder = MapGridDecoder(
                input_shape=(
                    encoder_channels[-1],
                    input_image_shape[1] * input_shape_scale,
                    input_image_shape[2] * input_shape_scale,
                ),
                encoder_channels=encoder_channels[:-1],
                output_channel=grid_feature_dim,
                batchnorm=True,
            )
        self.encoder_feat_scales = list(encoder.feature_scales().values())

    def feat_map_out_dim(self, H, W):
        dim_scale = self.encoder_feat_scales[-4]
        return (H * dim_scale, W * dim_scale)

    def forward(self, map_inputs, encoder_feats=None):
        if encoder_feats is None:
            map_inputs = map_inputs.contiguous()
            encoder_feats = self.encoder_heads(map_inputs)
        fc_out = encoder_feats["fc"] if self.return_global_feat else None
        encoder_feats = [
            encoder_feats[k] for k in ["layer1", "layer2", "layer3", "layer4"]
        ]
        feat_map_out = None
        if self.return_grid_feat:
            feat_map_out = self.decoder.forward(
                feat_to_decode=encoder_feats[-1],
                encoder_feats=encoder_feats[:-1],
            )
        return fc_out, feat_map_out


class MapGridDecoder(nn.Module):
    def __init__(
        self,
        input_shape,
        output_channel,
        encoder_channels,
        bilinear=True,
        batchnorm=True,
    ):
        super().__init__()
        input_channel = input_shape[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                input_channel,
                512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),
        )

        self.up1 = Up(512 + encoder_channels[-1], 256, bilinear)
        self.up2 = Up(256 + encoder_channels[-2], 128, bilinear)
        self.up3 = Up(128 + encoder_channels[-3], 64, bilinear)

        self.conv2 = ConvBlock(
            64,
            [64, 64, 64],
            kernel_size=3,
            stride=1,
            batchnorm=batchnorm,
        )
        self.conv3 = nn.Conv2d(64, output_channel, kernel_size=1)
        self.out_norm = nn.GroupNorm(output_channel, output_channel)

    def forward(self, feat_to_decode: torch.Tensor, encoder_feats: List[torch.Tensor]):
        assert len(encoder_feats) >= 3
        x = self.conv1(feat_to_decode)
        x = self.up1(x, encoder_feats[-1])
        x = self.up2(x, encoder_feats[-2])
        x = self.up3(x, encoder_feats[-3])
        x = self.conv2(x)
        x = self.conv3(x)
        return self.out_norm(x)


class SceneAgentHistoryEncoder(nn.Module):
    def __init__(self, num_steps, out_dim=128, hidden=64, norm_info=None):
        super().__init__()
        self.num_steps = num_steps

        self.add_coeffs = torch.tensor(norm_info["add_coeffs"])
        self.div_coeffs = torch.tensor(norm_info["div_coeffs"])

        input_dim = 7
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, out_dim, kernel_size=3, dilation=4, padding=4),
            nn.ReLU(),
        )
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, pos, yaw, extent, avail):
        B, N, T, _ = pos.shape
        add_coeffs = self.add_coeffs.to(pos.device)
        div_coeffs = self.div_coeffs.to(pos.device)

        hvec = torch.cat([torch.cos(yaw), torch.sin(yaw)], -1)
        lw = extent.unsqueeze(2).expand(-1, -1, T, -1)

        pos = (pos - add_coeffs[:2][None, None]) / div_coeffs[:2][None, None]
        lw = (lw - add_coeffs[-2:][None, None]) / div_coeffs[-2:][None, None]

        feat = torch.cat([pos, hvec, lw, avail.unsqueeze(-1)], -1)
        feat = feat * avail.unsqueeze(-1)

        feat = feat.view(B * N, T, 7).transpose(1, 2)
        out = self.conv(feat).mean(-1)
        out = self.layer_norm(out).view(B, N, -1)

        padded_mask = avail.sum(-1) == 0
        out = out.masked_fill(padded_mask.unsqueeze(-1), 0.0)
        return out


class AgentStateTransformer(nn.Module):
    """
    Transformer encoder for interactions between agent states.

    Input: [B, N, state_dim], e.g. x, y, v, yaw.
    Output: [B, N, feature_dim].
    """

    def __init__(
        self,
        input_dim,
        feature_dim,
        nhead=4,
        num_layers=2,
        ff_dim=None,
        dropout=0.1,
    ):
        super().__init__()
        assert feature_dim % nhead == 0, "feature_dim must be divisible by nhead"

        if ff_dim is None:
            ff_dim = feature_dim * 4

        self.input_norm = nn.LayerNorm(input_dim)

        self.physics_encoder = nn.Sequential(
            nn.Linear(input_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),
        )

        self.spatial_encoder = SpatialPositionalEncoding(
            d_model=feature_dim,
            max_dist=50.0,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(self, x, agent_mask=None):
        """
        x: [B, N, input_dim] agent state features.
        agent_mask: [B, N], true for valid agents.
        """
        x = self.input_norm(x)

        physics_feat = self.physics_encoder(x)

        spatial_feat = self.spatial_encoder(x[..., :2])

        x = physics_feat + spatial_feat

        x = x + self.transformer(x, src_key_padding_mask=(~agent_mask))

        x = self.output_norm(x)

        return x


class SpatialPositionalEncoding(nn.Module):
    """Spatial positional encoding."""

    def __init__(self, d_model, max_dist=50.0):
        super().__init__()
        self.max_dist = max_dist
        self.d_model = d_model

        self.dist_embed = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, pos):
        """
        pos: [B, N, 2] position coordinates.
        """
        dist = torch.norm(pos, dim=-1, keepdim=True)
        dist = dist / self.max_dist
        pos_encoding = self.dist_embed(dist)
        return pos_encoding
