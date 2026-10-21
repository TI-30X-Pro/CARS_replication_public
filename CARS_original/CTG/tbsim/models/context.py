import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor

import tbsim.models.base_models as base_models
from tbsim.models.context_encoder import MapGridDecoder


class MapEncoder(nn.Module):
    """
    Encode a BEV raster map into:
      - grid_feat: [B, Cg, Hf, Wf] for trajectory-grid sampling / cross-attention
      - global_feat: [B, Cg] from average pooling over the same grid feature map
    """

    def __init__(
        self,
        model_arch: str,
        input_image_shape: tuple = (3, 224, 224),
        grid_feature_dim: int = 64,
    ) -> None:
        super().__init__()
        self.input_image_shape = input_image_shape

        self.backbone = base_models.RasterizedMapEncoder(
            model_arch=model_arch,
            input_image_shape=input_image_shape,
            feature_dim=None,
        )

        self.encoder_heads = create_feature_extractor(
            self.backbone,
            {
                "map_model.layer1": "l1",
                "map_model.layer2": "l2",
                "map_model.layer3": "l3",
                "map_model.layer4": "l4",
            },
        )

        with torch.no_grad():
            C, H, W = input_image_shape
            dummy = torch.zeros(1, C, H, W)
            feats = self.encoder_heads(dummy)
            C1, H1, W1 = feats["l1"].shape[1:]
            C2, H2, W2 = feats["l2"].shape[1:]
            C3, H3, W3 = feats["l3"].shape[1:]
            C4, H4, W4 = feats["l4"].shape[1:]

        self.decoder = MapGridDecoder(
            input_shape=(C4, H4, W4),
            encoder_channels=[C1, C2, C3],
            output_channel=grid_feature_dim,
            batchnorm=True,
        )

        with torch.no_grad():
            grid = self.decoder(
                feat_to_decode=torch.zeros(1, C4, H4, W4),
                encoder_feats=[
                    torch.zeros(1, C1, H1, W1),
                    torch.zeros(1, C2, H2, W2),
                    torch.zeros(1, C3, H3, W3),
                ],
            )
            Hf, Wf = grid.shape[-2:]
            self._scale_h = Hf / H
            self._scale_w = Wf / W

        self.grid_feature_dim = grid_feature_dim

    def feat_map_out_dim(self, H, W):
        """Infer grid feature map size for an arbitrary input resolution."""
        Hf = int(round(H * self._scale_h))
        Wf = int(round(W * self._scale_w))
        return (Hf, Wf)

    def forward(self, map_inputs: torch.Tensor, encoder_feats: dict = None):
        """
        Args:
            map_inputs: [B, 3, H, W], expected to match the initialization shape.

        Returns:
            global_feat: [B, Cg]
            grid_feat: [B, Cg, Hf, Wf]
        """
        x = map_inputs.contiguous()

        if encoder_feats is None:
            feats = self.encoder_heads(x)
        else:
            feats = encoder_feats

        l1, l2, l3, l4 = feats["l1"], feats["l2"], feats["l3"], feats["l4"]
        grid_feat = self.decoder(feat_to_decode=l4, encoder_feats=[l1, l2, l3])
        global_feat = grid_feat.mean(dim=(2, 3))

        return global_feat, grid_feat


class AgentHistoryEncoderTCN(nn.Module):
    def __init__(self, num_steps, out_dim, norm_info, hidden_dim=64):
        super().__init__()

        mean, std = norm_info
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

        self.state_dim = 7
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.conv1 = nn.Conv1d(
            self.state_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            dilation=1,
        )
        self.conv2 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=2,
            dilation=2,
        )
        self.norm1 = nn.GroupNorm(1, hidden_dim)
        self.norm2 = nn.GroupNorm(1, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, pos, speed, yaw, extent, avail):
        """
        Args:
            pos: [B, T, 2]
            speed: [B, T]
            yaw: [B, T]
            extent: [B, 2]
            avail: [B, T] bool / {0, 1}
        """
        B, T = pos.shape[:2]
        mean = self.mean.to(pos.device)
        std = self.std.to(pos.device).clamp_min(1e-6)

        heading_vec = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
        pos_n = (pos - mean[:2]) / std[:2]
        speed_n = (speed.unsqueeze(-1) - mean[2:3]) / std[2:3]
        extent_n = (extent.unsqueeze(1).repeat(1, T, 1) - mean[3:5]) / std[3:5]

        feat = torch.cat([pos_n, speed_n, heading_vec, extent_n], dim=-1)
        feat = feat * avail.unsqueeze(-1).float()
        feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        x = self.act(self.norm1(self.conv1(feat.permute(0, 2, 1))))
        x = self.act(self.norm2(self.conv2(x))).permute(0, 2, 1)

        mask = avail.unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        return self.fc_out(pooled)


class EgoEncoder(nn.Module):
    def __init__(self, out_dim, norm_info):
        super().__init__()
        mean, std = norm_info
        self.register_buffer("mean_pos", torch.tensor(mean[:2], dtype=torch.float32))
        self.register_buffer("std_pos", torch.tensor(std[:2], dtype=torch.float32))
        self.register_buffer("mean_speed", torch.tensor([mean[2]], dtype=torch.float32))
        self.register_buffer("std_speed", torch.tensor([std[2]], dtype=torch.float32))

        hidden = 64
        self.mlp = nn.Sequential(
            nn.Linear(2 + 2 + 1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, pos, speed, yaw):
        """
        Args:
            pos: [B, 2], ego position relative to center.
            speed: [B], ego scalar speed.
            yaw: [B], ego heading relative to center.
        """
        device = pos.device
        heading = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
        pos_n = (pos - self.mean_pos.to(device)) / self.std_pos.to(device).clamp_min(1e-6)
        speed_n = (
            (speed.unsqueeze(-1) - self.mean_speed.to(device))
            / self.std_speed.to(device).clamp_min(1e-6)
        )

        feat = torch.cat([pos_n, heading, speed_n], dim=-1)
        token = self.mlp(feat)
        return token


class NeighborHistoryEncoder(nn.Module):
    def __init__(self, num_steps, out_dim=128, norm_info=None):
        super().__init__()
        self.neigh_encode = AgentHistoryEncoderTCN(
            num_steps=num_steps,
            out_dim=out_dim,
            norm_info=norm_info,
            hidden_dim=64,
        )

    def forward(self, pos, speed, yaw, extent, avail):
        """
        Args:
            pos: [B, N, T, 2]
            speed: [B, N, T]
            yaw: [B, N, T]
            extent: [B, N, 2]
            avail: [B, N, T] bool / {0, 1}

        Returns:
            neigh_tokens: [B, N, D]
            neigh_mask: [B, N], true when a neighbor is invalid for all frames.
        """
        B, N, T, _ = pos.size()

        pos_flat = pos.reshape(B * N, T, -1)
        speed_flat = speed.reshape(B * N, T)
        yaw_flat = yaw.reshape(B * N, T)
        extent_flat = extent.reshape(B * N, -1)
        avail_flat = avail.reshape(B * N, T)

        neigh_hist_enc_flat = self.neigh_encode(
            pos_flat,
            speed_flat,
            yaw_flat,
            extent_flat,
            avail_flat,
        )

        neigh_tokens = neigh_hist_enc_flat.view(B, N, -1)
        neigh_mask = avail.sum(dim=-1) == 0

        return neigh_tokens, neigh_mask


class AgentsMHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, center_token, neigh_tokens, neigh_mask):
        q = center_token.unsqueeze(1)

        all_masked = neigh_mask.all(dim=1)
        safe_mask = neigh_mask.clone()
        safe_mask[all_masked, 0] = False

        attn_out, _ = self.attn(q, neigh_tokens, neigh_tokens, key_padding_mask=safe_mask)
        attn_out = attn_out.squeeze(1)
        attn_out = attn_out.masked_fill(all_masked.unsqueeze(1), 0.0)

        return self.norm(center_token + attn_out)
