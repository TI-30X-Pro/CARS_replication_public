"""Neural-network blocks for the Gaussian-mixture diffusion model.

Parameter names preserve compatibility with the released checkpoints.

Context: a ResNet-18 map encoder with a U-Net style decoder to a 56 x 56 x 32 feature grid,
a temporal-conv encoder for the agent's history and one for each neighbour's, attention from
the agent to its neighbours, and an MLP over [agent token, mean map feature, city one-hot].
Denoiser: dilated temporal convs over the noisy action sequence, FiLM by the diffusion time,
cross-attention to the context token and to the map grid, map features sampled along the
decoded trajectory, and per-step mixture parameters (logits, means, scales) for K components.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims):
        super().__init__()
        dims = [in_dim, *hidden_dims, out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Map encoder

class _Backbone(nn.Module):
    """torchvision ResNet-18 with a fresh first conv, no pooling head (RasterizedMapEncoder)."""

    def __init__(self, in_channels=3):
        super().__init__()
        self.map_model = resnet18()
        self.map_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.map_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.map_model.fc = nn.Identity()

    def features(self, x):
        m = self.map_model
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        l1 = m.layer1(x)
        l2 = m.layer2(l1)
        l3 = m.layer3(l2)
        return l1, l2, l3, m.layer4(l3)


class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class _Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = _DoubleConv(in_ch, out_ch, in_ch // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy, dx = x2.size(2) - x1.size(2), x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, filters, kernel_size=3):
        super().__init__()
        f1, f2, f3 = filters
        self.conv1 = nn.Conv2d(in_ch, f1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        self.conv2 = nn.Conv2d(f1, f2, kernel_size, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(f2)
        self.conv3 = nn.Conv2d(f2, f3, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(f3)
        self.shortcut = nn.Sequential(nn.Conv2d(in_ch, f3, 1, bias=False), nn.BatchNorm2d(f3))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class _MapGridDecoder(nn.Module):
    def __init__(self, in_ch=512, enc_ch=(64, 128, 256), out_ch=32):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, 512, 3, padding=1, bias=False), nn.ReLU(True))
        self.up1 = _Up(512 + enc_ch[2], 256)
        self.up2 = _Up(256 + enc_ch[1], 128)
        self.up3 = _Up(128 + enc_ch[0], 64)
        self.conv2 = _ConvBlock(64, [64, 64, 64])
        self.conv3 = nn.Conv2d(64, out_ch, kernel_size=1)
        self.out_norm = nn.GroupNorm(out_ch, out_ch)

    def forward(self, l4, l1, l2, l3):
        x = self.conv1(l4)
        x = self.up1(x, l3)
        x = self.up2(x, l2)
        x = self.up3(x, l1)
        return self.out_norm(self.conv3(self.conv2(x)))


class MapEncoder(nn.Module):
    """(B, 3, 224, 224) raster -> grid (B, C, 56, 56) and its spatial mean (B, C)."""

    def __init__(self, grid_feature_dim=32, in_channels=3):
        super().__init__()
        self.backbone = _Backbone(in_channels)
        self.decoder = _MapGridDecoder(out_ch=grid_feature_dim)

    def forward(self, maps):
        l1, l2, l3, l4 = self.backbone.features(maps.contiguous())
        grid = self.decoder(l4, l1, l2, l3)
        return grid.mean(dim=(2, 3)), grid


# Agent encoders and attention

class HistoryEncoder(nn.Module):
    """History (B, T) of positions, speed, heading, extent -> one token (AgentHistoryEncoderTCN).
    Unavailable steps are zeroed and left out of the mean pooling."""

    def __init__(self, out_dim, norm_info, hidden_dim=64):
        super().__init__()
        mean, std = norm_info
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.conv1 = nn.Conv1d(7, hidden_dim, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.norm1 = nn.GroupNorm(1, hidden_dim)
        self.norm2 = nn.GroupNorm(1, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, pos, speed, yaw, extent, avail):
        T = pos.shape[1]
        mean, std = self.mean, self.std.clamp_min(1e-6)
        heading = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
        pos_n = (pos - mean[:2]) / std[:2]
        speed_n = (speed.unsqueeze(-1) - mean[2:3]) / std[2:3]
        extent_n = (extent.unsqueeze(1).repeat(1, T, 1) - mean[3:5]) / std[3:5]
        feat = torch.cat([pos_n, speed_n, heading, extent_n], dim=-1) * avail.unsqueeze(-1).float()
        feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        x = self.act(self.norm1(self.conv1(feat.permute(0, 2, 1))))
        x = self.act(self.norm2(self.conv2(x))).permute(0, 2, 1)
        mask = avail.unsqueeze(-1).float()
        return self.fc_out((x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0))


class NeighborHistoryEncoder(nn.Module):
    """(B, N, T, ...) -> tokens (B, N, D) and a mask of neighbours with no history at all."""

    def __init__(self, out_dim, norm_info):
        super().__init__()
        self.neigh_encode = HistoryEncoder(out_dim, norm_info)

    def forward(self, pos, speed, yaw, extent, avail):
        B, N, T, _ = pos.shape
        tok = self.neigh_encode(pos.reshape(B * N, T, -1), speed.reshape(B * N, T), yaw.reshape(B * N, T),
                                extent.reshape(B * N, -1), avail.reshape(B * N, T))
        return tok.view(B, N, -1), avail.sum(dim=-1) == 0


class EgoEncoder(nn.Module):
    """In the checkpoints but not used by the model's forward pass."""

    def __init__(self, out_dim, norm_info):
        super().__init__()
        mean, std = norm_info
        self.register_buffer("mean_pos", torch.tensor(mean[:2], dtype=torch.float32))
        self.register_buffer("std_pos", torch.tensor(std[:2], dtype=torch.float32))
        self.register_buffer("mean_speed", torch.tensor([mean[2]], dtype=torch.float32))
        self.register_buffer("std_speed", torch.tensor([std[2]], dtype=torch.float32))
        self.mlp = nn.Sequential(nn.Linear(5, 64), nn.ReLU(inplace=True), nn.Linear(64, out_dim))


class AgentsMHA(nn.Module):
    """The agent's token attends to its neighbours' (residual + LayerNorm)."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, center, neigh, neigh_mask):
        all_masked = neigh_mask.all(dim=1)
        safe = neigh_mask.clone()
        safe[all_masked, 0] = False
        out, _ = self.attn(center.unsqueeze(1), neigh, neigh, key_padding_mask=safe)
        out = out.squeeze(1).masked_fill(all_masked.unsqueeze(1), 0.0)
        return self.norm(center + out)


# Diffusion denoiser

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = torch.exp(torch.arange(half, device=x.device) * -(math.log(10000) / (half - 1)))
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ResidualDilatedConv1d(nn.Module):
    """Parallel dilated convs (GroupNorm, ReLU), concatenated, plus a 1x1 shortcut."""

    def __init__(self, in_ch, out_ch, dilations=(1, 2, 4), kernel_size=3):
        super().__init__()
        part = out_ch // len(dilations)
        self.branches = nn.ModuleList(
            nn.Sequential(nn.Conv1d(in_ch, part, kernel_size, padding=d * (kernel_size - 1) // 2, dilation=d),
                          nn.GroupNorm(1, part), nn.ReLU(inplace=True))
            for d in dilations)
        extra = out_ch % len(dilations)
        self.extra_branch = nn.Sequential(
            nn.Conv1d(in_ch, extra, kernel_size, padding=(kernel_size - 1) // 2), nn.GroupNorm(1, extra),
            nn.ReLU(inplace=True)) if extra else None
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        outs = [b(x) for b in self.branches]
        if self.extra_branch is not None:
            outs.append(self.extra_branch(x))
        return torch.cat(outs, dim=1) + self.shortcut(x)


class Denoiser(nn.Module):
    """(x_t (B, T, 2), context (B, C), time in [0, 1] (B,), map grid, map features along the
    trajectory (B, T, Cg)) -> mixture logits (B, T, K), means (B, T, K, 2), log scales
    (B, T, K, 2) with scale = softplus(.) + 0.1 (ConvCrossAttnDiffuser)."""

    def __init__(self, in_dim, cond_dim, time_emb_dim, hidden_dim=64, out_dim=2, dilations=(1, 2, 4),
                 n_heads=4, grid_map_dim=32, grid_map_traj_dim=32, mix_gauss=3):
        super().__init__()
        self.out_dim, self.mix_gauss = out_dim, mix_gauss
        self.temporal_conv = ResidualDilatedConv1d(in_dim, hidden_dim, dilations, 3)
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(time_emb_dim), nn.Linear(time_emb_dim, hidden_dim * 2))
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.cond_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.cond_norm = nn.LayerNorm(hidden_dim)
        self.cond_ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.ReLU(inplace=True),
                                      nn.Linear(hidden_dim * 4, hidden_dim))
        self.cond_ffn_norm = nn.LayerNorm(hidden_dim)
        self.grid_map_pos_mlp = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU(inplace=True),
                                              nn.Linear(hidden_dim, hidden_dim))
        self.grid_map_gate = nn.Parameter(torch.zeros(1))
        self.grid_map_proj = nn.Linear(grid_map_dim, hidden_dim)
        self.grid_map_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.grid_map_norm = nn.LayerNorm(hidden_dim)
        self.register_buffer("grid_pos_cache", None, persistent=False)
        self.grid_traj_proj = nn.Linear(grid_map_traj_dim, hidden_dim)
        self.grid_traj_norm = nn.LayerNorm(hidden_dim)
        self.traj_pos_emb = nn.Embedding(512, hidden_dim)
        self.logits_pi = nn.Conv1d(hidden_dim, mix_gauss, kernel_size=1)
        self.logits_mu = nn.Conv1d(hidden_dim, out_dim * mix_gauss, kernel_size=1)
        self.logits_sigma = nn.Conv1d(hidden_dim, out_dim * mix_gauss, kernel_size=1)
        nn.init.constant_(self.logits_sigma.bias, -1.0)

    def _grid_pos(self, B, H, W, device):
        if self.grid_pos_cache is None or self.grid_pos_cache.shape[1] != H * W:
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=device), torch.linspace(-1, 1, W, device=device),
                                    indexing="ij")
            self.grid_pos_cache = torch.stack([yy, xx], dim=-1).view(1, H * W, 2)
        return self.grid_pos_cache.expand(B, -1, -1)

    def forward(self, x, cond, t, grid, grid_traj):
        h = self.temporal_conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        gamma, beta = self.time_mlp(t).chunk(2, dim=-1)
        h = gamma.unsqueeze(1) * h + beta.unsqueeze(1)

        c = self.cond_proj(cond).unsqueeze(1)
        h = self.cond_norm(h + self.cond_attn(h, c, c)[0])
        h = self.cond_ffn_norm(h + self.cond_ffn(h))

        B, C, H, W = grid.shape
        m = self.grid_map_proj(grid.view(B, C, -1).permute(0, 2, 1)) + self.grid_map_pos_mlp(self._grid_pos(B, H, W, grid.device))
        h = self.grid_map_norm(h + torch.sigmoid(self.grid_map_gate) * self.grid_map_attn(h, m, m)[0])

        T = h.size(1)
        h = h + self.traj_pos_emb(torch.arange(T, device=h.device)).unsqueeze(0)
        h = self.grid_traj_norm(h + self.grid_traj_proj(grid_traj))

        h = h.permute(0, 2, 1)
        K, D = self.mix_gauss, self.out_dim
        logits = self.logits_pi(h).permute(0, 2, 1).view(B, T, K)
        mu = self.logits_mu(h).permute(0, 2, 1).view(B, T, K, D)
        sigma = F.softplus(self.logits_sigma(h).permute(0, 2, 1).view(B, T, K, D)) + 0.1
        return logits, mu, torch.log(torch.clamp(sigma, min=1e-6))


def query_feature_grid(pos, grid):
    """Bilinear lookup of grid (B, C, H, W) at pixel positions pos (B, T, 2) -> (B, T, C)."""
    B, T, _ = pos.shape
    x = torch.clamp(pos[..., 0].reshape(-1), 0, grid.shape[-1] - 1 - 1e-3)
    y = torch.clamp(pos[..., 1].reshape(-1), 0, grid.shape[-2] - 1 - 1e-3)
    x0, y0 = torch.floor(x).long(), torch.floor(y).long()
    x1, y1 = x0 + 1, y0 + 1
    g = torch.permute(grid, (0, 2, 3, 1))
    b = torch.arange(g.size(0), device=g.device)[:, None].expand(B, T).reshape(-1)
    ia, ib, ic, id_ = g[b, y0, x0], g[b, y1, x0], g[b, y0, x1], g[b, y1, x1]
    x0, x1, y0, y1 = x0.float(), x1.float(), y0.float(), y1.float()
    norm = 1.0 / ((x1 - x0) * (y1 - y0))
    wa = (x1 - x) * (y1 - y) * norm
    wb = (x1 - x) * (y - y0) * norm
    wc = (x - x0) * (y1 - y) * norm
    wd = (x - x0) * (y - y0) * norm
    out = wa.unsqueeze(1) * ia + wb.unsqueeze(1) * ib + wc.unsqueeze(1) * ic + wd.unsqueeze(1) * id_
    return out.reshape(B, T, -1)
