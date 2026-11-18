import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Any, Dict, Optional, Tuple


_GAUSSIAN_5X5_CACHE: Dict[Tuple[int, str, torch.dtype], torch.Tensor] = {}


class SeverityConditioner(nn.Module):
    """
    Encodes `(codec_family_id, normalized_severity)` into a compact conditioning vector (`F10` fix):
      0: `libx265` (HEVC Main 4:2:0)
      1: `libx264` (AVC High/Main 4:2:0)
    Both embedding rows are actively trained and evaluated in `tscope-v3`.
    """

    def __init__(
        self,
        num_families: int = 2,
        family_emb_dim: int = 8,
        cond_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_families = num_families
        self.family_emb = nn.Embedding(num_families, family_emb_dim)
        # Basis: [s, s^2, s^3, h_0.50(s), h_0.75(s), h_0.90(s)] -> 6 dims
        in_dim = family_emb_dim + 6
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
        )

    @staticmethod
    def severity_basis(s: torch.Tensor) -> torch.Tensor:
        if s.dim() == 1:
            s = s.unsqueeze(-1)
        s = torch.clamp(s, 0.0, 1.0)
        h50 = torch.clamp((s - 0.50) / 0.50, 0.0, 1.0)
        h75 = torch.clamp((s - 0.75) / 0.25, 0.0, 1.0)
        h90 = torch.clamp((s - 0.90) / 0.10, 0.0, 1.0)
        return torch.cat([s, s ** 2, s ** 3, h50, h75, h90], dim=-1)

    def forward(self, family_id: torch.Tensor, severity: torch.Tensor) -> torch.Tensor:
        fam_clamped = torch.clamp(family_id.long(), 0, self.num_families - 1)
        ef = self.family_emb(fam_clamped)
        phi_s = self.severity_basis(severity)
        return self.mlp(torch.cat([ef, phi_s], dim=-1))


class FiLMConvBlock(nn.Module):
    """
    Two 3x3 Conv + optional GroupNorm + GELU layers modulated by Feature-wise Linear Modulation (FiLM):
        FiLM(x, c) = x * (1 + gamma(c)) + beta(c)
    """

    def __init__(
        self, in_ch: int, out_ch: int, cond_dim: int = 64, use_norm: bool = True
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        if use_norm:
            num_groups = min(8, out_ch)
            while num_groups > 1 and (out_ch % num_groups) != 0:
                num_groups -= 1
            self.norm1: nn.Module = nn.GroupNorm(num_groups, out_ch)
            self.norm2: nn.Module = nn.GroupNorm(num_groups, out_ch)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act2 = nn.GELU()
        self.film_proj = nn.Linear(cond_dim, out_ch * 2)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        gamma_beta = self.film_proj(c).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        h = h * (1.0 + gamma) + beta
        return self.act2(h)


class TemporalConvGRUCell(nn.Module):
    """
    Lightweight Bottleneck ConvGRU at H/4 x W/4 resolution.
    Ensures temporal consistency of object attention A_t and residuals across video frames
    so x265/x264 Motion Estimation does not suffer from frame-to-frame pre-editing flicker.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.conv_gates = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1)
        self.conv_can = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)

    def forward(
        self, x_t: torch.Tensor, h_prev: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if h_prev is None:
            h_prev = torch.zeros_like(x_t)
        combined = torch.cat([x_t, h_prev], dim=1)
        gates = torch.sigmoid(self.conv_gates(combined))
        reset_gate, update_gate = torch.chunk(gates, 2, dim=1)
        candidate = torch.tanh(self.conv_can(torch.cat([x_t, reset_gate * h_prev], dim=1)))
        h_curr = (1.0 - update_gate) * h_prev + update_gate * candidate
        return h_curr


def gaussian_blur_5x5(x: torch.Tensor) -> torch.Tensor:
    """Fast separable 5x5 Gaussian blur with per-(channel, device, dtype) cached weights."""
    c = x.shape[1]
    key = (c, str(x.device), x.dtype)
    weight = _GAUSSIAN_5X5_CACHE.get(key)
    if weight is None:
        kernel_1d = torch.tensor(
            [1.0, 4.0, 6.0, 4.0, 1.0], device=x.device, dtype=x.dtype
        ) / 16.0
        kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, 5, 5)
        weight = kernel_2d.repeat(c, 1, 1, 1).contiguous()
        _GAUSSIAN_5X5_CACHE[key] = weight
    x_pad = F.pad(x, (2, 2, 2, 2), mode="reflect")
    return F.conv2d(x_pad, weight, groups=c)


@torch.no_grad()
def _estimate_coarse_translation(
    g_prev: torch.Tensor,
    g_curr: torch.Tensor,
    h: int,
    w: int,
    scale_div: int = 8,
    search_r: int = 6,
) -> torch.Tensor:
    """Coarse H/8 integer + subpixel translation search in float32 normalized affine coords."""
    b = g_curr.shape[0]
    device = g_curr.device
    hc = max(8, h // scale_div)
    wc = max(8, w // scale_div)
    r = min(search_r, max(1, min(hc, wc) // 4))

    gc = F.interpolate(g_curr, size=(hc, wc), mode="bilinear", align_corners=False)
    gp = F.interpolate(g_prev, size=(hc, wc), mode="bilinear", align_corners=False)

    core = gc[:, :, r : hc - r, r : wc - r]
    costs = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            patch = gp[:, :, r - dy : hc - r - dy, r - dx : wc - r - dx]
            tie = 1e-6 * float(dx * dx + dy * dy)
            diff = (core - patch).abs().mean(dim=(1, 2, 3)) + tie
            costs.append(diff)

    width = 2 * r + 1
    cost_vol = torch.stack(costs, dim=1).view(b, width, width)
    flat_idx = cost_vol.view(b, -1).argmin(dim=1)
    iy = torch.div(flat_idx, width, rounding_mode="floor")
    ix = flat_idx % width

    dx_f = (ix - r).float()
    dy_f = (iy - r).float()
    for bi in range(b):
        yi, xi = int(iy[bi].item()), int(ix[bi].item())
        if 0 < xi < 2 * r:
            c_m, c_0, c_p = cost_vol[bi, yi, xi - 1], cost_vol[bi, yi, xi], cost_vol[bi, yi, xi + 1]
            denom = 2.0 * (c_m - 2.0 * c_0 + c_p)
            if denom > 1e-8:
                dx_f[bi] += torch.clamp((c_m - c_p) / denom, -0.5, 0.5)
        if 0 < yi < 2 * r:
            c_m, c_0, c_p = cost_vol[bi, yi - 1, xi], cost_vol[bi, yi, xi], cost_vol[bi, yi + 1, xi]
            denom = 2.0 * (c_m - 2.0 * c_0 + c_p)
            if denom > 1e-8:
                dy_f[bi] += torch.clamp((c_m - c_p) / denom, -0.5, 0.5)

    sx = float(w) / float(wc)
    sy = float(h) / float(hc)
    theta = torch.zeros((b, 2, 3), device=device, dtype=torch.float32)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = -2.0 * (dx_f * sx) / float(w)
    theta[:, 1, 2] = -2.0 * (dy_f * sy) / float(h)
    return theta


@torch.no_grad()
def estimate_global_affine_irls(
    i_prev: torch.Tensor,
    i_curr: torch.Tensor,
    iters_per_level: Tuple[int, int] = (4, 3),
    corner_snap_px: float = 0.75,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    6-DOF Cauchy-IRLS global affine motion estimator (H/8 -> H/4) with corner-displacement static snap.
    Runs strictly in float32 (AMP-safe for linalg.solve) and returns `(theta [N,2,3], is_static [N])`.
    """
    n, _, h, w = i_curr.shape
    device = i_curr.device
    with torch.amp.autocast(device_type=device.type, enabled=False):
        id_theta = torch.zeros((n, 2, 3), device=device, dtype=torch.float32)
        id_theta[:, 0, 0] = 1.0
        id_theta[:, 1, 1] = 1.0
        if min(h, w) < 16 or n == 0:
            return id_theta, torch.ones((n,), dtype=torch.bool, device=device)

        ip_f = i_prev.float()
        ic_f = i_curr.float()
        luma_w = torch.tensor([0.299, 0.587, 0.114], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        g_prev_full = (ip_f * luma_w).sum(dim=1, keepdim=True)
        g_curr_full = (ic_f * luma_w).sum(dim=1, keepdim=True)

        theta = _estimate_coarse_translation(g_prev_full, g_curr_full, h=h, w=w, scale_div=8, search_r=6)
        eye6 = torch.eye(6, device=device, dtype=torch.float32).unsqueeze(0)

        for div, n_iter in zip((8, 4), iters_per_level):
            hs = max(8, h // div)
            ws = max(8, w // div)
            gc = F.interpolate(g_curr_full, size=(hs, ws), mode="bilinear", align_corners=False)
            gp = F.interpolate(g_prev_full, size=(hs, ws), mode="bilinear", align_corners=False)

            ys = (torch.arange(hs, device=device, dtype=torch.float32) + 0.5) * (2.0 / hs) - 1.0
            xs = (torch.arange(ws, device=device, dtype=torch.float32) + 0.5) * (2.0 / ws) - 1.0
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            gx_flat = gx.reshape(1, -1)
            gy_flat = gy.reshape(1, -1)

            gx_pad = F.pad(gc, (1, 1, 1, 1), mode="replicate")
            ix_flat = ((gx_pad[:, :, 1:-1, 2:] - gx_pad[:, :, 1:-1, :-2]) * (ws / 4.0)).reshape(n, -1)
            iy_flat = ((gx_pad[:, :, 2:, 1:-1] - gx_pad[:, :, :-2, 1:-1]) * (hs / 4.0)).reshape(n, -1)

            for _ in range(n_iter):
                grid = F.affine_grid(theta, size=(n, 1, hs, ws), align_corners=False)
                gp_w = F.grid_sample(gp, grid, mode="bilinear", padding_mode="border", align_corners=False)
                valid = ((grid[..., 0].abs() <= 0.96) & (grid[..., 1].abs() <= 0.96)).float().reshape(n, -1)

                gp_pad = F.pad(gp_w, (1, 1, 1, 1), mode="replicate")
                ix_w = ((gp_pad[:, :, 1:-1, 2:] - gp_pad[:, :, 1:-1, :-2]) * (ws / 4.0)).reshape(n, -1)
                iy_w = ((gp_pad[:, :, 2:, 1:-1] - gp_pad[:, :, :-2, 1:-1]) * (hs / 4.0)).reshape(n, -1)
                ix_f = 0.5 * (ix_w + ix_flat)
                iy_f = 0.5 * (iy_w + iy_flat)

                err = (gc - gp_w).reshape(n, -1)
                delta = 0.04
                w_irls = valid * (delta * delta) / (delta * delta + err * err)

                j_mat = torch.stack(
                    [ix_f * gx_flat, ix_f * gy_flat, ix_f, iy_f * gx_flat, iy_f * gy_flat, iy_f],
                    dim=-1,
                )
                jw = j_mat * w_irls.unsqueeze(-1)
                h_mat = torch.bmm(jw.transpose(1, 2), j_mat)
                rhs = torch.bmm(jw.transpose(1, 2), err.unsqueeze(-1))
                dp = torch.linalg.solve(h_mat + 1e-2 * eye6, rhs).squeeze(-1)
                dp = torch.clamp(dp, -0.25, 0.25)
                theta = theta + dp.view(n, 2, 3)

        corners = torch.tensor(
            [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            device=device,
            dtype=torch.float32,
        ).t()
        disp = torch.matmul(theta, corners) - corners[:2].unsqueeze(0)
        disp_px = torch.stack([disp[:, 0].abs() * (0.5 * w), disp[:, 1].abs() * (0.5 * h)], dim=1)
        max_corner_px = disp_px.amax(dim=(1, 2))
        flat_frame = (g_curr_full - g_prev_full).abs().amax(dim=(1, 2, 3)) < 1e-5
        is_static = (max_corner_px < corner_snap_px) | flat_frame
        theta = torch.where(is_static.view(n, 1, 1), id_theta, theta)
        return theta, is_static


class TemporalSCOPE(nn.Module):
    """
    T-SCOPE v3: Temporally-Consistent Multi-Codec Semantic Pre-Editor & QG Delta-QP Predictor.
      - `F10`: `num_families=2` (`0=libx265`, `1=libx264`).
      - `F01`: Sensor-noise coring on `diff_mc_t`, spatially smoothed motion-compensated background EMA
        locking, and background attention anti-flicker gating to genuinely suppress P-frame residual bitrate
        across 32-frame GOPs without spatial seam artifacts.
      - `F08`: `3x3` max-pool foreground halo guard before `compute_ctu_qp_offsets` so `+qoffset` never
        spills onto pedestrian boundaries.
      - `F17`: Genuine FLOP-saving P-frame bypass when `reuse_iframe_qp=True` (runs full U-Net only on
        keyframes `step % keyframe_stride == 0` and affine-warps `attn`/`eff_attn`/`i_bg` with a tiny
        `H/2` refinement head on inter-frames, cutting inter-frame neural MACs by >95%).
    """

    def __init__(
        self,
        base_ch: int = 48,
        num_families: int = 2,
        cond_dim: int = 64,
        max_obj_res: float = 0.025,
        max_bg_res: float = 0.03,
        qg_size: int = 16,
        max_delta_qp: float = 4.0,
        keyframe_stride: int = 32,
    ) -> None:
        super().__init__()
        self.base_ch = base_ch
        self.num_families = num_families
        self.cond_dim = cond_dim
        self.max_obj_res = max_obj_res
        self.max_bg_res = max_bg_res
        self.qg_size = qg_size
        self.max_delta_qp = max_delta_qp
        self.keyframe_stride = max(1, int(keyframe_stride))

        self.conditioner = SeverityConditioner(
            num_families=num_families, family_emb_dim=8, cond_dim=cond_dim
        )

        # Encoder: HxW -> H/2xW/2 -> H/4xW/4 (+ H/8xW/8 global context branch)
        self.enc1 = FiLMConvBlock(6, base_ch, cond_dim, use_norm=False)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1)
        self.enc2 = FiLMConvBlock(base_ch * 2, base_ch * 2, cond_dim, use_norm=True)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=4, stride=2, padding=1)
        self.bottleneck = FiLMConvBlock(base_ch * 4, base_ch * 4, cond_dim, use_norm=True)
        self.down3 = nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size=4, stride=2, padding=1)
        self.enc3 = FiLMConvBlock(base_ch * 4, base_ch * 4, cond_dim, use_norm=True)
        self.ctx_fuse = FiLMConvBlock(base_ch * 8, base_ch * 4, cond_dim, use_norm=True)

        # Recurrent Temporal Alignment at H/4 x W/4
        self.temporal_gru = TemporalConvGRUCell(base_ch * 4)

        # Decoder: H/4xW/4 -> H/2xW/2 -> HxW
        self.up2 = nn.Conv2d(base_ch * 4, base_ch * 2, kernel_size=3, padding=1)
        self.dec2 = FiLMConvBlock(base_ch * 4, base_ch * 2, cond_dim, use_norm=True)
        self.up1 = nn.Conv2d(base_ch * 2, base_ch, kernel_size=3, padding=1)
        self.dec1 = FiLMConvBlock(base_ch * 2, base_ch, cond_dim, use_norm=False)

        # Output prediction heads
        self.attn_head = nn.Conv2d(base_ch, 1, kernel_size=3, padding=1)
        self.attn_cond_bias = nn.Linear(cond_dim, 1)
        self.obj_res_head = nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)
        self.bg_res_head = nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)
        self.bg_blur_strength = nn.Sequential(
            nn.Linear(cond_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        # Temporal EMA gate head for static background locking across P-frames
        self.bg_temporal_gate = nn.Conv2d(base_ch, 1, kernel_size=3, padding=1)
        nn.init.constant_(self.bg_temporal_gate.bias, 1.5)

        # Lightweight H/2 inter-frame refinement head for `reuse_iframe_qp=True` P-frame bypass (F17 fix).
        # Zero-initialized final conv ensures smooth identity-warped start even without dedicated fine-tuning.
        self.inter_refine = nn.Sequential(
            nn.Conv2d(7, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 4, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.inter_refine[-1].weight)
        nn.init.zeros_(self.inter_refine[-1].bias)

        self.last_forward_stats: Dict[str, Any] = {}

    def estimate_gmacs(
        self,
        height: int = 384,
        width: int = 640,
        clip_len: int = 32,
        reuse_iframe_qp: bool = False,
        disable_temporal: bool = False,
        keyframe_stride: Optional[int] = None,
    ) -> Dict[str, float]:
        """Analytical MAC accounting for U-Net + ConvGRU vs inter-frame P-bypass (`F16`/`F17`)."""
        c = self.base_ch
        px = float(height * width)
        # Full U-Net + H/8 context + H/4 ConvGRU MACs per frame
        macs_enc1 = px * (6 * c * 9 + c * c * 9)
        macs_enc2 = (px / 4.0) * (c * (2 * c) * 16 + 2 * (2 * c) * (2 * c) * 9)
        macs_enc3_ctx = (px / 16.0) * (
            (2 * c) * (4 * c) * 16
            + 2 * (4 * c) * (4 * c) * 9
            + (8 * c) * (4 * c) * 9
            + (4 * c) * (4 * c) * 9
        ) + (px / 64.0) * ((4 * c) * (4 * c) * 16 + 2 * (4 * c) * (4 * c) * 9)
        macs_gru = 0.0 if disable_temporal else (px / 16.0) * ((8 * c) * (8 * c) * 9 + (8 * c) * (4 * c) * 9)
        macs_dec2 = (px / 4.0) * ((4 * c) * (2 * c) * 9 + (4 * c) * (2 * c) * 9 + (2 * c) * (2 * c) * 9)
        macs_dec1 = px * ((2 * c) * c * 9 + (2 * c) * c * 9 + c * c * 9)
        macs_heads = px * (c * 8 * 9 + 75.0)
        full_step_macs = macs_enc1 + macs_enc2 + macs_enc3_ctx + macs_gru + macs_dec2 + macs_dec1 + macs_heads

        # Lightweight H/2 inter-frame bypass MACs
        bypass_step_macs = (px / 4.0) * (7 * 16 * 9 + 16 * 4 * 9) + px * 90.0
        mc_pair_macs = 0.0 if (disable_temporal or clip_len <= 1) else px * 420.0

        kf_s = max(1, int(keyframe_stride or self.keyframe_stride))
        if reuse_iframe_qp and (not disable_temporal) and clip_len > 1:
            n_full = sum(1 for s in range(clip_len) if s % kf_s == 0)
            n_bypass = clip_len - n_full
        else:
            n_full = clip_len
            n_bypass = 0

        total_macs = n_full * full_step_macs + n_bypass * bypass_step_macs + max(0, clip_len - 1) * mc_pair_macs
        return {
            "total_gmacs": round(total_macs / 1e9, 4),
            "gmacs_per_frame": round((total_macs / max(1, clip_len)) / 1e9, 4),
            "full_step_gmacs": round(full_step_macs / 1e9, 4),
            "bypass_step_gmacs": round(bypass_step_macs / 1e9, 4),
            "full_unet_steps": int(n_full),
            "bypass_inter_steps": int(n_bypass),
        }

    def compute_ctu_qp_offsets(
        self,
        attn_map: torch.Tensor,
        reuse_iframe_across_gop: bool = False,
        disable_temporal: bool = False,
        halo_kernel: int = 3,
        keyframe_stride: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Converts pixel-level effective attention `A_t` `[B, T, 1, H, W]` into QG-level
        Delta-QP offsets `[B, T, H//qg_size, W//qg_size]` (`40x24` at `640x384`) with:
          1. A `3x3` max-pool foreground halo guard (`F08` fix) so `+qoffset` never spills
             onto pedestrian boundaries in adjacent `16x16` QG blocks.
          2. Foreground-protected non-decreasing P-frame background `+dQP` carry-forward.
        """
        squeezed = attn_map.dim() == 4
        if squeezed:
            attn_map = attn_map.unsqueeze(1)
        b, t, _, h, w = attn_map.shape
        attn_flat = attn_map.reshape(b * t, 1, h, w)
        if halo_kernel > 1:
            pad = halo_kernel // 2
            attn_guarded = F.max_pool2d(attn_flat, kernel_size=halo_kernel, stride=1, padding=pad)
        else:
            attn_guarded = attn_flat

        avg_qg = F.avg_pool2d(
            attn_guarded, kernel_size=self.qg_size, stride=self.qg_size
        )
        max_qg = F.max_pool2d(
            attn_guarded, kernel_size=self.qg_size, stride=self.qg_size
        )
        # 50/50 avg+max mix on 3x3-dilated mask ensures ANY single FG pixel >= 0.76 zeroes QG +dQP
        qg_attn = 0.50 * avg_qg + 0.50 * max_qg  # [B*T, 1, H_qg, W_qg]
        h_qg, w_qg = qg_attn.shape[-2], qg_attn.shape[-1]

        tau = 0.38
        bg_part = F.relu(tau - qg_attn) / tau
        dqp_frame = self.max_delta_qp * (1.75 * bg_part)
        dqp_frame = torch.clamp(dqp_frame, min=0.0, max=0.14 * 51.0)
        dqp_frame = dqp_frame.view(b, t, h_qg, w_qg)

        kf_s = max(1, int(keyframe_stride or self.keyframe_stride))
        if reuse_iframe_across_gop and t > 1:
            if t <= kf_s:
                out_qp = dqp_frame[:, 0:1].expand_as(dqp_frame)
            else:
                segs = []
                for start in range(0, t, kf_s):
                    end = min(t, start + kf_s)
                    segs.append(dqp_frame[:, start : start + 1].expand(b, end - start, h_qg, w_qg))
                out_qp = torch.cat(segs, dim=1)
            return out_qp.squeeze(1) if squeezed else out_qp

        if disable_temporal or t <= 1 or self.max_delta_qp <= 0.0:
            return dqp_frame.squeeze(1) if squeezed else dqp_frame

        qg_attn_t = qg_attn.view(b, t, h_qg, w_qg)
        max_qg_t = max_qg.view(b, t, h_qg, w_qg)
        out = [dqp_frame[:, 0]]
        prev_bg = dqp_frame[:, 0]
        for s in range(1, t):
            if s % kf_s == 0:
                prev_bg = dqp_frame[:, s]
                out.append(prev_bg)
                continue
            fg_score = torch.maximum(qg_attn_t[:, s], 0.90 * max_qg_t[:, s])
            fg_safe = 1.0 - torch.clamp((fg_score - 0.24) / 0.12, 0.0, 1.0)
            prev_smooth = 0.5 * prev_bg + 0.5 * F.avg_pool2d(
                prev_bg.unsqueeze(1), kernel_size=3, stride=1, padding=1, count_include_pad=False
            ).squeeze(1)
            prev_carry = torch.maximum(prev_bg, prev_smooth)
            curr = torch.clamp(
                torch.maximum(dqp_frame[:, s], fg_safe * prev_carry),
                min=0.0,
                max=0.14 * 51.0,
            )
            out.append(curr)
            prev_bg = torch.maximum(fg_safe * prev_bg, fg_safe * curr)

        out_qp = torch.stack(out, dim=1)
        return out_qp.squeeze(1) if squeezed else out_qp

    def _compute_smooth_bg_blur(self, i_t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        blur_mild = gaussian_blur_5x5(i_t)
        blur_strong = F.interpolate(
            gaussian_blur_5x5(F.avg_pool2d(blur_mild, kernel_size=2, stride=2)),
            size=i_t.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        blur_mix = self.bg_blur_strength(c).view(-1, 1, 1, 1)
        return (1.0 - blur_mix) * blur_mild + blur_mix * blur_strong

    @staticmethod
    def _compute_motion_bg_confidence(
        diff_raw: torch.Tensor, valid_full: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        diff_mag = diff_raw.abs().mean(dim=1, keepdim=True)
        motion_mag = gaussian_blur_5x5(diff_mag)
        v_mask = valid_full if valid_full is not None else torch.ones_like(motion_mag)
        raw_conf = v_mask * torch.exp(-28.0 * motion_mag) * torch.clamp(
            1.0 - motion_mag / 0.08, 0.0, 1.0
        )
        static_bg_conf = gaussian_blur_5x5(raw_conf)
        return motion_mag, torch.clamp(static_bg_conf, 0.0, 1.0)

    def forward_single_step(
        self,
        i_t: torch.Tensor,
        i_prev_warped: torch.Tensor,
        c: torch.Tensor,
        h_prev_warped: Optional[torch.Tensor] = None,
        bg_prev_warped: Optional[torch.Tensor] = None,
        eff_attn_prev_warped: Optional[torch.Tensor] = None,
        valid_full: Optional[torch.Tensor] = None,
        valid_h4: Optional[torch.Tensor] = None,
        sev_gate: Optional[torch.Tensor] = None,
        disable_temporal: bool = False,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
    ]:
        """Single full U-Net step `t` with sensor-noise-cored motion diff and locked background EMA."""
        diff_raw = i_t - i_prev_warped
        diff_valid = diff_raw * valid_full if valid_full is not None else diff_raw
        # Soft deadzone suppresses camera sensor grain on P-frames so static background matches I-frame statistics
        diff_mc_t = torch.sign(diff_valid) * F.relu(diff_valid.abs() - 0.012)
        x_in = torch.cat([i_t, diff_mc_t], dim=1)

        f1 = self.enc1(x_in, c)
        f2 = self.enc2(self.down1(f1), c)
        f3 = self.bottleneck(self.down2(f2), c)
        f4 = self.enc3(self.down3(f3), c)
        f4_up = F.interpolate(f4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        f3_ctx = self.ctx_fuse(torch.cat([f3, f4_up], dim=1), c)

        if disable_temporal:
            h_curr = self.temporal_gru(f3_ctx, None)
        else:
            h_in = (
                h_prev_warped * valid_h4
                if (h_prev_warped is not None and valid_h4 is not None)
                else h_prev_warped
            )
            h_curr = self.temporal_gru(f3_ctx, h_in)

        u2 = F.interpolate(h_curr, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.dec2(torch.cat([self.up2(u2), f2], dim=1), c)

        u1 = F.interpolate(u2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.dec1(torch.cat([self.up1(u1), f1], dim=1), c)

        attn_bias = self.attn_cond_bias(c).view(-1, 1, 1, 1)
        attn_logits = self.attn_head(u1) + attn_bias
        attn_t = torch.sigmoid(attn_logits)

        # Smoothstep STE saturation removes FG background-blur leak while preserving gradients
        u_sat = torch.clamp((attn_t - 0.12) / 0.46, 0.0, 1.0)
        eff_hard = u_sat * u_sat * (3.0 - 2.0 * u_sat)
        eff_proxy = torch.sigmoid(2.0 * (attn_logits + 0.65))
        eff_attn_raw = eff_proxy + (eff_hard - eff_proxy).detach()

        motion_mag: Optional[torch.Tensor] = None
        static_bg_conf: Optional[torch.Tensor] = None
        if bg_prev_warped is not None or eff_attn_prev_warped is not None:
            motion_mag, static_bg_conf = self._compute_motion_bg_confidence(diff_raw, valid_full)

        # Anti-flicker background attention lock: suppresses spurious P-frame clutter spikes while
        # immediately passing high-confidence or moving foreground (`eff_attn_raw >= 0.45`).
        if eff_attn_prev_warped is not None and static_bg_conf is not None:
            fg_override = torch.clamp((eff_attn_raw - 0.45) / 0.25, 0.0, 1.0)
            bg_lock_w = (
                0.70
                * static_bg_conf
                * (1.0 - fg_override)
                * torch.clamp(1.0 - eff_attn_prev_warped.detach(), 0.0, 1.0)
            )
            eff_attn_t = (1.0 - bg_lock_w) * eff_attn_raw + bg_lock_w * eff_attn_prev_warped.detach()
        else:
            eff_attn_t = eff_attn_raw

        s_gate = 1.0 if sev_gate is None else sev_gate
        delta_obj_t = (self.max_obj_res * s_gate) * eff_attn_t * torch.tanh(self.obj_res_head(u1))
        delta_bg_raw = self.max_bg_res * torch.tanh(self.bg_res_head(u1))

        i_obj_t = torch.clamp(i_t + delta_obj_t, 0.0, 1.0)
        blur_i_t = self._compute_smooth_bg_blur(i_t, c)

        if bg_prev_warped is not None and static_bg_conf is not None:
            # Damp per-frame background residual jitter on verified static background and blend with spatially smooth weight
            bg_res_damp = 1.0 - 0.75 * static_bg_conf
            delta_bg_t = bg_res_damp * delta_bg_raw
            i_bg_spatial = torch.clamp(blur_i_t + delta_bg_t, 0.0, 1.0)
            gate_s = gaussian_blur_5x5(torch.sigmoid(self.bg_temporal_gate(u1)))
            temporal_weight = torch.clamp(0.90 * gate_s * static_bg_conf, 0.0, 0.92)
            i_bg_t = temporal_weight * bg_prev_warped + (1.0 - temporal_weight) * i_bg_spatial
        else:
            delta_bg_t = delta_bg_raw
            i_bg_t = torch.clamp(blur_i_t + delta_bg_t, 0.0, 1.0)

        i_edit_t = eff_attn_t * i_obj_t + (1.0 - eff_attn_t) * i_bg_t
        return i_edit_t, attn_t, eff_attn_t, delta_obj_t, delta_bg_t, h_curr, i_bg_t

    def forward_inter_bypass_step(
        self,
        i_t: torch.Tensor,
        i_prev_warped: torch.Tensor,
        c: torch.Tensor,
        attn_prev_warped: torch.Tensor,
        eff_attn_prev_warped: torch.Tensor,
        bg_prev_warped: torch.Tensor,
        valid_full: Optional[torch.Tensor] = None,
        sev_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Genuine FLOP-saving inter-frame P-bypass (`F17` fix for `reuse_iframe_qp=True`):
        Skips the entire U-Net (`enc1..dec1`) and `temporal_gru`, propagating keyframe/previous
        `attn_t`, `eff_attn_t`, and `i_bg_t` via 6-DOF affine motion warping + a lightweight `H/2`
        refinement and disocclusion gate (>95% inter-frame neural MAC reduction).
        """
        h, w = i_t.shape[-2], i_t.shape[-1]
        diff_raw = i_t - i_prev_warped
        diff_valid = diff_raw * valid_full if valid_full is not None else diff_raw
        diff_mc_t = torch.sign(diff_valid) * F.relu(diff_valid.abs() - 0.012)

        motion_mag, static_bg_conf = self._compute_motion_bg_confidence(diff_raw, valid_full)

        inp_full = torch.cat([i_t, diff_mc_t, eff_attn_prev_warped], dim=1)
        if h >= 4 and w >= 4:
            inp_half = F.avg_pool2d(inp_full, kernel_size=2, stride=2)
            ref_half = self.inter_refine(inp_half)
            ref_full = F.interpolate(ref_half, size=(h, w), mode="bilinear", align_corners=False)
        else:
            ref_full = self.inter_refine(inp_full)

        d_attn_logit = ref_full[:, :1]
        d_obj_raw = ref_full[:, 1:]

        # Promote newly disoccluded / fast-moving foreground while keeping warped foreground sharp.
        # Compute logit/sigmoid fusion strictly in FP32 so `1.0 - 1e-4` never rounds to `1.0` in FP16
        # (where `torch.logit(1.0) = +inf` and `SigmoidBackward * LogitBackward = 0 * inf = NaN`).
        with torch.amp.autocast(device_type=i_t.device.type, enabled=False):
            mot_promote = torch.clamp((motion_mag.float() - 0.045) / 0.08, 0.0, 1.0)
            base_raw = torch.maximum(attn_prev_warped.float(), 0.80 * mot_promote)
            base_prob = torch.nan_to_num(base_raw, nan=0.5, posinf=1.0 - 1e-4, neginf=1e-4).clamp(
                1e-4, 1.0 - 1e-4
            )
            attn_logits_fp32 = torch.logit(base_prob) + d_attn_logit.float()
            attn_t = torch.sigmoid(attn_logits_fp32).to(dtype=i_t.dtype)

            u_sat = torch.clamp((attn_t.float() - 0.12) / 0.46, 0.0, 1.0)
            eff_hard = (u_sat * u_sat * (3.0 - 2.0 * u_sat)).to(dtype=i_t.dtype)
            eff_proxy = torch.sigmoid(2.0 * (attn_logits_fp32 + 0.65)).to(dtype=i_t.dtype)
            eff_attn_t = eff_proxy + (eff_hard - eff_proxy).detach()

        s_gate = 1.0 if sev_gate is None else sev_gate
        delta_obj_t = (self.max_obj_res * s_gate) * eff_attn_t * torch.tanh(d_obj_raw)
        delta_bg_t = torch.zeros_like(i_t)

        i_obj_t = torch.clamp(i_t + delta_obj_t, 0.0, 1.0)
        blur_i_t = self._compute_smooth_bg_blur(i_t, c)
        temporal_weight = torch.clamp(0.90 * static_bg_conf, 0.0, 0.92)
        i_bg_t = temporal_weight * bg_prev_warped + (1.0 - temporal_weight) * blur_i_t
        i_edit_t = eff_attn_t * i_obj_t + (1.0 - eff_attn_t) * i_bg_t
        return i_edit_t, attn_t, eff_attn_t, delta_obj_t, delta_bg_t, i_bg_t

    def forward(
        self,
        video: torch.Tensor,
        family_id: torch.Tensor,
        severity: torch.Tensor,
        reuse_iframe_qp: bool = False,
        disable_temporal: bool = False,
        disable_mc: bool = False,
        keyframe_stride: Optional[int] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            video: `[B, T, 3, H, W]` or `[B, 3, H, W]` in `[0, 1]`.
            family_id: `[B]` integer codec family index (`0=libx265`, `1=libx264`).
            severity: `[B]` normalized compression severity in `[0, 1]`.
            reuse_iframe_qp: If True, activates genuine FLOP-saving P-frame bypass on inter-frames
                (`step % keyframe_stride != 0`) and reuses keyframe QG Delta-QP across the GOP.
            disable_temporal: If True, processes every frame independently (`h_prev=None`, `bg_prev=None`,
                `i_prev=i_t`) for a strict framewise Spatial-SCOPE baseline.
            disable_mc: Optional ablation toggle disabling 6-DOF Cauchy-IRLS affine warping while keeping ConvGRU.
            keyframe_stride: Intra/keyframe refresh period for `reuse_iframe_qp=True` (default `self.keyframe_stride=32`).
        """
        single_frame_input = video.dim() == 4
        if single_frame_input:
            video = video.unsqueeze(1)

        b, t, _, h, w = video.shape
        kf_s = max(1, int(keyframe_stride or self.keyframe_stride))
        c = self.conditioner(family_id, severity)
        sev_t = severity.to(device=video.device, dtype=video.dtype).reshape(-1)
        if sev_t.numel() == 1 and b > 1:
            sev_t = sev_t.expand(b)
        sev_gate = 0.30 + 0.70 * torch.clamp(sev_t.view(b, 1, 1, 1), 0.0, 1.0)

        use_motion_comp = (t > 1) and (not disable_temporal)
        aligned_prev_orig: Optional[torch.Tensor] = None
        warp_valid_mask: Optional[torch.Tensor] = None
        aligned_prev_edit: Optional[torch.Tensor] = None
        grid_full_pairs: Optional[torch.Tensor] = None
        grid_h4_pairs: Optional[torch.Tensor] = None
        valid_h4_pairs: Optional[torch.Tensor] = None
        is_static_pairs: Optional[torch.Tensor] = None
        st_mask_flat: Optional[torch.Tensor] = None

        if use_motion_comp:
            bt_pairs = b * (t - 1)
            prev_flat = video[:, :-1].reshape(bt_pairs, 3, h, w)
            curr_flat = video[:, 1:].reshape(bt_pairs, 3, h, w)
            with torch.no_grad():
                h4, w4 = max(1, h // 4), max(1, w // 4)
                if disable_mc:
                    thetas_flat = torch.zeros((bt_pairs, 2, 3), device=video.device, dtype=torch.float32)
                    thetas_flat[:, 0, 0] = 1.0
                    thetas_flat[:, 1, 1] = 1.0
                    is_static_flat = torch.ones((bt_pairs,), device=video.device, dtype=torch.bool)
                else:
                    thetas_flat, is_static_flat = estimate_global_affine_irls(prev_flat, curr_flat)

                grid_full_flat = F.affine_grid(
                    thetas_flat.to(dtype=video.dtype), size=(bt_pairs, 3, h, w), align_corners=False
                )
                grid_h4_flat = F.affine_grid(
                    thetas_flat.to(dtype=video.dtype),
                    size=(bt_pairs, self.base_ch * 4, h4, w4),
                    align_corners=False,
                )
                warped_orig = F.grid_sample(
                    prev_flat, grid_full_flat, mode="bilinear", padding_mode="border", align_corners=False
                )
                valid_raw = (
                    (grid_full_flat[..., 0].abs() <= 0.985) & (grid_full_flat[..., 1].abs() <= 0.985)
                ).to(dtype=video.dtype).unsqueeze(1)
                valid_h4_raw = (
                    (grid_h4_flat[..., 0].abs() <= 0.96) & (grid_h4_flat[..., 1].abs() <= 0.96)
                ).to(dtype=video.dtype).unsqueeze(1)
                st_mask_flat = is_static_flat.view(bt_pairs, 1, 1, 1)
                aligned_prev_orig = torch.where(st_mask_flat, prev_flat, warped_orig).view(
                    b, t - 1, 3, h, w
                )
                warp_valid_mask = torch.where(
                    st_mask_flat, torch.ones_like(valid_raw), valid_raw
                ).view(b, t - 1, 1, h, w)
                valid_h4_pairs = torch.where(
                    st_mask_flat, torch.ones_like(valid_h4_raw), valid_h4_raw
                ).view(b, t - 1, 1, h4, w4)
                grid_full_pairs = grid_full_flat.view(b, t - 1, h, w, 2)
                grid_h4_pairs = grid_h4_flat.view(b, t - 1, h4, w4, 2)
                is_static_pairs = is_static_flat.view(b, t - 1)

        edits, attns, eff_attns, d_objs, d_bgs = [], [], [], [], []
        h_prev: Optional[torch.Tensor] = None
        bg_prev: Optional[torch.Tensor] = None
        attn_prev: Optional[torch.Tensor] = None
        eff_attn_prev: Optional[torch.Tensor] = None

        use_ckpt = self.training and torch.is_grad_enabled() and t > 1
        full_unet_steps = 0
        bypass_inter_steps = 0

        def _warp_2d_map(
            tensor_2d: Optional[torch.Tensor],
            grid_2d: torch.Tensor,
            st_mask: torch.Tensor,
            all_static: bool,
            pad_mode: str = "border",
        ) -> Optional[torch.Tensor]:
            if tensor_2d is None or all_static:
                return tensor_2d
            return torch.where(
                st_mask,
                tensor_2d,
                F.grid_sample(
                    tensor_2d,
                    grid_2d.to(dtype=tensor_2d.dtype),
                    mode="bilinear",
                    padding_mode=pad_mode,
                    align_corners=False,
                ),
            )

        for step in range(t):
            i_t = video[:, step]
            is_keyframe = disable_temporal or (step == 0) or (
                reuse_iframe_qp and (step % kf_s == 0)
            )
            run_bypass = (
                reuse_iframe_qp
                and (not disable_temporal)
                and (step % kf_s != 0)
                and (attn_prev is not None)
                and (eff_attn_prev is not None)
                and (bg_prev is not None)
            )

            if disable_temporal or step == 0:
                i_prev_w = i_t
                valid_full_s: Optional[torch.Tensor] = None
                valid_h4_s: Optional[torch.Tensor] = None
                h_prev_w = None
                bg_prev_w = None
                attn_prev_w = None
                eff_attn_prev_w = None
            else:
                p_idx = step - 1
                assert aligned_prev_orig is not None and warp_valid_mask is not None
                assert (
                    is_static_pairs is not None
                    and grid_full_pairs is not None
                    and grid_h4_pairs is not None
                    and valid_h4_pairs is not None
                )
                i_prev_w = aligned_prev_orig[:, p_idx]
                valid_full_s = warp_valid_mask[:, p_idx]
                valid_h4_s = valid_h4_pairs[:, p_idx]
                all_static = bool(is_static_pairs[:, p_idx].all())
                st_step = is_static_pairs[:, p_idx].view(b, 1, 1, 1)
                grid_f = grid_full_pairs[:, p_idx]
                grid_4 = grid_h4_pairs[:, p_idx]

                h_prev_w = (
                    None
                    if run_bypass
                    else _warp_2d_map(h_prev, grid_4, st_step, all_static, pad_mode="zeros")
                )
                bg_prev_w = _warp_2d_map(bg_prev, grid_f, st_step, all_static, pad_mode="border")
                attn_prev_w = _warp_2d_map(attn_prev, grid_f, st_step, all_static, pad_mode="zeros")
                eff_attn_prev_w = _warp_2d_map(
                    eff_attn_prev, grid_f, st_step, all_static, pad_mode="zeros"
                )

            if run_bypass:
                assert attn_prev_w is not None and eff_attn_prev_w is not None and bg_prev_w is not None
                bypass_inter_steps += 1
                i_edit_t, attn_t, eff_attn_t, d_obj_t, d_bg_t, bg_prev = self.forward_inter_bypass_step(
                    i_t,
                    i_prev_w,
                    c,
                    attn_prev_w,
                    eff_attn_prev_w,
                    bg_prev_w,
                    valid_full_s,
                    sev_gate,
                )
            else:
                full_unet_steps += 1
                if is_keyframe and reuse_iframe_qp and step > 0:
                    h_prev_w = None
                if use_ckpt:
                    i_edit_t, attn_t, eff_attn_t, d_obj_t, d_bg_t, h_prev, bg_prev = grad_checkpoint(
                        self.forward_single_step,
                        i_t,
                        i_prev_w,
                        c,
                        h_prev_w,
                        bg_prev_w,
                        eff_attn_prev_w,
                        valid_full_s,
                        valid_h4_s,
                        sev_gate,
                        disable_temporal,
                        use_reentrant=False,
                    )
                else:
                    i_edit_t, attn_t, eff_attn_t, d_obj_t, d_bg_t, h_prev, bg_prev = self.forward_single_step(
                        i_t,
                        i_prev_w,
                        c,
                        h_prev_w,
                        bg_prev_w,
                        eff_attn_prev_w,
                        valid_full_s,
                        valid_h4_s,
                        sev_gate,
                        disable_temporal,
                    )

            attn_prev = attn_t
            eff_attn_prev = eff_attn_t
            edits.append(i_edit_t)
            attns.append(attn_t)
            eff_attns.append(eff_attn_t)
            d_objs.append(d_obj_t)
            d_bgs.append(d_bg_t)

        gmac_info = self.estimate_gmacs(
            height=h,
            width=w,
            clip_len=t,
            reuse_iframe_qp=reuse_iframe_qp,
            disable_temporal=disable_temporal,
            keyframe_stride=kf_s,
        )
        gmac_info["full_unet_steps"] = full_unet_steps
        gmac_info["bypass_inter_steps"] = bypass_inter_steps
        self.last_forward_stats = gmac_info

        i_edit = torch.stack(edits, dim=1)
        attn_map = torch.stack(attns, dim=1)
        eff_attn_map = torch.stack(eff_attns, dim=1)
        delta_obj = torch.stack(d_objs, dim=1)
        delta_bg = torch.stack(d_bgs, dim=1)
        ctu_qp_offsets = self.compute_ctu_qp_offsets(
            eff_attn_map,
            reuse_iframe_across_gop=reuse_iframe_qp,
            disable_temporal=disable_temporal,
            halo_kernel=3,
            keyframe_stride=kf_s,
        )

        if use_motion_comp:
            assert grid_full_pairs is not None and st_mask_flat is not None
            bt_pairs = b * (t - 1)
            prev_edit_flat = i_edit[:, :-1].reshape(bt_pairs, 3, h, w)
            warped_edit = F.grid_sample(
                prev_edit_flat,
                grid_full_pairs.reshape(bt_pairs, h, w, 2).to(dtype=i_edit.dtype),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            aligned_prev_edit = torch.where(st_mask_flat, prev_edit_flat, warped_edit).view(
                b, t - 1, 3, h, w
            )

        if single_frame_input:
            return {
                "i_edit": i_edit.squeeze(1),
                "attn_map": attn_map.squeeze(1),
                "eff_attn_map": eff_attn_map.squeeze(1),
                "delta_obj": delta_obj.squeeze(1),
                "delta_bg": delta_bg.squeeze(1),
                "ctu_qp_offsets": ctu_qp_offsets.squeeze(1),
                "aligned_prev_edit": None,
                "aligned_prev_orig": None,
                "warp_valid_mask": None,
            }

        return {
            "i_edit": i_edit,
            "attn_map": attn_map,
            "eff_attn_map": eff_attn_map,
            "delta_obj": delta_obj,
            "delta_bg": delta_bg,
            "ctu_qp_offsets": ctu_qp_offsets,
            "aligned_prev_edit": aligned_prev_edit,
            "aligned_prev_orig": aligned_prev_orig,
            "warp_valid_mask": warp_valid_mask,
        }
