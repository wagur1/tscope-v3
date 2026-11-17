import json
import math
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_FFMPEG_FILTER_SCRIPT_FLAG: Optional[str] = None


def detect_ffmpeg_filter_script_flag() -> str:
    """
    Detects the CLI option supported by the installed `ffmpeg` binary for reading a video
    filtergraph from a text file:
      - FFmpeg 4.x / 5.x / 6.x / 7.0 (standard Linux / Kaggle Ubuntu): `-filter_script:v`
      - FFmpeg 7.1+ / 8.x / 9.x (modern builds where `-filter_script` was removed): `-/vf`
    Cached once per process so 32-frame full-QG `addroi` filtergraphs (> 4000 chars) never hit
    OS command-line length limits on either Windows or Linux.
    """
    global _FFMPEG_FILTER_SCRIPT_FLAG
    if _FFMPEG_FILTER_SCRIPT_FLAG is not None:
        return _FFMPEG_FILTER_SCRIPT_FLAG

    if shutil.which("ffmpeg") is None:
        _FFMPEG_FILTER_SCRIPT_FLAG = "-filter_script:v"
        return _FFMPEG_FILTER_SCRIPT_FLAG

    with tempfile.TemporaryDirectory() as td:
        vf_file = Path(td) / "probe_vf.txt"
        vf_file.write_text("null", encoding="utf-8")
        raw_in = Path(td) / "probe.rgb"
        np.zeros((1, 16, 16, 3), dtype=np.uint8).tofile(str(raw_in))
        for candidate in ("-filter_script:v", "-/vf", "-/filter:v"):
            cmd = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "16x16",
                "-r",
                "25",
                "-i",
                str(raw_in),
                candidate,
                str(vf_file),
                "-frames:v",
                "1",
                "-f",
                "null",
                "-",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                _FFMPEG_FILTER_SCRIPT_FLAG = candidate
                return _FFMPEG_FILTER_SCRIPT_FLAG

    _FFMPEG_FILTER_SCRIPT_FLAG = "-/vf"
    return _FFMPEG_FILTER_SCRIPT_FLAG


def create_dct_matrix(
    n: int = 8,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Orthonormal 1D DCT-II matrix of size NxN for block transform simulation."""
    mat = torch.zeros((n, n), device=device, dtype=dtype)
    for k in range(n):
        for i in range(n):
            alpha = math.sqrt(1.0 / n) if k == 0 else math.sqrt(2.0 / n)
            mat[k, i] = alpha * math.cos(math.pi * (2 * i + 1) * k / (2.0 * n))
    return mat


def estimate_motion_compensated_pred(
    curr: torch.Tensor,
    prev: torch.Tensor,
    block_size: int = 16,
) -> torch.Tensor:
    """
    Coarse-to-fine motion-compensated inter prediction (H/8 global +-24px + H/2 16x16 block +-8px).
    Motion vectors are computed under no_grad while F.grid_sample preserves gradients through `prev`.
    """
    b, _, h, w = curr.shape
    if h < 16 or w < 16:
        return prev

    device = curr.device
    dtype = prev.dtype
    with torch.no_grad():
        c_l = curr.detach().float().mean(dim=1, keepdim=True)
        p_l = prev.detach().float().mean(dim=1, keepdim=True)

        g_dy = torch.zeros((b,), device=device, dtype=torch.float32)
        g_dx = torch.zeros((b,), device=device, dtype=torch.float32)
        if h >= 32 and w >= 32:
            c8 = F.avg_pool2d(c_l, 8)
            p8 = F.avg_pool2d(p_l, 8)
            h8, w8 = c8.shape[-2], c8.shape[-1]
            r8 = min(3, max(0, min(h8, w8) // 2 - 1))
            if r8 >= 1:
                p8_pad = F.pad(p8, (r8, r8, r8, r8), mode="replicate")
                cand_shifts = [(0, 0)] + [
                    (dy, dx)
                    for dy in range(-r8, r8 + 1)
                    for dx in range(-r8, r8 + 1)
                    if not (dy == 0 and dx == 0)
                ]
                errs = []
                for dy, dx in cand_shifts:
                    sh = p8_pad[:, :, r8 + dy : r8 + dy + h8, r8 + dx : r8 + dx + w8]
                    errs.append((c8 - sh).abs().mean(dim=(1, 2, 3)))
                best_idx = torch.stack(errs, dim=1).argmin(dim=1)
                shifts_t = torch.tensor(cand_shifts, device=device, dtype=torch.float32)
                g_dy = shifts_t[best_idx, 0] * 8.0
                g_dx = shifts_t[best_idx, 1] * 8.0

        gy = (
            (2.0 * (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h) - 1.0
        ).view(1, h, 1).expand(b, h, w)
        gx = (
            (2.0 * (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w) - 1.0
        ).view(1, 1, w).expand(b, h, w)

        if h >= 32 and w >= 32 and h % block_size == 0 and w % block_size == 0:
            c2 = F.avg_pool2d(c_l, 2)
            p2 = F.avg_pool2d(p_l, 2)
            h2, w2 = c2.shape[-2], c2.shape[-1]
            bs2 = block_size // 2
            bh, bw = h // block_size, w // block_size

            gy2 = (
                (2.0 * (torch.arange(h2, device=device, dtype=torch.float32) + 0.5) / h2) - 1.0
            ).view(1, h2, 1).expand(b, h2, w2)
            gx2 = (
                (2.0 * (torch.arange(w2, device=device, dtype=torch.float32) + 0.5) / w2) - 1.0
            ).view(1, 1, w2).expand(b, h2, w2)
            g_grid2 = torch.stack(
                [
                    gx2 + (g_dx.view(b, 1, 1) * 0.5) * (2.0 / w2),
                    gy2 + (g_dy.view(b, 1, 1) * 0.5) * (2.0 / h2),
                ],
                dim=-1,
            )
            p2_global = F.grid_sample(
                p2, g_grid2, mode="bilinear", padding_mode="border", align_corners=False
            )

            r2 = min(4, max(2, min(h2, w2) // 4))
            p2_pad = F.pad(p2_global, (r2, r2, r2, r2), mode="replicate")
            best_sad = torch.full((b, 1, bh, bw), float("inf"), device=device, dtype=torch.float32)
            best_dy2 = torch.zeros_like(best_sad)
            best_dx2 = torch.zeros_like(best_sad)
            local_offsets = [(0, 0)] + [
                (dy2, dx2)
                for dy2 in range(-r2, r2 + 1, 2)
                for dx2 in range(-r2, r2 + 1, 2)
                if not (dy2 == 0 and dx2 == 0)
            ]
            for dy2, dx2 in local_offsets:
                sh2 = p2_pad[:, :, r2 + dy2 : r2 + dy2 + h2, r2 + dx2 : r2 + dx2 + w2]
                sad = F.avg_pool2d((c2 - sh2).abs(), bs2, bs2)
                better = sad < best_sad
                best_sad = torch.where(better, sad, best_sad)
                best_dy2 = torch.where(better, float(dy2), best_dy2)
                best_dx2 = torch.where(better, float(dx2), best_dx2)

            dx_full = g_dx.view(b, 1, 1) + (
                best_dx2.repeat_interleave(block_size, dim=2).repeat_interleave(block_size, dim=3)[:, 0]
                * 2.0
            )
            dy_full = g_dy.view(b, 1, 1) + (
                best_dy2.repeat_interleave(block_size, dim=2).repeat_interleave(block_size, dim=3)[:, 0]
                * 2.0
            )
        else:
            dx_full = g_dx.view(b, 1, 1).expand(b, h, w)
            dy_full = g_dy.view(b, 1, 1).expand(b, h, w)

        mv_grid = torch.stack(
            [gx + dx_full * (2.0 / w), gy + dy_full * (2.0 / h)],
            dim=-1,
        ).to(dtype=dtype)

    return F.grid_sample(prev, mv_grid, mode="bilinear", padding_mode="border", align_corners=False)


class DifferentiableVideoProxy(nn.Module):
    """
    Differentiable Multi-Codec (`libx265` HEVC & `libx264` AVC) QP- and QG-Delta-QP-Aware Proxy.
      - Preserves full `qg_size=16` (`H//16 x W//16`) spatial quantization granularity without
        the legacy `16x16` macro-cell bilinear downsampling cap (`F08` & `F12` fix).
      - Models codec-family rate/distortion scaling across `family_id=0` (`libx265`, HEVC) and
        `family_id=1` (`libx264`, AVC) (`F10` fix).
    """

    def __init__(self, block_size: int = 8) -> None:
        super().__init__()
        self.block_size = block_size
        dct_1d = create_dct_matrix(block_size)
        self.register_buffer("dct_mat", dct_1d)

        u = torch.arange(block_size, dtype=torch.float32).unsqueeze(1)
        v = torch.arange(block_size, dtype=torch.float32).unsqueeze(0)
        freq_weight = 1.0 + 0.25 * (u + v)
        self.register_buffer("freq_weight", freq_weight.view(1, 1, 1, block_size, block_size))
        ac_intra_weight = torch.ones((1, 1, 1, block_size, block_size), dtype=torch.float32)
        ac_intra_weight[..., 0, 0] = 0.12
        self.register_buffer("ac_intra_weight", ac_intra_weight)

    def block_dct2d(self, x: torch.Tensor) -> torch.Tensor:
        """Applies 2D DCT to non-overlapping 8x8 blocks of x [B, C, H, W]."""
        b, c, h, w = x.shape
        bs = self.block_size
        x_blocks = x.view(b, c, h // bs, bs, w // bs, bs).permute(0, 1, 2, 4, 3, 5)
        x_blocks = x_blocks.reshape(b, c, -1, bs, bs)
        dct_m = self.dct_mat.to(dtype=x.dtype)
        coeff = torch.matmul(dct_m, torch.matmul(x_blocks, dct_m.t()))
        return coeff

    def block_idct2d(self, coeff: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Applies 2D IDCT to 8x8 block coefficients [B, C, N_blocks, 8, 8]."""
        b, c, _, bs, _ = coeff.shape
        dct_m = self.dct_mat.to(dtype=coeff.dtype)
        blocks = torch.matmul(dct_m.t(), torch.matmul(coeff, dct_m))
        blocks = blocks.view(b, c, h // bs, w // bs, bs, bs).permute(0, 1, 2, 4, 3, 5)
        return blocks.reshape(b, c, h, w)

    def quantize_blocks(
        self,
        x: torch.Tensor,
        q_step_map: torch.Tensor,
        is_inter: bool = False,
        block_qp: Optional[torch.Tensor] = None,
        rate_family_mult: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable DCT quantization + calibrated C-infinity deadzone/SKIP entropy rate estimation.
        """
        b, c, h, w = x.shape
        calc_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = x.to(dtype=calc_dtype) if x.dtype != calc_dtype else x
            q_step_map = (
                q_step_map.to(dtype=calc_dtype) if q_step_map.dtype != calc_dtype else q_step_map
            )
            if block_qp is not None and block_qp.dtype != calc_dtype:
                block_qp = block_qp.to(dtype=calc_dtype)

            coeff = self.block_dct2d(x)
            effective_q = q_step_map * self.freq_weight.to(dtype=calc_dtype)

            scaled = coeff / (effective_q + 1e-5)
            rounded = torch.round(scaled)
            quantized = scaled + (rounded - scaled).detach()

            recon_coeff = quantized * effective_q
            x_recon = self.block_idct2d(recon_coeff, h, w)

            mag = torch.sqrt(scaled * scaled + 1e-6)
            gate = torch.sigmoid((mag - 0.65) / 0.22)
            if block_qp is None:
                q_mult = 1.15 if is_inter else 1.0
                qp_flat = (
                    22.0
                    + 6.0 * torch.log2((q_step_map / (0.008 * q_mult)).clamp_min(1e-6))
                )
            else:
                qp_flat = block_qp.view(b, 1, -1, 1, 1).to(dtype=calc_dtype)
            if is_inter:
                rate_scale = 0.68 * torch.exp(-0.0947 * (qp_flat - 22.0))
                rate_weight = 1.0
            else:
                rate_scale = (1.45 * 1.659) * torch.exp(-0.0300 * (qp_flat - 22.0))
                rate_weight = self.ac_intra_weight.to(dtype=calc_dtype)
            if rate_family_mult is not None:
                rate_scale = rate_scale * rate_family_mult.to(dtype=calc_dtype).view(b, 1, 1, 1, 1)
            rate_bpp = (rate_scale * rate_weight * gate * torch.log2(1.0 + mag)).mean(
                dim=(1, 2, 3, 4)
            )
        return x_recon, rate_bpp

    def forward(
        self,
        video_edit: torch.Tensor,
        severity: torch.Tensor,
        ctu_qp_offsets: Optional[torch.Tensor] = None,
        family_id: Optional[Union[int, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            video_edit: [B, T, 3, H, W] in [0, 1].
            severity: [B] normalized severity in [0, 1] (maps to CRF/QP in [22, 47]).
            ctu_qp_offsets: Optional [B, T, H_qg, W_qg] Delta-QP offsets.
            family_id: Optional codec family index (0 = libx265/HEVC, 1 = libx264/AVC).
        """
        b, t, c, h, w = video_edit.shape
        bs = self.block_size
        num_blocks_h, num_blocks_w = h // bs, w // bs
        calc_dtype = (
            torch.float32
            if video_edit.dtype in (torch.float16, torch.bfloat16)
            else video_edit.dtype
        )

        with torch.amp.autocast(device_type=video_edit.device.type, enabled=False):
            video_edit = video_edit.to(dtype=calc_dtype) if video_edit.dtype != calc_dtype else video_edit
            severity = severity.to(dtype=calc_dtype) if severity.dtype != calc_dtype else severity
            if ctu_qp_offsets is not None and ctu_qp_offsets.dtype != calc_dtype:
                ctu_qp_offsets = ctu_qp_offsets.to(dtype=calc_dtype)

            if family_id is None:
                is_avc = torch.zeros((b,), device=video_edit.device, dtype=calc_dtype)
            elif isinstance(family_id, int):
                is_avc = torch.full(
                    (b,), 1.0 if int(family_id) == 1 else 0.0, device=video_edit.device, dtype=calc_dtype
                )
            else:
                fam_t = family_id.to(device=video_edit.device).reshape(-1)
                if fam_t.numel() == 1 and b > 1:
                    fam_t = fam_t.expand(b)
                is_avc = (fam_t == 1).to(dtype=calc_dtype)

            # AVC (x264) exhibits ~1.18x higher entropy rate and slightly coarser 8x8/4x4 reconstruction at same CRF
            q_family_mult = (1.0 + 0.06 * is_avc).view(b, 1, 1, 1)
            rate_family_mult = 1.0 + 0.18 * is_avc

            base_qp = 22.0 + 25.0 * severity.view(b, 1, 1, 1)

            recon_frames = []
            frame_rates = []
            prev_recon: Optional[torch.Tensor] = None

            for step in range(t):
                if ctu_qp_offsets is not None:
                    d_qp = ctu_qp_offsets[:, step].unsqueeze(1)  # [B, 1, H_qg, W_qg]
                    # Preserve full QG resolution (e.g. 24x40 at 384x640, 32x32 at 512x512) with nearest mapping to 8x8 TUs
                    d_qp_8x8 = F.interpolate(
                        d_qp, size=(num_blocks_h, num_blocks_w), mode="nearest"
                    )
                else:
                    d_qp_8x8 = torch.zeros(
                        (b, 1, num_blocks_h, num_blocks_w),
                        device=video_edit.device,
                        dtype=calc_dtype,
                    )

                d_qp_8x8 = torch.clamp(d_qp_8x8, min=-0.18 * 51.0, max=0.14 * 51.0)
                block_qp = torch.clamp(base_qp + d_qp_8x8, min=0.0, max=51.0)
                q_step = (0.008 * q_family_mult) * torch.pow(2.0, (block_qp - 22.0) / 6.0)
                q_step_flat = q_step.view(b, 1, -1, 1, 1)

                curr = video_edit[:, step]
                if step == 0 or prev_recon is None:
                    recon_t, rate_t = self.quantize_blocks(
                        curr,
                        q_step_flat,
                        is_inter=False,
                        block_qp=block_qp,
                        rate_family_mult=rate_family_mult,
                    )
                    recon_t = torch.clamp(recon_t, 0.0, 1.0)
                else:
                    pred_t = estimate_motion_compensated_pred(curr, prev_recon, block_size=16)
                    res_t = curr - pred_t
                    recon_res_t, rate_t = self.quantize_blocks(
                        res_t,
                        q_step_flat * 1.15,
                        is_inter=True,
                        block_qp=block_qp,
                        rate_family_mult=rate_family_mult,
                    )
                    recon_t = torch.clamp(pred_t + recon_res_t, 0.0, 1.0)

                recon_frames.append(recon_t)
                frame_rates.append(rate_t)
                prev_recon = recon_t

            video_recon = torch.stack(recon_frames, dim=1)
            rate_per_frame = torch.stack(frame_rates, dim=1)
            i_frame_bpp_proxy = rate_per_frame[:, 0]
            p_frame_bpp_proxy = (
                rate_per_frame[:, 1:].mean(dim=1)
                if t > 1
                else torch.zeros_like(i_frame_bpp_proxy)
            )
            rate_proxy_bpp = rate_per_frame.mean(dim=1)
        return {
            "video_recon": video_recon,
            "rate_proxy_bpp": rate_proxy_bpp,
            "rate_per_frame": rate_per_frame,
            "i_frame_bpp_proxy": i_frame_bpp_proxy,
            "p_frame_bpp_proxy": p_frame_bpp_proxy,
        }


class HybridX265STE(nn.Module):
    """
    Hybrid Forward-Real-Codec (`libx265` HEVC & `libx264` AVC) / Backward-Proxy STE (`tscope-v3`):
      - `F13`: Enforces explicit `-pix_fmt yuv420p` (Main 4:2:0) on all encoded MP4 bitstreams.
      - `F10`: Supports both `family_id=0` (`libx265`) and `family_id=1` (`libx264`) with `aq-mode=2` + `addroi`.
      - `F08` & `F12`: Preserves full `qg_size=16` (`H//16 x W//16`, e.g. `40x24` at `640x384`) `addroi` granularity
        with horizontal run-length merging and automatic filter-script file offloading (> 4,000 chars).
      - `F05`: Supports configurable `gop_len` (e.g. `keyint=32`) and `bframes` (`0` Low-Delay P `IPPP` and `4`
        Random Access `IBBP`) with exact PTS-ordered packet-to-frame byte accounting under B-frame reordering.
    """

    def __init__(
        self,
        use_real_x265_in_forward: bool = False,
        preset: str = "medium",
        gop_len: Optional[int] = None,
        bframes: int = 0,
    ) -> None:
        super().__init__()
        self.proxy = DifferentiableVideoProxy(block_size=8)
        self.use_real_x265 = use_real_x265_in_forward and (shutil.which("ffmpeg") is not None)
        self.preset = preset
        self.gop_len = gop_len
        self.bframes = int(bframes)

    @staticmethod
    def _build_addroi_filter_from_offsets(
        ctu_qp_offsets: Optional[torch.Tensor],
        h: int,
        w: int,
        qg_size: int = 16,
    ) -> List[str]:
        """
        Converts dynamic per-frame spatial QG Delta-QP tensor `[T, H_qg, W_qg]` (or `[H_qg, W_qg]`)
        into an exact per-frame `ffmpeg` `addroi` filtergraph at full `H//qg_size x W//qg_size`
        spatial resolution (`40x24` on `640x384`, `32x32` on `512x512`) with horizontal run-length merging.
        """
        if ctu_qp_offsets is None:
            return []
        if ctu_qp_offsets.dim() == 2:
            grid_t = ctu_qp_offsets.detach().unsqueeze(0).unsqueeze(1)
        elif ctu_qp_offsets.dim() == 3:
            grid_t = ctu_qp_offsets.detach().unsqueeze(1)
        else:
            return []

        if float(grid_t.abs().max().item()) < 1e-4:
            return []

        t_frames = grid_t.shape[0]
        gh = max(1, h // qg_size)
        gw = max(1, w // qg_size)
        if tuple(grid_t.shape[-2:]) == (gh, gw):
            grid_qg = grid_t[:, 0]
        else:
            # Nearest-neighbor resize preserves exact zero foreground halo boundaries
            grid_qg = F.interpolate(grid_t.float(), size=(gh, gw), mode="nearest")[:, 0]
        cell_h = max(1, h // gh)
        cell_w = max(1, w // gw)

        def build_frame_rois(frame_grid: torch.Tensor) -> List[str]:
            frame_np = frame_grid.detach().cpu().numpy()
            rois: List[str] = []
            for r in range(gh):
                y0 = r * cell_h
                bh = (h - y0) if r == gh - 1 else cell_h
                if bh <= 0:
                    continue
                c = 0
                while c < gw:
                    d_qp = float(frame_np[r, c])
                    if abs(d_qp) < 0.15:
                        c += 1
                        continue
                    qoff = max(-0.18, min(0.14, d_qp / 51.0))
                    c_end = c + 1
                    qoff_sum = qoff
                    while c_end < gw:
                        d_qp_next = float(frame_np[r, c_end])
                        if abs(d_qp_next) < 0.15:
                            break
                        qoff_next = max(-0.18, min(0.14, d_qp_next / 51.0))
                        if abs(qoff_next - qoff) > 0.005:
                            break
                        qoff_sum += qoff_next
                        c_end += 1
                    qoff_avg = qoff_sum / float(c_end - c)
                    x0 = c * cell_w
                    bw = (w - x0) if c_end == gw else (c_end - c) * cell_w
                    if bw > 0:
                        clear_flag = 1 if len(rois) == 0 else 0
                        rois.append(
                            f"addroi=x={x0}:y={y0}:w={bw}:h={bh}:qoffset={qoff_avg:.5f}:clear={clear_flag}"
                        )
                    c = c_end
            return rois

        all_same = t_frames == 1 or bool((grid_qg - grid_qg[0:1]).abs().max().item() < 1e-5)
        if all_same:
            rois_0 = build_frame_rois(grid_qg[0])
            return ["-vf", ",".join(rois_0)] if rois_0 else []

        split_pads = "".join(f"[s{i}]" for i in range(t_frames))
        branches: List[str] = [f"split={t_frames}{split_pads}"]
        out_pads = "".join(f"[v{i}]" for i in range(t_frames))
        any_roi = False
        for i in range(t_frames):
            f_rois = build_frame_rois(grid_qg[i])
            if f_rois:
                any_roi = True
                chain = ",".join([f"select='eq(n,{i})'"] + f_rois)
            else:
                chain = f"select='eq(n,{i})'"
            branches.append(f"[s{i}]{chain}[v{i}]")

        if not any_roi:
            return []
        branches.append(f"{out_pads}concat=n={t_frames}:v=1:a=0,settb=1/25,setpts=N")
        return ["-vf", ";".join(branches)]

    @torch.no_grad()
    def encode_decode_x265_clip(
        self,
        clip: torch.Tensor,
        qp: int,
        ctu_qp_offsets: Optional[torch.Tensor] = None,
        return_packet_sizes: bool = False,
        family_id: Union[int, torch.Tensor] = 0,
        gop_len: Optional[int] = None,
        bframes: Optional[int] = None,
        return_stream_info: bool = False,
    ) -> Tuple[Any, ...]:
        """
        Encodes a single clip `[T, 3, H, W]` in `[0, 1]` using REAL `libx265` (`family_id=0`)
        or `libx264` (`family_id=1`) via `ffmpeg` in standard `yuv420p` 4:2:0 format,
        applies full-resolution QG `addroi` offsets, and extracts exact per-frame entropy-coded
        packet bytes and frame types (`I`/`P`/`B`) in presentation order via `ffprobe`.
        """
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            raise RuntimeError("Real codec evaluation requires both `ffmpeg` and `ffprobe` in PATH.")

        fam = int(family_id.item()) if isinstance(family_id, torch.Tensor) else int(family_id)
        if fam not in (0, 1):
            raise ValueError(f"Unsupported codec family_id={fam}; expected 0 (libx265) or 1 (libx264).")

        t, _, h, w = clip.shape
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(f"yuv420p encoding requires even frame dimensions, got {w}x{h}.")

        eff_gop = (
            int(gop_len)
            if (gop_len is not None and int(gop_len) > 0)
            else (int(self.gop_len) if (self.gop_len is not None and int(self.gop_len) > 0) else t)
        )
        req_bframes = self.bframes if bframes is None else int(bframes)
        eff_bframes = max(0, min(req_bframes, max(0, t - 1)))

        frames_np = (
            torch.clamp(clip, 0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy() * 255.0
        ).round().astype(np.uint8)

        vf_args = self._build_addroi_filter_from_mask_or_offsets(ctu_qp_offsets, h, w)

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_in = Path(tmpdir) / "in.rgb"
            bitstream = Path(tmpdir) / "out.mp4"
            raw_out = Path(tmpdir) / "out.rgb"
            frames_np.tofile(str(raw_in))

            # Offload filtergraphs > 4,000 chars to a script file on ALL platforms (Windows & Linux)
            vf_cli_args = vf_args
            used_filter_script = False
            if len(vf_args) == 2 and vf_args[0] == "-vf" and len(vf_args[1]) > 4000:
                vf_script = Path(tmpdir) / "vf_script.txt"
                vf_script.write_text(vf_args[1], encoding="utf-8")
                vf_cli_args = [detect_ffmpeg_filter_script_flag(), str(vf_script)]
                used_filter_script = True

            if fam == 0:
                encoder_name = "libx265"
                x265_params = (
                    f"crf={qp}:bframes={eff_bframes}:keyint={eff_gop}:min-keyint={eff_gop}:"
                    f"scenecut=0:aq-mode=2:log-level=none"
                )
                codec_args = [
                    "-c:v",
                    "libx265",
                    "-preset",
                    self.preset,
                    "-x265-params",
                    x265_params,
                ]
            else:
                encoder_name = "libx264"
                x264_params = (
                    f"bframes={eff_bframes}:keyint={eff_gop}:min-keyint={eff_gop}:"
                    f"scenecut=0:aq-mode=2"
                )
                codec_args = [
                    "-c:v",
                    "libx264",
                    "-preset",
                    self.preset,
                    "-crf",
                    str(qp),
                    "-x264-params",
                    x264_params,
                ]

            cmd_enc = (
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{w}x{h}",
                    "-r",
                    "25",
                    "-i",
                    str(raw_in),
                ]
                + vf_cli_args
                + codec_args
                + [
                    "-pix_fmt",
                    "yuv420p",
                    str(bitstream),
                ]
            )
            subprocess.run(cmd_enc, check=True)

            cmd_probe = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-show_packets",
                "-show_frames",
                "-show_entries",
                "stream=pix_fmt,codec_name:packet=pos,size,pts:frame=pict_type,pkt_size,pkt_pos,pts,best_effort_timestamp",
                "-of",
                "json",
                str(bitstream),
            ]
            probe_res = subprocess.run(cmd_probe, capture_output=True, text=True, check=True)
            probe_data = json.loads(probe_res.stdout)

            streams = probe_data.get("streams", [])
            stream_pix_fmt = str(streams[0].get("pix_fmt", "")) if streams else ""
            if stream_pix_fmt != "yuv420p":
                raise RuntimeError(
                    f"Codec F13 violation: expected yuv420p bitstream, got pix_fmt='{stream_pix_fmt}'"
                )

            combined = probe_data.get("packets_and_frames", [])
            probe_frames = probe_data.get(
                "frames", [row for row in combined if row.get("type") == "frame"]
            )
            packets = probe_data.get(
                "packets", [row for row in combined if row.get("type") == "packet"]
            )

            packet_by_pos: Dict[str, int] = {}
            packet_by_pts: Dict[int, Tuple[str, int]] = {}
            for packet in packets:
                position = str(packet.get("pos", "")).strip()
                size = int(packet.get("size", 0))
                if not position or position in packet_by_pos or size <= 0:
                    raise RuntimeError("ffprobe returned invalid or duplicate video packets")
                packet_by_pos[position] = size
                if packet.get("pts") is not None:
                    packet_by_pts[int(packet["pts"])] = (position, size)

            if len(packet_by_pos) != t:
                raise RuntimeError(
                    f"ffprobe returned {len(packet_by_pos)} video packets, expected {t}"
                )
            if len(probe_frames) != t:
                raise RuntimeError(
                    f"ffprobe returned {len(probe_frames)} frames, expected {t}"
                )

            # Ensure frames are strictly ordered by presentation timestamp (PTS) under B-frame reordering
            def _frame_pts_key(idx_and_frame: Tuple[int, Dict[str, Any]]) -> Tuple[int, int]:
                f_idx, pf = idx_and_frame
                pts_raw = pf.get("pts", pf.get("best_effort_timestamp"))
                return (int(pts_raw) if pts_raw is not None else f_idx, f_idx)

            ordered_frames = [
                pf for _, pf in sorted(enumerate(probe_frames), key=_frame_pts_key)
            ]

            packet_sizes: List[int] = []
            matched_positions = set()
            frame_bpps_list: List[float] = []
            frame_types: List[str] = []

            for pf_idx, pf in enumerate(ordered_frames):
                if "pict_type" not in pf:
                    raise RuntimeError(f"ffprobe frame {pf_idx} missing pict_type: {pf}")
                position = str(pf.get("pkt_pos", "")).strip()
                pkt_bytes: Optional[int] = None
                if position and position in packet_by_pos and position not in matched_positions:
                    pkt_bytes = packet_by_pos[position]
                else:
                    pts_raw = pf.get("pts", pf.get("best_effort_timestamp"))
                    if pts_raw is not None and int(pts_raw) in packet_by_pts:
                        cand_pos, cand_size = packet_by_pts[int(pts_raw)]
                        if cand_pos not in matched_positions:
                            position = cand_pos
                            pkt_bytes = cand_size
                if pkt_bytes is None or not position:
                    raise RuntimeError(
                        f"ffprobe frame {pf_idx} could not be uniquely matched to a video packet"
                    )
                matched_positions.add(position)
                if "pkt_size" in pf and int(pf["pkt_size"]) != pkt_bytes:
                    raise RuntimeError("ffprobe packet size differs from matched frame pkt_size")
                if pkt_bytes <= 0:
                    raise RuntimeError(
                        f"ffprobe frame {pf_idx} returned non-positive pkt_size={pkt_bytes}"
                    )
                pict_t = str(pf["pict_type"]).strip()
                if pict_t not in ("I", "P", "B"):
                    raise RuntimeError(
                        f"ffprobe frame {pf_idx} returned invalid pict_type={pict_t}"
                    )
                packet_sizes.append(pkt_bytes)
                frame_bpps_list.append((pkt_bytes * 8.0) / float(h * w))
                frame_types.append(pict_t)

            if sum(packet_sizes) != sum(packet_by_pos.values()):
                raise RuntimeError("Matched frame packet byte sum differs from total stream packet bytes")

            cmd_dec = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(bitstream),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                str(raw_out),
            ]
            subprocess.run(cmd_dec, check=True)

            frame_bpps = torch.tensor(
                frame_bpps_list, device=clip.device, dtype=clip.dtype
            )
            total_bpp = float(frame_bpps.mean().item())
            dec_np = np.fromfile(str(raw_out), dtype=np.uint8).reshape(t, h, w, 3)
            recon_clip = (
                torch.from_numpy(dec_np).to(device=clip.device, dtype=clip.dtype) / 255.0
            )
            recon_clip = recon_clip.permute(0, 3, 1, 2).contiguous()

            stream_meta = {
                "pix_fmt": stream_pix_fmt,
                "encoder": encoder_name,
                "family_id": fam,
                "gop_len": eff_gop,
                "bframes": eff_bframes,
                "used_filter_script": used_filter_script,
            }
            if return_packet_sizes and return_stream_info:
                return recon_clip, total_bpp, frame_bpps, frame_types, packet_sizes, stream_meta
            if return_packet_sizes:
                return recon_clip, total_bpp, frame_bpps, frame_types, packet_sizes
            if return_stream_info:
                return recon_clip, total_bpp, frame_bpps, frame_types, stream_meta
            return recon_clip, total_bpp, frame_bpps, frame_types

    encode_decode_codec_clip = encode_decode_x265_clip

    def _build_addroi_filter_from_mask_or_offsets(
        self, ctu_qp_offsets: Optional[torch.Tensor], h: int, w: int
    ) -> List[str]:
        return self._build_addroi_filter_from_offsets(ctu_qp_offsets, h, w)

    @torch.no_grad()
    def evaluate_real_x265(
        self,
        video_edit: torch.Tensor,
        qp: int,
        ctu_qp_offsets: Optional[torch.Tensor] = None,
        family_id: Union[int, torch.Tensor] = 0,
        gop_len: Optional[int] = None,
        bframes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Strict real-codec evaluation pathway (`libx265` for `family_id=0`, `libx264` for `family_id=1`).
        Encodes in `yuv420p` with full-resolution QG `addroi` offsets and measures actual stream packet bytes.
        """
        b, t, _, h, w = video_edit.shape
        recons = []
        frame_rate_rows = []
        all_types: List[List[str]] = []
        all_packet_sizes: List[List[int]] = []
        stream_metas: List[Dict[str, Any]] = []

        for idx in range(b):
            off_i = ctu_qp_offsets[idx] if ctu_qp_offsets is not None else None
            if isinstance(family_id, torch.Tensor):
                fam_flat = family_id.reshape(-1)
                fam_i = int(fam_flat[min(idx, fam_flat.numel() - 1)].item())
            else:
                fam_i = int(family_id)
            recon_i, _, f_bpps_i, f_types_i, sizes_i, meta_i = self.encode_decode_x265_clip(
                video_edit[idx].detach(),
                qp,
                off_i,
                return_packet_sizes=True,
                family_id=fam_i,
                gop_len=gop_len,
                bframes=bframes,
                return_stream_info=True,
            )
            recons.append(recon_i)
            frame_rate_rows.append(f_bpps_i)
            all_types.append(f_types_i)
            all_packet_sizes.append(sizes_i)
            stream_metas.append(meta_i)

        video_recon = torch.stack(recons, dim=0)
        rate_per_frame = torch.stack(frame_rate_rows, dim=0)
        i_rates: List[float] = []
        inter_rates: List[float] = []
        only_p_rates: List[float] = []
        b_rates: List[float] = []
        for b_i, f_types in enumerate(all_types):
            for t_i, ft in enumerate(f_types):
                val = float(rate_per_frame[b_i, t_i].item())
                if ft == "I":
                    i_rates.append(val)
                elif ft == "P":
                    inter_rates.append(val)
                    only_p_rates.append(val)
                elif ft == "B":
                    inter_rates.append(val)
                    b_rates.append(val)

        meta0 = stream_metas[0] if stream_metas else {
            "pix_fmt": "yuv420p",
            "encoder": "libx265",
            "family_id": 0,
            "gop_len": gop_len or t,
            "bframes": 0 if bframes is None else int(bframes),
        }
        return {
            "video_recon": video_recon,
            "rate_per_frame": rate_per_frame,
            "total_bpp": float(rate_per_frame.mean().item()),
            "i_frame_bpp": float(np.mean(i_rates)) if i_rates else float("nan"),
            "p_frame_bpp": float(np.mean(inter_rates)) if inter_rates else float("nan"),
            "only_p_frame_bpp": float(np.mean(only_p_rates)) if only_p_rates else float("nan"),
            "b_frame_bpp": float(np.mean(b_rates)) if b_rates else float("nan"),
            "frame_types": all_types[0] if all_types else [],
            "packet_sizes": all_packet_sizes[0] if b == 1 else all_packet_sizes,
            "height": h,
            "width": w,
            "pix_fmt": meta0["pix_fmt"],
            "encoder": meta0["encoder"],
            "family_id": meta0["family_id"],
            "gop_len": meta0["gop_len"],
            "bframes": meta0["bframes"],
            "bitrate_accounting": "mp4_video_packet_payload_excluding_container_and_global_headers",
            "rate_control_mode": (
                f"encoder={meta0['encoder']}:crf={qp}:aq-mode=2:"
                f"bframes={meta0['bframes']}:keyint={meta0['gop_len']}:pix_fmt={meta0['pix_fmt']}"
            ),
        }

    evaluate_real_codec = evaluate_real_x265

    def forward(
        self,
        video_edit: torch.Tensor,
        severity: torch.Tensor,
        ctu_qp_offsets: Optional[torch.Tensor] = None,
        family_id: Optional[Union[int, torch.Tensor]] = None,
        gop_len: Optional[int] = None,
        bframes: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        proxy_out = self.proxy(video_edit, severity, ctu_qp_offsets, family_id=family_id)
        video_proxy = proxy_out["video_recon"]

        if not self.use_real_x265:
            return proxy_out

        b = video_edit.shape[0]

        def _get_fam(idx: int) -> int:
            if family_id is None:
                return 0
            if isinstance(family_id, int):
                return int(family_id)
            fam_flat = family_id.detach().reshape(-1)
            return int(fam_flat[min(idx, fam_flat.numel() - 1)].item())

        def _encode_one(idx: int):
            qp_val = int(round(22.0 + 25.0 * float(severity[idx].item())))
            off_i = ctu_qp_offsets[idx].detach() if ctu_qp_offsets is not None else None
            fam_i = _get_fam(idx)
            recon_i, bpp_i, f_bpps_i, _ = self.encode_decode_x265_clip(
                video_edit[idx].detach(),
                qp_val,
                off_i,
                family_id=fam_i,
                gop_len=gop_len,
                bframes=bframes,
            )
            return recon_i, bpp_i, f_bpps_i

        if b > 1:
            workers = min(b, max(2, os.cpu_count() or 4), 8)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_encode_one, range(b)))
        else:
            results = [_encode_one(0)]

        real_recons = [r[0] for r in results]
        real_bpps = [r[1] for r in results]
        real_frame_bpps = [r[2] for r in results]

        video_real = torch.stack(real_recons, dim=0)
        frame_bpps_real = torch.stack(real_frame_bpps, dim=0)
        out_dtype = proxy_out["rate_proxy_bpp"].dtype
        with torch.amp.autocast(device_type=video_edit.device.type, enabled=False):
            video_real = video_real.to(dtype=proxy_out["video_recon"].dtype)
            frame_bpps_real = frame_bpps_real.to(dtype=out_dtype)
            video_ste = video_proxy + (video_real - video_proxy).detach()
            rate_real = torch.tensor(real_bpps, device=video_edit.device, dtype=out_dtype)
            rate_ste = proxy_out["rate_proxy_bpp"] + (rate_real - proxy_out["rate_proxy_bpp"]).detach()
            rate_per_frame_ste = (
                proxy_out["rate_per_frame"]
                + (frame_bpps_real - proxy_out["rate_per_frame"]).detach()
            )
            t = video_edit.shape[1]
            i_frame_bpp_ste = rate_per_frame_ste[:, 0]
            p_frame_bpp_ste = (
                rate_per_frame_ste[:, 1:].mean(dim=1)
                if t > 1
                else torch.zeros_like(i_frame_bpp_ste)
            )

        return {
            "video_recon": video_ste,
            "rate_proxy_bpp": rate_ste,
            "rate_per_frame": rate_per_frame_ste,
            "i_frame_bpp_proxy": i_frame_bpp_ste,
            "p_frame_bpp_proxy": p_frame_bpp_ste,
        }
