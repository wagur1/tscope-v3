"""Focused Stage 2 regression tests for T-SCOPE v2 architecture, proxy, and loss."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from losses.vcm_temporal_loss import VCMTemporalPreEditLoss
from models.temporal_scope import TemporalSCOPE, estimate_global_affine_irls
from video_codecs.hybrid_x265_ste import (
    DifferentiableVideoProxy,
    estimate_motion_compensated_pred,
)


def _smooth_textured_frame(height: int = 64, width: int = 64) -> torch.Tensor:
    yy = torch.linspace(-1.0, 1.0, height).view(1, 1, height, 1)
    xx = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width)
    ch0 = 0.5 + 0.35 * torch.sin(5.0 * xx + 2.0 * yy) * torch.cos(3.0 * yy)
    ch1 = 0.5 + 0.35 * torch.cos(4.0 * xx - 4.0 * yy)
    ch2 = 0.4 + 0.30 * torch.sin(7.0 * (xx * xx + yy * yy))
    return torch.clamp(torch.cat([ch0, ch1, ch2], dim=1), 0.0, 1.0)


def _warp_frame(frame: torch.Tensor, shift_x_px: float, shift_y_px: float) -> torch.Tensor:
    _, _, h, w = frame.shape
    theta = torch.tensor(
        [[[1.0, 0.0, -2.0 * shift_x_px / float(w)], [0.0, 1.0, -2.0 * shift_y_px / float(h)]]],
        dtype=frame.dtype,
    )
    grid = F.affine_grid(theta, size=frame.shape, align_corners=False)
    return F.grid_sample(frame, grid, mode="bilinear", padding_mode="border", align_corners=False)


class TemporalScopeV2Tests(unittest.TestCase):
    def test_irls_affine_static_snap_and_panning_compensation(self):
        frame = _smooth_textured_frame(64, 64)
        theta_s, is_static_s = estimate_global_affine_irls(frame, frame)
        self.assertTrue(bool(is_static_s.all()))
        eye = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        torch.testing.assert_close(theta_s, eye, rtol=0.0, atol=1e-6)

        shifted = _warp_frame(frame, shift_x_px=4.0, shift_y_px=-2.0)
        theta_m, is_static_m = estimate_global_affine_irls(frame, shifted)
        self.assertFalse(bool(is_static_m.any()))
        grid = F.affine_grid(theta_m, size=frame.shape, align_corners=False)
        warped = F.grid_sample(frame, grid, mode="bilinear", padding_mode="border", align_corners=False)
        raw_l1 = (shifted[:, :, 6:-6, 6:-6] - frame[:, :, 6:-6, 6:-6]).abs().mean().item()
        mc_l1 = (shifted[:, :, 6:-6, 6:-6] - warped[:, :, 6:-6, 6:-6]).abs().mean().item()
        self.assertLess(mc_l1, 0.25 * raw_l1)

    def test_eff_attn_gradient_flow_and_max_obj_res_severity_scaling(self):
        model = TemporalSCOPE(base_ch=16, cond_dim=16)
        self.assertAlmostEqual(model.max_obj_res, 0.025)

        for target_attn in (0.1, 0.5, 0.9):
            logits = torch.logit(torch.tensor(target_attn)).requires_grad_(True)
            attn_t = torch.sigmoid(logits)
            u_sat = torch.clamp((attn_t - 0.12) / 0.46, 0.0, 1.0)
            eff_hard = u_sat * u_sat * (3.0 - 2.0 * u_sat)
            eff_proxy = torch.sigmoid(2.0 * (logits + 0.65))
            eff_attn = eff_proxy + (eff_hard - eff_proxy).detach()
            eff_attn.backward()
            self.assertIsNotNone(logits.grad)
            self.assertGreater(float(logits.grad.item()), 0.0)

        clip = _smooth_textured_frame(32, 32).unsqueeze(1).expand(1, 2, 3, 32, 32).clone()
        fam = torch.zeros(1, dtype=torch.long)
        low_sev = model(clip, fam, torch.tensor([0.0]))
        high_sev = model(clip, fam, torch.tensor([1.0]))
        self.assertLessEqual(float(low_sev["delta_obj"].abs().max().item()), 0.025 * 0.30 + 1e-6)
        self.assertLessEqual(float(high_sev["delta_obj"].abs().max().item()), 0.025 + 1e-6)

        with torch.no_grad():
            model.attn_head.weight.zero_()
            model.attn_head.bias.zero_()
            model.attn_cond_bias.weight.zero_()
        for bias_val in (-2.2, 0.0, 2.2):
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                model.attn_cond_bias.bias.fill_(bias_val)
            out = model(clip, fam, torch.tensor([0.5]))
            out["i_edit"].pow(2).mean().backward()
            self.assertIsNotNone(model.attn_cond_bias.bias.grad)
            self.assertTrue(torch.isfinite(model.attn_cond_bias.bias.grad).all())
            self.assertGreater(float(model.attn_cond_bias.bias.grad.abs().item()), 0.0)

    def test_nondecreasing_p_frame_dqp_with_fg_protection(self):
        model = TemporalSCOPE(base_ch=16, cond_dim=16, qg_size=16, max_delta_qp=4.0)
        attn = torch.zeros(1, 4, 1, 64, 64)
        attn[:, :, :, :16, :16] = 1.0  # Upper-left QG block is pure foreground
        for t_idx, bg_val in enumerate((0.05, 0.14, 0.20, 0.23)):
            attn[:, t_idx, :, 16:, :] = bg_val
            attn[:, t_idx, :, :16, 16:] = bg_val

        dqp_temporal = model.compute_ctu_qp_offsets(attn, disable_temporal=False)
        dqp_spatial = model.compute_ctu_qp_offsets(attn, disable_temporal=True)

        self.assertEqual(float(dqp_temporal[:, :, 0, 0].abs().max().item()), 0.0)
        self.assertGreater(float(dqp_temporal[0, 3, 2, 2].item()), float(dqp_spatial[0, 3, 2, 2].item()))
        bg_traj = dqp_temporal[0, :, 2, 2].tolist()
        corner_traj = dqp_temporal[0, :, 3, 3].tolist()
        for t_idx in range(1, len(bg_traj)):
            self.assertGreaterEqual(bg_traj[t_idx], bg_traj[t_idx - 1] - 1e-4)
            self.assertGreaterEqual(corner_traj[t_idx], corner_traj[t_idx - 1] - 1e-4)

        dqp_gop = model.compute_ctu_qp_offsets(attn, reuse_iframe_across_gop=True)
        for t_idx in range(1, 4):
            torch.testing.assert_close(dqp_gop[:, t_idx], dqp_gop[:, 0])

    def test_disable_temporal_invariance_and_zero_smoke(self):
        model = TemporalSCOPE(base_ch=16, cond_dim=16)
        model.eval()
        f0 = _smooth_textured_frame(32, 32)
        f1 = _warp_frame(f0, shift_x_px=3.0, shift_y_px=1.0)
        clip = torch.stack([f0, f1], dim=1)
        fam = torch.zeros(1, dtype=torch.long)
        sev = torch.tensor([0.4])

        out_clip = model(clip, fam, sev, disable_temporal=True)
        self.assertIsNone(out_clip["aligned_prev_edit"])
        self.assertIsNone(out_clip["aligned_prev_orig"])
        self.assertIsNone(out_clip["warp_valid_mask"])

        out_0 = model(f0, fam, sev, disable_temporal=True)
        out_1 = model(f1, fam, sev, disable_temporal=True)
        torch.testing.assert_close(out_clip["i_edit"][:, 0], out_0["i_edit"], rtol=0.0, atol=1e-5)
        torch.testing.assert_close(out_clip["i_edit"][:, 1], out_1["i_edit"], rtol=0.0, atol=1e-5)
        torch.testing.assert_close(
            out_clip["ctu_qp_offsets"][:, 1], out_1["ctu_qp_offsets"], rtol=0.0, atol=1e-5
        )

        model.train()
        z_clip = torch.zeros(1, 2, 3, 32, 32)
        z_out = model(z_clip, fam, sev)
        z_loss = sum(v.mean() for v in z_out.values() if isinstance(v, torch.Tensor))
        z_loss.backward()
        for tensor in z_out.values():
            if isinstance(tensor, torch.Tensor):
                self.assertTrue(torch.isfinite(tensor).all())


class VideoProxyV2Tests(unittest.TestCase):
    def test_mc_proxy_panned_vs_uncorrelated_and_t1_handling(self):
        proxy = DifferentiableVideoProxy()
        f0 = _smooth_textured_frame(128, 128)
        f1_pan = _warp_frame(f0, shift_x_px=8.0, shift_y_px=-4.0)

        mc_pred = estimate_motion_compensated_pred(f1_pan, f0, block_size=16)
        raw_res = (f1_pan[:, :, 12:-12, 12:-12] - f0[:, :, 12:-12, 12:-12]).abs().mean().item()
        mc_res = (f1_pan[:, :, 12:-12, 12:-12] - mc_pred[:, :, 12:-12, 12:-12]).abs().mean().item()
        self.assertLess(mc_res, 0.25 * raw_res)

        sev = torch.tensor([0.3])
        pan_clip = torch.stack([f0, f1_pan], dim=1)
        out_pan = proxy(pan_clip, sev)

        gen = torch.Generator().manual_seed(7)
        uncorr = torch.rand_like(f0, generator=gen)
        uncorr_clip = torch.stack([f0, uncorr], dim=1)
        out_uncorr = proxy(uncorr_clip, sev)

        self.assertLess(float(out_pan["p_frame_bpp_proxy"].item()), float(out_uncorr["p_frame_bpp_proxy"].item()))
        self.assertGreater(float(out_pan["i_frame_bpp_proxy"].item()), float(out_pan["p_frame_bpp_proxy"].item()))

        for t_len in (1, 2):
            clip_32 = torch.rand(1, t_len, 3, 32, 32, generator=gen, requires_grad=True)
            out_32 = proxy(clip_32, torch.tensor([0.5]))
            if t_len == 1:
                self.assertEqual(float(out_32["p_frame_bpp_proxy"].item()), 0.0)
            (out_32["video_recon"].mean() + out_32["i_frame_bpp_proxy"].mean() + out_32["p_frame_bpp_proxy"].mean()).backward()
            self.assertIsNotNone(clip_32.grad)
            self.assertTrue(torch.isfinite(clip_32.grad).all())

    def test_fp16_qp22_bright_block_gradscaler_safety(self):
        proxy = DifferentiableVideoProxy()
        bright_fp16 = torch.ones(1, 2, 3, 32, 32, dtype=torch.float16, requires_grad=True)
        out = proxy(bright_fp16, torch.tensor([0.0]))
        self.assertEqual(out["rate_proxy_bpp"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(out["i_frame_bpp_proxy"]).all())
        self.assertTrue(torch.isfinite(out["p_frame_bpp_proxy"]).all())
        ((out["video_recon"].mean() + out["i_frame_bpp_proxy"].mean() + out["p_frame_bpp_proxy"].mean()) * 65536.0).backward()
        self.assertIsNotNone(bright_fp16.grad)
        self.assertTrue(torch.isfinite(bright_fp16.grad).all())


class VCMTemporalLossV2Tests(unittest.TestCase):
    def _base_tensors(self, h: int = 32, w: int = 32, with_fg: bool = True):
        gen = torch.Generator().manual_seed(99)
        i_orig = torch.rand(1, 2, 3, h, w, generator=gen)
        i_edit = i_orig.clone().detach().requires_grad_(True)
        i_recon = i_orig.clone().detach().requires_grad_(True)
        attn_map = torch.full((1, 2, 1, h, w), 0.15, requires_grad=True)
        delta_obj = torch.zeros_like(i_orig, requires_grad=True)
        delta_bg = torch.zeros_like(i_orig, requires_grad=True)
        rate_proxy = torch.tensor([0.2], requires_grad=True)
        i_bpp = torch.tensor([0.4], requires_grad=True)
        p_bpp = torch.tensor([0.05], requires_grad=True)
        m_attn = torch.zeros(1, 2, 1, h, w)
        m_det = torch.zeros(1, 2, 1, h, w)
        if with_fg:
            m_attn[:, :, :, 4:12, 4:12] = 1.0
            m_det[:, :, :, 2:14, 2:14] = 1.0
        t_feats = {"p2": torch.ones(2, 8, h // 4, w // 4)}
        s_feats = {"p2": torch.ones(2, 8, h // 4, w // 4, requires_grad=True)}
        return {
            "i_orig": i_orig,
            "i_edit": i_edit,
            "i_recon": i_recon,
            "attn_map": attn_map,
            "delta_obj": delta_obj,
            "delta_bg": delta_bg,
            "rate_proxy_bpp": rate_proxy,
            "i_frame_bpp_proxy": i_bpp,
            "p_frame_bpp_proxy": p_bpp,
            "m_attn": m_attn,
            "m_det": m_det,
            "teacher_fpn_feats": t_feats,
            "student_fpn_feats": s_feats,
        }

    def test_pbpp_weight_fg_preservation_saturation_and_dqp_penalties(self):
        crit = VCMTemporalPreEditLoss()
        kw = self._base_tensors(32, 32, with_fg=True)
        dqp_good = torch.zeros(1, 2, 2, 2, requires_grad=True)
        losses = crit(**kw, ctu_qp_offsets=dqp_good)
        losses["loss_total"].backward()

        self.assertGreater(float(kw["p_frame_bpp_proxy"].grad.item()), float(kw["i_frame_bpp_proxy"].grad.item()))

        with torch.no_grad():
            kw_fg = self._base_tensors(32, 32, with_fg=True)
            base_out = crit(**kw_fg, ctu_qp_offsets=torch.zeros(1, 2, 2, 2))
            kw_bad_pix = self._base_tensors(32, 32, with_fg=True)
            kw_bad_pix["i_edit"][:, :, :, 4:10, 4:10] += 0.2
            bad_pix_out = crit(**kw_bad_pix, ctu_qp_offsets=torch.zeros(1, 2, 2, 2))
            self.assertGreater(float(bad_pix_out["loss_task"].item()), float(base_out["loss_task"].item()))

            kw_high_attn = self._base_tensors(32, 32, with_fg=True)
            kw_high_attn["attn_map"][:, :, :, 2:14, 2:14] = 0.96
            high_attn_out = crit(**kw_high_attn, ctu_qp_offsets=torch.zeros(1, 2, 2, 2))
            self.assertLess(float(high_attn_out["loss_att"].item()), float(base_out["loss_att"].item()))

            dqp_fg_leak = torch.zeros(1, 2, 2, 2)
            dqp_fg_leak[:, :, 0, 0] = 3.0  # leaks onto FG block (top-left)
            dqp_fg_out = crit(**self._base_tensors(32, 32, with_fg=True), ctu_qp_offsets=dqp_fg_leak)
            self.assertGreater(float(dqp_fg_out["loss_reg"].item()), float(base_out["loss_reg"].item()))

            dqp_const_bg = torch.zeros(1, 2, 2, 2)
            dqp_const_bg[:, :, 1, 1] = 4.0
            dqp_decay_bg = torch.zeros(1, 2, 2, 2)
            dqp_decay_bg[:, 0, 1, 1] = 4.0
            dqp_decay_bg[:, 1, 1, 1] = 1.0
            const_out = crit(**self._base_tensors(32, 32, with_fg=True), ctu_qp_offsets=dqp_const_bg)
            decay_out = crit(**self._base_tensors(32, 32, with_fg=True), ctu_qp_offsets=dqp_decay_bg)
            self.assertGreater(float(decay_out["loss_temp"].item()), float(const_out["loss_temp"].item()))

    def test_empty_mask_zero_division_safety(self):
        crit = VCMTemporalPreEditLoss()
        kw = self._base_tensors(32, 32, with_fg=False)
        dqp = torch.zeros(1, 2, 2, 2, requires_grad=True)
        losses = crit(**kw, ctu_qp_offsets=dqp)
        for val in losses.values():
            self.assertTrue(torch.isfinite(val).all())
        losses["loss_total"].backward()
        for key in ("i_edit", "i_recon", "attn_map", "delta_obj", "delta_bg", "i_frame_bpp_proxy", "p_frame_bpp_proxy"):
            self.assertTrue(torch.isfinite(kw[key].grad).all())
        self.assertTrue(torch.isfinite(dqp.grad).all())

    def test_fp16_ctu_qp_gradscaler_65536_safety(self):
        crit = VCMTemporalPreEditLoss()
        kw = self._base_tensors(32, 32, with_fg=True)
        fp16_inputs = {
            "i_orig": kw["i_orig"],
            "i_edit": kw["i_edit"].detach().to(torch.float16).requires_grad_(True),
            "i_recon": kw["i_recon"].detach().to(torch.float16).requires_grad_(True),
            "attn_map": kw["attn_map"].detach().to(torch.float16).requires_grad_(True),
            "delta_obj": kw["delta_obj"].detach().to(torch.float16).requires_grad_(True),
            "delta_bg": kw["delta_bg"].detach().to(torch.float16).requires_grad_(True),
            "rate_proxy_bpp": kw["rate_proxy_bpp"].detach().to(torch.float16).requires_grad_(True),
            "i_frame_bpp_proxy": kw["i_frame_bpp_proxy"].detach().to(torch.float16).requires_grad_(True),
            "p_frame_bpp_proxy": kw["p_frame_bpp_proxy"].detach().to(torch.float16).requires_grad_(True),
            "m_attn": kw["m_attn"].to(torch.float16),
            "m_det": kw["m_det"].to(torch.float16),
            "teacher_fpn_feats": {k: v.to(torch.float16) for k, v in kw["teacher_fpn_feats"].items()},
            "student_fpn_feats": {k: v.detach().to(torch.float16).requires_grad_(True) for k, v in kw["student_fpn_feats"].items()},
        }
        dqp_fp16 = torch.full((1, 2, 2, 2), 2.0, dtype=torch.float16, requires_grad=True)
        losses = crit(**fp16_inputs, ctu_qp_offsets=dqp_fp16)
        self.assertEqual(losses["loss_total"].dtype, torch.float32)
        self.assertEqual(losses["loss_reg"].dtype, torch.float32)
        for val in losses.values():
            self.assertTrue(torch.isfinite(val).all())
        (losses["loss_total"] * 65536.0).backward()
        self.assertIsNotNone(dqp_fp16.grad)
        self.assertTrue(torch.isfinite(dqp_fp16.grad).all())
        self.assertTrue(torch.isfinite(fp16_inputs["attn_map"].grad).all())
        self.assertTrue(torch.isfinite(fp16_inputs["i_edit"].grad).all())


if __name__ == "__main__":
    unittest.main()
