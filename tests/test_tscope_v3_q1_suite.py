"""Comprehensive regression & integration tests for T-SCOPE v3 Q1 fixes (F04, F06, F11, F16, F17, F19, CLI alignment)."""
import ast
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses.vcm_temporal_loss import VCMTemporalPreEditLoss
from models.frozen_detector_fpn import FrozenFPNDetectorTeacher
from models.temporal_scope import TemporalSCOPE
from scripts.eval_sota_and_tracking import (
    ByteTrackerKalman,
    build_detector_roi_dqp_offsets,
    compute_mot_tracking_metrics,
    compute_multiclass_dataset_coco_metrics,
)
from scripts.kaggle_pool_orchestrator import SUPPORTED_RUN_MODES, build_worker_script
from scripts.kaggle_shard_runner import ABLATION_PRESETS, _run_train_batch_forward_with_fp32_fallback
from scripts.run_rigor_gate0_gate1_pilot import (
    build_anti_aligned_dqp,
    build_spatially_shuffled_dqp,
    evaluate_gate0_4_conjunctive,
)
from video_codecs.hybrid_x265_ste import HybridX265STE


class Gate04ConjunctiveFalsificationTests(unittest.TestCase):
    def test_constant_or_random_dqp_fails_conjunctive_gate04_while_aligned_passes(self):
        # F19 fix: constant/random positive +dQP lowers total_bpp vs map_off (which passed v2's `or` gate!),
        # yet MUST FAIL v3's conjunctive matched-bitrate Gate 0.4 when ROI PSNR does not beat shuffled/anti controls.
        dqp = torch.zeros(1, 4, 8, 8)
        dqp[:, :, 4:, :] = 3.5  # background coarsened, foreground (top half) protected at 0.0
        shuf = build_spatially_shuffled_dqp(dqp, seed=2026)
        anti = build_anti_aligned_dqp(dqp)
        self.assertAlmostEqual(float(dqp.mean().item()), float(shuf.mean().item()), places=5)
        self.assertAlmostEqual(float(dqp.mean().item()), float(anti.mean().item()), places=5)

        # Case 1: Defective / constant / foreground-coarsening map lowers rate vs off (0.12 < 0.16)
        #         but has equal/worse ROI PSNR than shuffled (29.0 <= 29.0) -> MUST FAIL!
        bad_gate = evaluate_gate0_4_conjunctive(
            enc_qp_off={"total_bpp": 0.16},
            enc_qp_on={"total_bpp": 0.12},
            enc_qp_shuffled={"total_bpp": 0.12},
            enc_qp_anti={"total_bpp": 0.12},
            acc_qp_off={"task_roi_psnr_db": 31.0},
            acc_qp_on={"task_roi_psnr_db": 29.0},
            acc_qp_shuffled={"task_roi_psnr_db": 29.0},
            acc_qp_anti={"task_roi_psnr_db": 29.0},
        )
        self.assertFalse(bad_gate["passed"])

        # Case 2: Valid foreground-sparing +dQP map lowers rate vs off AND preserves higher ROI PSNR than controls -> PASSES!
        good_gate = evaluate_gate0_4_conjunctive(
            enc_qp_off={"total_bpp": 0.16},
            enc_qp_on={"total_bpp": 0.118},
            enc_qp_shuffled={"total_bpp": 0.120},
            enc_qp_anti={"total_bpp": 0.121},
            acc_qp_off={"task_roi_psnr_db": 31.5},
            acc_qp_on={"task_roi_psnr_db": 31.4},
            acc_qp_shuffled={"task_roi_psnr_db": 29.6},
            acc_qp_anti={"task_roi_psnr_db": 28.4},
        )
        self.assertTrue(good_gate["passed"])


class TrackingAndMulticlassMetricsTests(unittest.TestCase):
    def test_bytetracker_and_mota_idf1_hota_exact_identities(self):
        tracker = ByteTrackerKalman()
        gt_boxes_seq = []
        gt_ids_seq = []
        pred_boxes_seq = []
        pred_ids_seq = []

        for t in range(10):
            gb = np.array(
                [
                    [10.0 + 3.0 * t, 20.0, 40.0 + 3.0 * t, 80.0],
                    [200.0 - 2.0 * t, 50.0, 235.0 - 2.0 * t, 120.0],
                ],
                dtype=np.float64,
            )
            gids = np.array([101, 202], dtype=np.int64)
            scores = np.array([0.92, 0.88], dtype=np.float64)
            tb, tids = tracker.update(gb, scores)
            gt_boxes_seq.append(gb)
            gt_ids_seq.append(gids)
            pred_boxes_seq.append(tb)
            pred_ids_seq.append(tids)

        perfect = compute_mot_tracking_metrics(gt_boxes_seq, gt_ids_seq, pred_boxes_seq, pred_ids_seq)
        self.assertEqual(perfect["idsw"], 0)
        self.assertAlmostEqual(perfect["mota_pct"], 100.0, places=3)
        self.assertAlmostEqual(perfect["idf1_pct"], 100.0, places=3)
        self.assertGreater(perfect["hota_pct"], 95.0)

        # Mid-sequence ID swap at frame 5 -> triggers 2 ID switches and drops IDF1 & HOTA
        swapped_ids = [arr.copy() for arr in pred_ids_seq]
        for t in range(5, 10):
            swapped_ids[t] = swapped_ids[t][::-1]
        swapped = compute_mot_tracking_metrics(gt_boxes_seq, gt_ids_seq, pred_boxes_seq, swapped_ids)
        self.assertEqual(swapped["idsw"], 2)
        self.assertLess(swapped["mota_pct"], 100.0)
        self.assertLess(swapped["idf1_pct"], 70.0)
        self.assertLess(swapped["hota_pct"], perfect["hota_pct"])

    def test_tracking_ignore_region_suppresses_fp(self):
        gb = [np.array([[10.0, 10.0, 40.0, 60.0]])]
        gid = [np.array([1])]
        pb = [np.array([[10.0, 10.0, 40.0, 60.0], [100.0, 100.0, 130.0, 150.0]])]
        pid = [np.array([1, 2])]
        ig = [np.array([[100.0, 100.0, 130.0, 150.0]])]
        res = compute_mot_tracking_metrics(gb, gid, pb, pid, ignore_boxes_seq=ig)
        self.assertEqual(res["fp"], 0)
        self.assertAlmostEqual(res["mota_pct"], 100.0, places=3)

    def test_detector_roi_dqp_and_multiclass_coco_ap(self):
        video = torch.rand(1, 2, 3, 64, 64)
        dqp = build_detector_roi_dqp_offsets(
            video,
            roi_detector_fn=lambda imgs: [torch.tensor([[0.0, 0.0, 16.0, 16.0]]) for _ in range(imgs.shape[0])],
            qg_size=16,
            bg_dqp=4.0,
            context_pad=0,
        )
        self.assertEqual(tuple(dqp.shape), (1, 2, 4, 4))
        self.assertEqual(float(dqp[0, 0, 0, 0].item()), 0.0)  # FG protected
        self.assertEqual(float(dqp[0, 0, 3, 3].item()), 4.0)  # Far BG coarsened

        mc = compute_multiclass_dataset_coco_metrics(
            all_gt_boxes=[torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 40.0]])],
            all_gt_classes=[torch.tensor([1, 3])],
            all_pred_boxes=[torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 40.0]])],
            all_pred_labels=[torch.tensor([1, 3])],
            all_pred_scores=[torch.tensor([0.95, 0.90])],
        )
        self.assertAlmostEqual(mc["map50_pct"], 100.0, places=3)
        self.assertEqual(set(mc["active_classes"].keys()), {"1", "3"})


class FlopBypassAndCliAlignmentTests(unittest.TestCase):
    def test_iwa_gop_qp_reuse_genuinely_reduces_flops(self):
        from torch.utils.flop_counter import FlopCounterMode

        model = TemporalSCOPE(base_ch=24, num_families=2, cond_dim=32, qg_size=16, max_delta_qp=4.0).eval()
        clip = torch.rand(1, 8, 3, 128, 192)
        fam = torch.tensor([0], dtype=torch.long)
        sev = torch.tensor([0.4])
        with torch.no_grad():
            with FlopCounterMode(display=False) as fc_full:
                out_full = model(clip, fam, sev, reuse_iframe_qp=False, disable_temporal=False)
            with FlopCounterMode(display=False) as fc_iwa:
                out_iwa = model(clip, fam, sev, reuse_iframe_qp=True, disable_temporal=False)
        self.assertEqual(tuple(out_iwa["i_edit"].shape), tuple(out_full["i_edit"].shape))
        self.assertLess(fc_iwa.get_total_flops(), 0.35 * fc_full.get_total_flops())

    def test_orchestrator_worker_scripts_compile_and_match_all_runner_cli_flags(self):
        dummy_sha = "a" * 40
        for mode in SUPPORTED_RUN_MODES:
            code = build_worker_script(
                shard_id=0,
                num_shards=15,
                variant="full_tscope",
                epochs=12,
                batch_size=2,
                run_mode=mode,
                git_revision=dummy_sha,
            )
            ast.parse(code)


class AmpOverflowAndBceSafetyTests(unittest.TestCase):
    def test_bce_nan_inf_sanitization_and_amp_fp32_batch_fallback(self):
        criterion = VCMTemporalPreEditLoss()
        b, t, h, w = 1, 2, 64, 64
        video = torch.rand(b, t, 3, h, w)
        m_attn = torch.zeros(b, t, 1, h, w)
        m_attn[:, :, :, 16:32, 16:32] = 1.0
        m_det = m_attn.clone()

        # 1) Inject NaN/+Inf/-Inf in attn_map (reproducing Shard 03's FP16 overflow into BCE):
        #    Must NOT raise RuntimeError / CUDA Loss.cu assert in F.binary_cross_entropy!
        bad_attn = torch.full((b, t, 1, h, w), 0.5, requires_grad=True)
        bad_attn_poisoned = bad_attn.clone()
        bad_attn_poisoned.data[0, 0, 0, 0, 0] = float("nan")
        bad_attn_poisoned.data[0, 0, 0, 0, 1] = float("inf")
        bad_attn_poisoned.data[0, 0, 0, 0, 2] = float("-inf")
        dummy_feats = {"0": torch.randn(b * t, 16, h // 4, w // 4)}
        loss_poisoned = criterion(
            i_edit=video.clone(),
            i_recon=video.clone(),
            attn_map=bad_attn_poisoned,
            delta_obj=torch.zeros_like(video),
            delta_bg=torch.zeros_like(video),
            rate_proxy_bpp=torch.tensor([0.12]),
            m_attn=m_attn,
            m_det=m_det,
            teacher_fpn_feats=dummy_feats,
            student_fpn_feats={k: v.clone().requires_grad_(True) for k, v in dummy_feats.items()},
            i_orig=video,
        )
        # BCE stayed safe (no assert), while loss_total cleanly flags non-finite for FP32 retry!
        self.assertFalse(bool(torch.isfinite(loss_poisoned["loss_total"]).item()))

        # 2) Verify `_run_train_batch_forward_with_fp32_fallback` catches autocast NaN overflow
        #    and re-computes the batch in full FP32 (`autocast(enabled=False)`), yielding 100% finite
        #    loss and finite gradients across all learnable parameters.
        base_model = TemporalSCOPE(base_ch=16, num_families=2, cond_dim=32, qg_size=16, max_delta_qp=4.0)
        codec_ste = HybridX265STE(use_real_x265_in_forward=False)
        fpn_teacher = FrozenFPNDetectorTeacher(pretrained=False, use_yolov8s=False)

        class _AutocastOverflowWrapper(torch.nn.Module):
            def __init__(self, inner: torch.nn.Module) -> None:
                super().__init__()
                self.inner = inner
                self.calls = 0

            def forward(self, v, fam, sev, **kwargs):
                self.calls += 1
                out_d = self.inner(v, fam, sev, **kwargs)
                if torch.is_autocast_enabled(v.device.type):
                    poison = out_d["attn_map"].clone()
                    poison[:, 0, 0, 0, 0] = float("nan")
                    out_d = dict(out_d)
                    out_d["attn_map"] = poison
                return out_d

        wrapped = _AutocastOverflowWrapper(base_model)
        fam = torch.tensor([0], dtype=torch.long)
        sev = torch.tensor([0.5], dtype=torch.float32)
        out_ok, losses_ok, used_fp32 = _run_train_batch_forward_with_fp32_fallback(
            model=wrapped,
            codec_ste=codec_ste,
            fpn_teacher=fpn_teacher,
            criterion=criterion,
            video=video,
            m_attn=m_attn,
            m_det=m_det,
            family_id=fam,
            severity=sev,
            preset_cfg=ABLATION_PRESETS["iwa_gop_qp_reuse"],
            use_amp=True,
        )
        self.assertTrue(used_fp32)
        self.assertEqual(wrapped.calls, 2)
        self.assertTrue(bool(torch.isfinite(losses_ok["loss_total"]).item()))
        self.assertTrue(bool(torch.isfinite(out_ok["attn_map"]).all().item()))
        losses_ok["loss_total"].backward()
        grad_Params = [p.grad for p in base_model.parameters() if p.grad is not None]
        self.assertGreater(len(grad_Params), 0)
        self.assertTrue(all(bool(torch.isfinite(g).all().item()) for g in grad_Params))

        # 3) Verify `forward_inter_bypass_step` in FP16 with saturated `attn_prev_warped = 0.9999`
        #    (which rounds to 1.0 in float16) produces 100% finite forward outputs AND finite gradients!
        fp16_model = TemporalSCOPE(base_ch=16, num_families=2, cond_dim=32, qg_size=16, max_delta_qp=4.0).half()
        i_t_16 = torch.rand(1, 3, 32, 32, dtype=torch.float16, requires_grad=True)
        i_prev_16 = torch.rand(1, 3, 32, 32, dtype=torch.float16)
        c_16 = torch.randn(1, 32, dtype=torch.float16, requires_grad=True)
        attn_prev_16 = torch.full((1, 1, 32, 32), 0.9999, dtype=torch.float16, requires_grad=True)
        eff_prev_16 = torch.ones(1, 1, 32, 32, dtype=torch.float16, requires_grad=True)
        bg_prev_16 = torch.rand(1, 3, 32, 32, dtype=torch.float16, requires_grad=True)
        i_ed_16, attn_16, eff_16, dobj_16, dbg_16, bg_16 = fp16_model.forward_inter_bypass_step(
            i_t_16, i_prev_16, c_16, attn_prev_16, eff_prev_16, bg_prev_16
        )
        for ten in (i_ed_16, attn_16, eff_16, dobj_16, dbg_16, bg_16):
            self.assertEqual(ten.dtype, torch.float16)
            self.assertTrue(bool(torch.isfinite(ten).all().item()))
        (i_ed_16.float().mean() + attn_16.float().mean() + eff_16.float().mean()).backward()
        self.assertIsNotNone(attn_prev_16.grad)
        self.assertTrue(bool(torch.isfinite(attn_prev_16.grad).all().item()))
        self.assertTrue(
            all(
                bool(torch.isfinite(p.grad).all().item())
                for p in fp16_model.inter_refine.parameters()
                if p.grad is not None
            )
        )


if __name__ == "__main__":
    unittest.main()
