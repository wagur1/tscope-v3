import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from models.frozen_detector_fpn import FrozenFPNDetectorTeacher as Teacher
from scripts.experiment_protocol import (
    CODEC_CONFIGURATION,
    CRFS,
    METRIC_POLICY,
    METHODS,
    SCHEMA_VERSION,
    TRAIN_SEQUENCES,
    VAL_SEQUENCES,
    VALIDATION_PROTOCOL,
    VARIANTS,
    restore_raw_detections,
    sha256_file,
    validate_checkpoint,
    validate_release_validation,
    verify_convergence_log,
)
from scripts.merge_shards import (
    _PreparedClipBootstrapCurve,
    average_compatible_states,
    bjontegaard_delta_rate,
    build_bdbr_summary,
    merge_checkpoints_soup,
    merge_eval_jsons_and_print_table,
    summarize_records,
)


def boxes(values=()):
    return torch.tensor(values, dtype=torch.float32).reshape(-1, 4)


class APTests(unittest.TestCase):
    def test_global_ap_not_mean_clip_ap(self):
        gt = [boxes([[0, 0, 10, 10]]), boxes([[0, 0, 10, 10]])]
        predicted = [gt[0], boxes([[20, 20, 30, 30]])]
        scores = [torch.tensor([0.5]), torch.tensor([0.9])]
        global_ap = Teacher.compute_dataset_level_ap(gt, predicted, scores)
        mean = sum(Teacher.compute_dataset_level_ap([g], [p], [s]) for g, p, s in zip(gt, predicted, scores)) / 2
        self.assertAlmostEqual(mean, 0.5)
        self.assertAlmostEqual(global_ap, 25.5 / 101)
        self.assertNotAlmostEqual(global_ap, mean)

    def test_empty_gt_false_positive_penalizes(self):
        gt = [boxes(), boxes([[0, 0, 10, 10]])]
        predicted = [boxes([[0, 0, 10, 10]]), gt[1]]
        self.assertAlmostEqual(Teacher.compute_dataset_level_ap(gt, predicted, [torch.tensor([0.9]), torch.tensor([0.5])]), 0.5)

    def test_ignore_policy(self):
        gt = [boxes(), boxes([[0, 0, 10, 10]])]
        predicted = [boxes([[0, 0, 10, 10]]), gt[1]]
        self.assertEqual(Teacher.compute_dataset_level_ap(gt, predicted, [torch.tensor([0.9]), torch.tensor([0.5])], [predicted[0], boxes()]), 1.0)

    def test_duplicate_detection_is_false_positive(self):
        gt = [boxes([[0, 0, 10, 10]]), boxes([[0, 0, 10, 10]])]
        predicted = [boxes([[0, 0, 10, 10], [0, 0, 10, 10]]), gt[1]]
        self.assertLess(Teacher.compute_dataset_level_ap(gt, predicted, [torch.tensor([0.9, 0.8]), torch.tensor([0.7])]), 1.0)

    def test_misaligned_frames_rejected(self):
        with self.assertRaises(ValueError):
            Teacher.compute_dataset_level_ap([boxes()], [], [])

    def test_score_box_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            Teacher.compute_dataset_level_ap([boxes()], [boxes([[0, 0, 10, 10]])], [torch.tensor([])])

    def test_ignore_count_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            Teacher.compute_dataset_level_ap([boxes()], [boxes()], [torch.tensor([])], [])

    def test_empty_serialization_shape_and_track_ids(self):
        raw = restore_raw_detections(
            {
                "gt_boxes": [[[1, 1, 5, 5]], []],
                "ignore_boxes": [[], []],
                "pred_boxes": [[], []],
                "pred_scores": [[], []],
                "gt_track_ids": [[7], []],
            },
            2,
        )
        self.assertEqual(tuple(raw["gt_boxes"][1].shape), (0, 4))
        self.assertEqual(raw["gt_track_ids"][0].tolist(), [7])


class SoupPchipAndBootstrapTests(unittest.TestCase):
    def test_validation_status_without_evidence_rejected(self):
        with self.assertRaises(ValueError):
            validate_release_validation({"status": "validated", "release_ready": True})

    def test_parameterized_convergence_log_identity(self):
        log = "Converged: no val improvement for 10 epochs. Stop at epoch 14.\n=== Shard 3/15 (full_tscope) finished successfully! ==="
        meta = verify_convergence_log(log, 3, "full_tscope", num_shards=15)
        self.assertFalse(meta["startup_visible"])
        self.assertEqual(meta["convergence_mode"], "early_stop")
        full_log = "  Converged: completed full 28-epoch cosine schedule (best_epoch=26).\n=== Shard 0/15 (full_tscope) finished successfully! ==="
        full_meta = verify_convergence_log(full_log, 0, "full_tscope", num_shards=15)
        self.assertEqual(full_meta["convergence_mode"], "full_schedule")
        with self.assertRaises(ValueError):
            verify_convergence_log(log, 4, "full_tscope", num_shards=15)

    def test_early_stopped_and_full_schedule_28ep_soup_merge_and_partial_rejection(self):
        from models.temporal_scope import TemporalSCOPE

        state = TemporalSCOPE(base_ch=48, num_families=2, cond_dim=64, qg_size=16, max_delta_qp=4.0).state_dict()
        base_cfg = {"epochs": 28, "min_epochs": 10, "patience": 6, "use_yolov8s": False, "model": {"num_families": 2}}

        # 1. Early-stopped at epoch 16 (best_epoch=10, patience=6) -> VALID
        hist_early = [{"epoch": ep, "loss_total": 3.0 - 0.01 * ep, "val_loss": 2.5 if ep == 10 else 2.6} for ep in range(1, 17)]
        ckpt_early = {
            "shard_id": 0, "num_shards": 10, "variant": "full_tscope", "state_dict": state,
            "history": hist_early, "best_epoch": 10, "best_val_loss": 2.5,
            "stopped_early": True, "partial": False, "epochs_run": 16,
            "training_configuration": base_cfg, "train_sequences": list(TRAIN_SEQUENCES), "val_sequences": list(VAL_SEQUENCES),
        }
        validate_checkpoint(ckpt_early, 0, "full_tscope", expected_num_shards=10)

        # 2. Full 28-epoch cosine schedule (best_epoch=26, stopped_early=False, partial=False) -> VALID
        hist_full = [{"epoch": ep, "loss_total": 3.0 - 0.01 * ep, "val_loss": 2.1 if ep == 26 else 2.4} for ep in range(1, 29)]
        ckpt_full = {
            "shard_id": 5, "num_shards": 10, "variant": "full_tscope", "state_dict": state,
            "history": hist_full, "best_epoch": 26, "best_val_loss": 2.1,
            "stopped_early": False, "partial": False, "epochs_run": 28,
            "training_configuration": base_cfg, "train_sequences": list(TRAIN_SEQUENCES), "val_sequences": list(VAL_SEQUENCES),
        }
        validate_checkpoint(ckpt_full, 5, "full_tscope", expected_num_shards=10)

        # 3. Partial checkpoint (partial=True) or truncated stopped_early=False (epochs_run=20 < 28) -> REJECTED
        ckpt_partial = {**ckpt_full, "partial": True}
        with self.assertRaisesRegex(ValueError, "partial"):
            validate_checkpoint(ckpt_partial, 5, "full_tscope", expected_num_shards=10)
        ckpt_trunc = {**ckpt_full, "history": hist_full[:20], "epochs_run": 20, "best_epoch": 18}
        ckpt_trunc["history"][17]["val_loss"] = 2.1
        with self.assertRaisesRegex(ValueError, "neither patience window nor full epoch schedule"):
            validate_checkpoint(ckpt_trunc, 5, "full_tscope", expected_num_shards=10)

        # 4. End-to-end merge_checkpoints_soup with v3 `t-scope-v3-shard-XX.log` filenames
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"schema_version": SCHEMA_VERSION, "expected_shards": 10, "shards": []}
            for sid, ckpt, log_line in (
                (0, ckpt_early, "  Converged: no val improvement for 6 epochs. Stop at epoch 16."),
                (5, ckpt_full, "  Converged: completed full 28-epoch cosine schedule (best_epoch=26)."),
            ):
                sdir = root / f"shard_{sid:02d}"
                sdir.mkdir()
                pt_p = sdir / f"tscope_full_tscope_shard{sid:02d}of10.pt"
                torch.save(ckpt, pt_p)
                lg_p = sdir / f"t-scope-v3-shard-{sid:02d}.log"
                lg_p.write_text(f"{log_line}\n=== Shard {sid}/10 (full_tscope) finished successfully! ===\n", encoding="utf-8")
                manifest["shards"].append({
                    "shard_id": sid, "kernel_slug": f"t-scope-v3-shard-{sid:02d}",
                    "expected_variant": "full_tscope", "status": "COMPLETE", "downloaded": True,
                    "directory": f"shard_{sid:02d}",
                    "files": [
                        {"path": pt_p.name, "bytes": pt_p.stat().st_size, "sha256": sha256_file(pt_p)},
                        {"path": lg_p.name, "bytes": lg_p.stat().st_size, "sha256": sha256_file(lg_p)},
                    ],
                })
            mpath = root / "collection_manifest.json"
            mpath.write_text(json.dumps(manifest), encoding="utf-8")
            out_soups = merge_checkpoints_soup(str(root / "soups"), "", str(mpath), variants=["full_tscope"])
            self.assertTrue(Path(out_soups["full_tscope"]).is_file())

    def test_float_average_and_integer_preservation(self):
        result = average_compatible_states(
            [{"weight": torch.tensor([1.0]), "count": torch.tensor(4)}, {"weight": torch.tensor([3.0]), "count": torch.tensor(4)}]
        )
        self.assertEqual(result["weight"].item(), 2.0)
        self.assertEqual(result["count"].dtype, torch.int64)

    def test_no_manifest_no_merge(self):
        with self.assertRaises(ValueError):
            merge_checkpoints_soup("unused", "unused.pt")
        with self.assertRaises(ValueError):
            merge_eval_jsons_and_print_table("unused")

    def test_bdbr_constant_savings_exact_under_pchip(self):
        self.assertAlmostEqual(
            bjontegaard_delta_rate([1, 2, 4, 8], [1, 2, 3, 4], [0.8, 1.6, 3.2, 6.4], [1, 2, 3, 4]),
            -20.0,
            places=6,
        )

    def test_pchip_eliminates_mot17_11_runge_overshoot(self):
        # Exact F14 pathological saturated curve from MOT17-11 where np.polyfit(deg=3) blew up to +213.93%!
        rate_anchor = [0.585, 0.285, 0.145, 0.072]
        acc_anchor = [62.49, 61.76, 60.04, 55.21]
        rate_tscope = [0.461, 0.231, 0.120, 0.061]  # 15-21% lower bpp at every CRF!
        acc_tscope = [62.34, 62.05, 60.20, 55.03]
        bdbr_pchip = bjontegaard_delta_rate(rate_anchor, acc_anchor, rate_tscope, acc_tscope)
        self.assertLess(bdbr_pchip, -10.0)
        self.assertGreater(bdbr_pchip, -30.0)

    def test_invalid_bdbr_rejected(self):
        for rate, accuracy in (([0, 2, 3, 4], [1, 2, 3, 4]), ([1, 2, 3, 4], [1, 1, 2, 3]), ([1, 2], [1, 2])):
            with self.assertRaises(ValueError):
                bjontegaard_delta_rate(rate, accuracy, rate, accuracy)

    def test_nonoverlap_is_unavailable(self):
        anchor = {str(crf): {"total_bpp": i + 1, "p_frame_bpp": i + 1, "gt_ap50_pct": i, "gt_ap50_95_pct": i} for i, crf in enumerate(CRFS)}
        test = copy.deepcopy(anchor)
        for row in test.values():
            row["gt_ap50_pct"] += 10
            row["gt_ap50_95_pct"] += 10
        result, _ = build_bdbr_summary(test, anchor)
        self.assertIsNone(result["bdbr_tot_gt_ap50_pct"])
        self.assertIn("bdbr_tot_gt_ap50_pct", result["bdbr_unavailable_reasons"])

    def test_vectorized_clip_bootstrap_matches_bruteforce_ap_identically(self):
        # Verify 0.000000 discrepancy between _PreparedClipBootstrapCurve and brute-force clip duplication!
        rng = np.random.default_rng(42)
        records_by_crf = {}
        for crf in CRFS:
            recs = []
            for cid in range(6):
                recs.append(
                    {
                        "frame_ids": [f"MOT17-11/{cid*2+1:06d}", f"MOT17-11/{cid*2+2:06d}"],
                        "packet_sizes": [200 + 10 * cid, 80 + 5 * cid],
                        "frame_types": ["I", "P"],
                        "height": 32,
                        "width": 32,
                        "raw_detections": {
                            "gt_boxes": [[[2, 2, 14, 14]], [[4, 4, 18, 18]] if cid % 2 == 0 else []],
                            "ignore_boxes": [[], [[20, 20, 28, 28]]],
                            "pred_boxes": [
                                [[2, 2, 13 + (cid % 3), 14], [19, 19, 27, 27]],
                                [[4, 4, 18, 18]] if cid < 4 else [[1, 1, 8, 8]],
                            ],
                            "pred_scores": [[0.92 - 0.05 * cid, 0.60], [0.75 - 0.08 * cid]],
                        },
                    }
                )
            records_by_crf[crf] = recs

        prep = _PreparedClipBootstrapCurve(records_by_crf, CRFS)
        weights = rng.multinomial(6, [1 / 6] * 6, size=12).astype(np.float64)
        fast_out = prep.evaluate_resampled_curves(weights, batch_size=5)

        crf0 = CRFS[0]
        recs0 = records_by_crf[crf0]
        for b_idx in range(weights.shape[0]):
            dup_records = []
            for c_idx, mult in enumerate(weights[b_idx].astype(int).tolist()):
                dup_records.extend([recs0[c_idx]] * mult)
            exact = summarize_records(dup_records)
            self.assertAlmostEqual(fast_out["total_bpp"][b_idx, 0], exact["total_bpp"], places=7)
            self.assertAlmostEqual(fast_out["p_frame_bpp"][b_idx, 0], exact["p_frame_bpp"], places=7)
            self.assertAlmostEqual(fast_out["gt_ap50_pct"][b_idx, 0], exact["gt_ap50_pct"], places=5)
            self.assertAlmostEqual(fast_out["gt_ap50_95_pct"][b_idx, 0], exact["gt_ap50_95_pct"], places=5)


class PairedMergeV3Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.num_shards = 5  # Parameterized 5-shard topology (NO hardcoded 21! F22 fix)
        self.ids = [
            f"MOT17-{11 if sid < 3 else 13:02d}/{sid * 2 + off + 1:06d}"
            for sid in range(self.num_shards)
            for off in range(2)
        ]
        p1 = patch("scripts.merge_shards.EXPECTED_TEST_FRAME_IDS", self.ids)
        p1.start()
        self.addCleanup(p1.stop)
        p2 = patch("scripts.merge_shards.EXCLUDED_TEST_TAIL_FRAME_IDS", ["MOT17-13/000750"])
        p2.start()
        self.addCleanup(p2.stop)
        self.test_preproc = {
            "resize": "bilinear_aspect_preserving_16_9",
            "img_h": 32,
            "img_w": 64,
            "clip_len": 2,
            "stride": 2,
            "tail_policy": "complete_nonoverlapping_clips_only",
        }
        p3 = patch("scripts.merge_shards.PREPROCESSING", self.test_preproc)
        p3.start()
        self.addCleanup(p3.stop)
        self.validation = {
            "status": "validated",
            "release_ready": True,
            "split": "MOT17-09",
            "loss_total": 1.0,
            "uniform_loss_total": 1.0,
            "protocol": copy.deepcopy(VALIDATION_PROTOCOL),
            "data_fingerprints": {
                "gt_sha256": "a" * 64,
                "frames": [{"frame": number, "sha256": "b" * 64} for number in range(1, 525)],
            },
            "source_soup_sha256": "c" * 64,
            "selection": "uniform_soup_no_test_selection",
        }

    def tearDown(self):
        self.temp.cleanup()

    def write_collection(self, mutate=None):
        manifest = {"schema_version": SCHEMA_VERSION, "expected_shards": self.num_shards, "shards": []}
        for sid in range(self.num_shards):
            directory = self.root / f"shard_{sid:02d}"
            directory.mkdir()
            # Portable collection-relative directory path (F22 fix)
            entry = {
                "shard_id": sid,
                "status": "COMPLETE",
                "downloaded": True,
                "directory": f"shard_{sid:02d}",
                "files": [],
            }
            fids = [self.ids[sid * 2], self.ids[sid * 2 + 1]]
            for index, variant in enumerate(VARIANTS):
                # Spatial_only_scope is index 2 -> spatial_checkpoint_sha256 == "3" * 64
                metadata = {
                    "shard_id": sid,
                    "num_shards": self.num_shards,
                    "variant": variant,
                    "crfs": list(CRFS),
                    "dataset": "MOT17",
                    "split": "test",
                    "expected_frame_ids": self.ids,
                    "excluded_tail_frame_ids": ["MOT17-13/000750"],
                    "checkpoint_sha256": str(index + 1) * 64,
                    "spatial_checkpoint_sha256": "3" * 64,
                    "detector": {"name": "fasterrcnn_resnet50_fpn_v2", "pretrained": True},
                    "codec": copy.deepcopy(CODEC_CONFIGURATION),
                    "metric_policy": copy.deepcopy(METRIC_POLICY),
                    "preprocessing": copy.deepcopy(self.test_preproc),
                    "soup_validation": copy.deepcopy(self.validation),
                }
                records = []
                for q_i, crf in enumerate(CRFS):
                    for method in METHODS:
                        rate_mult = 8 if method == "tscope_temporal_ctu_qp" else 10
                        # Distinct AP@50 across all 4 CRFs: sid < (4 - q_i) hits on frame 1 so each CRF drops recall strictly
                        hit_second = sid < (self.num_shards - q_i)
                        records.append(
                            {
                                "shard_id": sid,
                                "variant": variant,
                                "method": method,
                                "crf": crf,
                                "qp": crf,
                                "frame_ids": fids,
                                "packet_sizes": [rate_mult * (50 - crf), rate_mult * (45 - crf)],
                                "frame_types": ["I", "P"],
                                "height": 32,
                                "width": 64,
                                "task_roi_psnr_db": 32.0 - q_i,
                                "raw_detections": {
                                    "gt_boxes": [[[2.0, 2.0, 12.0, 12.0]], [[2.0, 2.0, 12.0, 12.0]]],
                                    "ignore_boxes": [[], []],
                                    "pred_boxes": [
                                        [[2.0 + 0.5 * q_i, 2.0, 12.0 + 0.5 * q_i, 12.0]],
                                        [[2.0, 2.0, 12.0, 12.0]] if hit_second else [[20.0, 20.0, 30.0, 30.0]],
                                    ],
                                    "pred_scores": [[0.95 - 0.05 * q_i], [0.80 - 0.05 * q_i]],
                                },
                            }
                        )
                bundle = {"schema_version": SCHEMA_VERSION, "metadata": metadata, "records": records}
                if mutate:
                    mutate(sid, variant, bundle)
                path = directory / f"eval_shard_{sid:02d}of{self.num_shards:02d}_{variant}.json"
                path.write_text(json.dumps(bundle), encoding="utf-8")
                entry["files"].append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
            manifest["shards"].append(entry)
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_complete_v3_paired_merge_with_per_seq_bootstrap_and_claim_audit(self):
        manifest = self.write_collection()
        result = merge_eval_jsons_and_print_table(str(self.root), manifest, n_bootstrap=200)
        self.assertEqual(result["schema_version"], 3)
        self.assertTrue(result["eligibility"]["final_ready"])
        self.assertEqual(result["eligibility"]["expected_shards"], 5)
        self.assertEqual(len(result["ablation_variants_summary"]), 5)
        # F09: deterministic per_sequence_summary emitted inside merged_benchmarks.json
        self.assertIn("MOT17-11", result["per_sequence_summary"])
        self.assertIn("MOT17-13", result["per_sequence_summary"])
        self.assertEqual(result["per_sequence_summary"]["MOT17-11"]["frame_count"], 6)
        self.assertEqual(result["per_sequence_summary"]["MOT17-13"]["frame_count"], 4)
        # F15 & F21: bootstrap CIs + P-frame BDBR present
        full_sum = result["ablation_variants_summary"]["full_tscope"]
        self.assertLess(full_sum["bdbr_tot_gt_ap50_pct"], 0.0)
        self.assertLess(full_sum["bdbr_p_gt_ap50_pct"], 0.0)
        self.assertIsInstance(full_sum["bdbr_tot_gt_ap50_pct_ci95"], list)
        self.assertEqual(len(full_sum["bdbr_tot_gt_ap50_pct_ci95"]), 2)
        # F18: explicit claim falsification audit emitted
        self.assertIn("claim_falsification_audit", result)
        self.assertEqual(
            result["claim_falsification_audit"]["claims"]["C01_total_bdbr_vs_anchor"]["status"],
            "supported",
        )

    def test_missing_record_rejected(self):
        manifest = self.write_collection(
            lambda sid, variant, bundle: bundle["records"].pop() if sid == 0 and variant == "full_tscope" else None
        )
        with self.assertRaises(ValueError):
            merge_eval_jsons_and_print_table(str(self.root), manifest, n_bootstrap=20)
        self.assertFalse((self.root / "merged_benchmarks.json").exists())


if __name__ == "__main__":
    unittest.main()
