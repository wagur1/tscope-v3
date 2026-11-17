import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator

from scripts.experiment_protocol import (
    CLAIM_FALSIFICATION_CRITERIA,
    CODEC_CONFIGURATION,
    CRFS,
    EXCLUDED_TEST_TAIL_FRAME_IDS,
    EXPECTED_TEST_FRAME_IDS,
    METRIC_POLICY,
    METHODS,
    PREPROCESSING,
    SCHEMA_VERSION,
    VARIANT_CONFIGS,
    VARIANTS,
    read_manifest_files,
    restore_raw_detections,
    sha256_file,
    validate_checkpoint,
    validate_release_validation,
    verify_convergence_log,
)


def bjontegaard_delta_rate(
    rate_anchor: Sequence[float],
    acc_anchor: Sequence[float],
    rate_test: Sequence[float],
    acc_test: Sequence[float],
) -> float:
    """
    JCTVC-E137 / MPEG-VCM Bjontegaard Delta Bit-Rate (BDBR %) calculation using
    shape-preserving monotonic Piecewise Cubic Hermite Interpolating Polynomials
    (`scipy.interpolate.PchipInterpolator`) integrated analytically in log10-rate domain (F14 fix).
    """
    arrays = [np.asarray(value, dtype=np.float64) for value in (rate_anchor, acc_anchor, rate_test, acc_test)]
    if any(value.ndim != 1 or len(value) < 4 or not np.isfinite(value).all() for value in arrays):
        raise ValueError("BDBR requires at least four finite operating points")
    if len(arrays[0]) != len(arrays[1]) or len(arrays[2]) != len(arrays[3]):
        raise ValueError("BDBR rate and accuracy lengths differ")
    if any(np.any(value <= 0) for value in (arrays[0], arrays[2])):
        raise ValueError("BDBR requires strictly positive rates")
    if len(np.unique(arrays[1])) < 4 or len(np.unique(arrays[3])) < 4:
        raise ValueError("BDBR requires four distinct accuracy values per curve")

    r1 = np.log10(arrays[0])
    a1 = arrays[1]
    r2 = np.log10(arrays[2])
    a2 = arrays[3]

    order1 = np.argsort(a1)
    a1, r1 = a1[order1], r1[order1]
    order2 = np.argsort(a2)
    a2, r2 = a2[order2], r2[order2]

    min_int = float(max(np.min(a1), np.min(a2)))
    max_int = float(min(np.max(a1), np.max(a2)))
    if max_int <= min_int or np.isnan(min_int) or np.isnan(max_int):
        return float("nan")

    pchip1 = PchipInterpolator(a1, r1)
    pchip2 = PchipInterpolator(a2, r2)

    int1 = float(pchip1.integrate(min_int, max_int))
    int2 = float(pchip2.integrate(min_int, max_int))
    avg_diff = (int2 - int1) / (max_int - min_int)
    return float((np.power(10.0, avg_diff) - 1.0) * 100.0)


def average_compatible_states(states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not states:
        raise ValueError("Cannot average empty checkpoint collection")
    reference = states[0]
    for state in states:
        if set(state) != set(reference):
            raise ValueError("Soup state keys differ")
        for key, value in state.items():
            first = reference[key]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != first.shape
                or value.dtype != first.dtype
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"Incompatible soup tensor: {key}")
            if not value.is_floating_point() and not torch.equal(value, first):
                raise ValueError(f"Non-floating soup buffer differs: {key}")
    result: Dict[str, torch.Tensor] = {}
    for key, first in reference.items():
        if first.is_floating_point():
            result[key] = torch.stack([state[key].to(torch.float64) for state in states]).mean(0).to(first.dtype)
        else:
            result[key] = first.clone()
    return result


def merge_checkpoints_soup(
    ckpt_dir: str,
    output_ckpt: str,
    manifest_path: Any = None,
    variants: Optional[Sequence[str]] = None,
    expected_shards: Optional[int] = None,
    allow_unbalanced: bool = False,
) -> Dict[str, str]:
    """
    Aggregates trained shard checkpoints via Uniform Model Soup strictly grouped by `variant`
    with parameterized shard count (`F22`) and balanced shards per variant (`F07` & `F15`).
    """
    if manifest_path is None:
        raise ValueError("Soup preparation requires --manifest; implicit directory merging is disabled")
    manifest, files = read_manifest_files(manifest_path, ".pt", expected_shards=expected_shards)
    num_shards = int(manifest["expected_shards"])
    manifest_root = Path(manifest_path).resolve().parent

    selected = list(VARIANTS if variants is None else variants)
    if not selected or len(set(selected)) != len(selected) or any(v not in VARIANTS for v in selected):
        raise ValueError("Invalid soup variant selection")

    by_variant: Dict[str, Dict[int, Any]] = {variant: {} for variant in selected}
    for entry, path, digest in files:
        sid = entry["shard_id"]
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        variant = checkpoint.get("variant") or entry.get("expected_variant") or VARIANTS[sid % len(VARIANTS)]
        if variant not in by_variant:
            continue
        expected_filename = f"tscope_{variant}_shard{sid:02d}of{num_shards:02d}.pt"
        if (
            path.name != expected_filename
            or (entry.get("expected_variant") and entry.get("expected_variant") != variant)
            or sid in by_variant[variant]
        ):
            raise ValueError("Unexpected or duplicate shard checkpoint in manifest")
        validate_checkpoint(checkpoint, sid, variant, expected_num_shards=num_shards)
        cand_logs = []
        if entry.get("kernel_slug"):
            cand_logs.append(f"{entry['kernel_slug']}.log")
        cand_logs.extend([f"t-scope-v3-shard-{sid:02d}.log", f"t-scope-vcm-shard-{sid:02d}.log"])
        log_entry = next(
            (item for name in cand_logs for item in entry.get("files", []) if item["path"] == name),
            None,
        )
        log_path = (path.parent / log_entry["path"]) if log_entry is not None else (path.parent / cand_logs[0])
        if not log_entry or not log_path.is_file() or sha256_file(log_path) != log_entry.get("sha256"):
            raise ValueError("Missing checked convergence log")
        log = log_path.read_text(encoding="utf-8")
        log_meta = verify_convergence_log(log, sid, variant, num_shards=num_shards)
        if checkpoint.get("stopped_early") is True and log_meta.get("convergence_mode") != "early_stop":
            raise ValueError("Checkpoint stopped_early=True conflicts with convergence log")
        if checkpoint.get("stopped_early") is False and log_meta.get("convergence_mode") != "full_schedule":
            raise ValueError("Checkpoint stopped_early=False requires full-schedule completion log")
        by_variant[variant][sid] = (checkpoint, path, digest, log_entry["sha256"])

    outputs: Dict[str, Dict[str, Any]] = {}
    shard_counts: List[int] = []
    for variant, collection in by_variant.items():
        expected = {
            e["shard_id"]
            for e in manifest.get("shards", [])
            if e.get("downloaded") and (e.get("expected_variant") or VARIANTS[e["shard_id"] % len(VARIANTS)]) == variant
        } or {sid for sid in range(num_shards) if VARIANTS[sid % len(VARIANTS)] == variant}
        if not collection or set(collection) != expected:
            raise ValueError(f"Incomplete {variant}: missing shards {sorted(expected - set(collection))}")
        shard_counts.append(len(collection))

    if len(shard_counts) > 1 and len(set(shard_counts)) != 1 and not allow_unbalanced:
        raise ValueError(
            f"F07/F15 balanced-ablation contract violated: variants have unequal shard counts {dict(zip(selected, shard_counts))}"
        )

    for variant, collection in by_variant.items():
        rows = [collection[sid] for sid in sorted(collection)]
        state = average_compatible_states([row[0]["state_dict"] for row in rows])
        model = validate_checkpoint(
            {**rows[0][0], "state_dict": state},
            rows[0][0]["shard_id"],
            variant,
            expected_num_shards=num_shards,
        )
        model.eval()
        v_cfg = VARIANT_CONFIGS[variant]
        with torch.no_grad():
            for fam_test in (0, 1):
                prediction = model(
                    torch.zeros(1, 2, 3, 32, 32),
                    torch.tensor([fam_test], dtype=torch.long),
                    torch.tensor([0.4]),
                    reuse_iframe_qp=v_cfg["reuse_iframe_qp"],
                    disable_temporal=v_cfg["disable_temporal"],
                )
                if not all(bool(torch.isfinite(value).all()) for value in prediction.values() if isinstance(value, torch.Tensor)):
                    raise ValueError(f"Soup inference returned nonfinite outputs for family_id={fam_test}")

        constituents = []
        for row in rows:
            rel_src = (
                str(row[1].resolve().relative_to(manifest_root))
                if row[1].resolve().is_relative_to(manifest_root)
                else row[1].name
            )
            constituents.append(
                {
                    "shard_id": row[0]["shard_id"],
                    "checkpoint_sha256": row[2],
                    "log_sha256": row[3],
                    "best_epoch": row[0]["best_epoch"],
                    "best_val_loss": row[0]["best_val_loss"],
                    "epochs_run": row[0]["epochs_run"],
                    "git_revision": row[0].get("training_configuration", {}).get("git_revision"),
                    "source_path": rel_src,
                }
            )
        outputs[variant] = {
            "schema_version": SCHEMA_VERSION,
            "variant": variant,
            "merged_shards": len(rows),
            "state_dict": state,
            "constituents": constituents,
            "strict_load_passed": True,
            "inference_smoke_passed": True,
            "configuration": {
                "codec": CODEC_CONFIGURATION,
                "initialization_seed": 2026,
                "use_yolov8s_in_training": False,
                "num_families": 2,
                "max_delta_qp": float(v_cfg["max_delta_qp"]),
                "ablation": dict(v_cfg),
            },
            "validation": {"status": "not_evaluated", "split": "MOT17-09", "release_ready": False},
            "source_manifest_sha256": sha256_file(manifest_path),
        }

    destination = Path(ckpt_dir)
    paths = {variant: destination / f"tscope_soup_{variant}.pt" for variant in selected}
    alias = Path(output_ckpt) if output_ckpt and "full_tscope" in selected else None
    if any(path.exists() for path in paths.values()) or (alias is not None and alias.exists()):
        raise FileExistsError("Soup output already exists; use a new output directory")
    destination.mkdir(parents=True, exist_ok=True)
    for variant, path in paths.items():
        with path.open("xb") as handle:
            torch.save(outputs[variant], handle)
        print(f"Prepared {variant}: {outputs[variant]['merged_shards']} checked shards -> {path}; validation pending")
    if alias is not None and alias not in paths.values():
        alias.parent.mkdir(parents=True, exist_ok=True)
        with alias.open("xb") as handle:
            torch.save(outputs["full_tscope"], handle)
    return {variant: str(path) for variant, path in paths.items()}


def summarize_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    from models.frozen_detector_fpn import FrozenFPNDetectorTeacher

    raw: Dict[str, List[torch.Tensor]] = {
        key: []
        for key in ("gt_boxes", "ignore_boxes", "pred_boxes", "pred_scores", "yolo_pred_boxes", "yolo_pred_scores")
    }
    has_yolo = True
    pixels = i_pixels = p_pixels = 0
    bits = i_bits = p_bits = 0
    roi_sum = 0.0
    roi_frames = 0
    for record in records:
        count = len(record["frame_ids"])
        restored = restore_raw_detections(record["raw_detections"], count)
        for key in ("gt_boxes", "ignore_boxes", "pred_boxes", "pred_scores"):
            raw[key].extend(restored[key])
        if "yolo_pred_boxes" in restored and "yolo_pred_scores" in restored:
            raw["yolo_pred_boxes"].extend(restored["yolo_pred_boxes"])
            raw["yolo_pred_scores"].extend(restored["yolo_pred_scores"])
        else:
            has_yolo = False
        area = int(record["height"]) * int(record["width"])
        for size, frame_type in zip(record["packet_sizes"], record["frame_types"]):
            bits += int(size) * 8
            pixels += area
            if frame_type == "I":
                i_bits += int(size) * 8
                i_pixels += area
            else:
                p_bits += int(size) * 8
                p_pixels += area
        roi = record.get("task_roi_psnr_db")
        if roi is not None and np.isfinite(roi):
            roi_sum += float(roi) * count
            roi_frames += count

    metrics = FrozenFPNDetectorTeacher.compute_dataset_level_coco_metrics(
        raw["gt_boxes"], raw["pred_boxes"], raw["pred_scores"], raw["ignore_boxes"]
    )
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("Global AP unavailable: no active held-out GT")
    summary: Dict[str, Any] = {
        "total_bpp": bits / pixels,
        "i_frame_bpp": i_bits / i_pixels if i_pixels else None,
        "p_frame_bpp": p_bits / p_pixels if p_pixels else None,
        "roi_psnr_db": roi_sum / roi_frames if roi_frames else None,
        "roi_psnr_aggregation": "frame_weighted_mean_clip_roi_psnr_diagnostic_only",
        "packet_bytes": bits // 8,
        "pixel_count": pixels,
        "frame_count": len(raw["gt_boxes"]),
        **metrics,
    }
    if has_yolo and len(raw["yolo_pred_boxes"]) == len(raw["gt_boxes"]):
        yolo_metrics = FrozenFPNDetectorTeacher.compute_dataset_level_coco_metrics(
            raw["gt_boxes"], raw["yolo_pred_boxes"], raw["yolo_pred_scores"], raw["ignore_boxes"]
        )
        summary["yolo_gt_ap50_pct"] = yolo_metrics["gt_ap50_pct"]
        summary["yolo_gt_ap50_95_pct"] = yolo_metrics["gt_ap50_95_pct"]
    return summary


class _PreparedClipBootstrapCurve:
    """
    Precomputes exact per-image IoU matching tables across sorted clips so 10,000-resample
    paired clip-block bootstrap 101-point COCO AP + BDBR executes with zero approximation error.
    """

    IOU_THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(10)]
    RECALL_GRID = np.linspace(0.0, 1.0, 101, dtype=np.float64)

    def __init__(self, records_by_crf: Dict[int, List[Dict[str, Any]]], crfs: Sequence[int]) -> None:
        from models.frozen_detector_fpn import FrozenFPNDetectorTeacher

        self.crfs = list(crfs)
        first_records = records_by_crf[self.crfs[0]]
        self.num_clips = len(first_records)
        self.has_p_frames = True
        self.has_yolo = True

        self.total_bits: Dict[int, np.ndarray] = {}
        self.total_pixels: Dict[int, np.ndarray] = {}
        self.p_bits: Dict[int, np.ndarray] = {}
        self.p_pixels: Dict[int, np.ndarray] = {}
        self.frcnn_tables: Dict[int, Dict[str, Any]] = {}
        self.yolo_tables: Dict[int, Dict[str, Any]] = {}

        for crf in self.crfs:
            recs = records_by_crf[crf]
            if len(recs) != self.num_clips:
                raise ValueError("Clip count mismatch across CRFs in bootstrap preparation")

            t_bits = np.zeros(self.num_clips, dtype=np.float64)
            t_pix = np.zeros(self.num_clips, dtype=np.float64)
            pb_arr = np.zeros(self.num_clips, dtype=np.float64)
            pp_arr = np.zeros(self.num_clips, dtype=np.float64)

            all_gt: List[torch.Tensor] = []
            all_ig: List[torch.Tensor] = []
            all_pb: List[torch.Tensor] = []
            all_ps: List[torch.Tensor] = []
            all_ypb: List[torch.Tensor] = []
            all_yps: List[torch.Tensor] = []
            img_to_clip: List[int] = []

            for c_idx, rec in enumerate(recs):
                count = len(rec["frame_ids"])
                restored = restore_raw_detections(rec["raw_detections"], count)
                area = float(int(rec["height"]) * int(rec["width"]))
                for sz, ft in zip(rec["packet_sizes"], rec["frame_types"]):
                    b_val = float(int(sz) * 8)
                    t_bits[c_idx] += b_val
                    t_pix[c_idx] += area
                    if ft != "I":
                        pb_arr[c_idx] += b_val
                        pp_arr[c_idx] += area

                all_gt.extend(restored["gt_boxes"])
                all_ig.extend(restored["ignore_boxes"])
                all_pb.extend(restored["pred_boxes"])
                all_ps.extend(restored["pred_scores"])
                img_to_clip.extend([c_idx] * count)
                if "yolo_pred_boxes" in restored and "yolo_pred_scores" in restored:
                    all_ypb.extend(restored["yolo_pred_boxes"])
                    all_yps.extend(restored["yolo_pred_scores"])
                else:
                    self.has_yolo = False

            if float(pp_arr.sum()) <= 0.0:
                self.has_p_frames = False
            self.total_bits[crf] = t_bits
            self.total_pixels[crf] = t_pix
            self.p_bits[crf] = pb_arr
            self.p_pixels[crf] = pp_arr
            self.frcnn_tables[crf] = self._build_detector_table(
                FrozenFPNDetectorTeacher, all_gt, all_pb, all_ps, all_ig, img_to_clip
            )
            if self.has_yolo:
                self.yolo_tables[crf] = self._build_detector_table(
                    FrozenFPNDetectorTeacher, all_gt, all_ypb, all_yps, all_ig, img_to_clip
                )

    def _build_detector_table(
        self,
        teacher_cls: Any,
        all_gt: List[torch.Tensor],
        all_pb: List[torch.Tensor],
        all_ps: List[torch.Tensor],
        all_ig: List[torch.Tensor],
        img_to_clip: List[int],
    ) -> Dict[str, Any]:
        total_gt, n_gts, img_gt_ious, img_ig_max_ious, sorted_pairs = teacher_cls._prepare_dataset_ap_cache(
            all_gt, all_pb, all_ps, all_ignore_boxes=all_ig
        )
        clip_gt = np.zeros(self.num_clips, dtype=np.float64)
        for img_idx, g_cnt in enumerate(n_gts):
            clip_gt[img_to_clip[img_idx]] += float(g_cnt)

        per_thr = []
        neg_one = np.float32(-1.0)
        for thr in self.IOU_THRESHOLDS:
            if total_gt == 0 or not sorted_pairs:
                per_thr.append(
                    (
                        np.zeros(0, dtype=np.float64),
                        np.zeros(0, dtype=np.float64),
                        np.zeros(0, dtype=np.int64),
                    )
                )
                continue
            gt_matched = [np.zeros(n, dtype=bool) for n in n_gts]
            tp_list: List[float] = []
            fp_list: List[float] = []
            c_list: List[int] = []
            for img_idx, pred_j in sorted_pairs:
                gt_iou_mat = img_gt_ious[img_idx]
                matched_active = False
                if gt_iou_mat is not None:
                    m_flags = gt_matched[img_idx]
                    if not m_flags.all():
                        masked = np.where(m_flags, neg_one, gt_iou_mat[pred_j])
                        best_gt_idx = int(np.argmax(masked))
                        if float(masked[best_gt_idx]) >= thr:
                            m_flags[best_gt_idx] = True
                            tp_list.append(1.0)
                            fp_list.append(0.0)
                            c_list.append(img_to_clip[img_idx])
                            matched_active = True
                if matched_active:
                    continue
                ig_max = img_ig_max_ious[img_idx]
                if ig_max is not None and float(ig_max[pred_j]) >= thr:
                    continue
                tp_list.append(0.0)
                fp_list.append(1.0)
                c_list.append(img_to_clip[img_idx])
            per_thr.append(
                (
                    np.asarray(tp_list, dtype=np.float64),
                    np.asarray(fp_list, dtype=np.float64),
                    np.asarray(c_list, dtype=np.int64),
                )
            )
        return {"clip_gt": clip_gt, "per_thr": per_thr}

    @classmethod
    def _weighted_ap_batch(
        cls,
        weights: np.ndarray,
        clip_gt: np.ndarray,
        tp_arr: np.ndarray,
        fp_arr: np.ndarray,
        clip_idx_arr: np.ndarray,
    ) -> np.ndarray:
        """Exact 101-point AP for a batch of integer clip-multiplicity vectors `weights` [B, K]."""
        bsz = weights.shape[0]
        tot_gt = weights @ clip_gt  # [B]
        out = np.full(bsz, np.nan, dtype=np.float64)
        valid_gt_rows = np.where(tot_gt > 0.0)[0]
        if valid_gt_rows.size == 0:
            return out
        if tp_arr.size == 0:
            out[valid_gt_rows] = 0.0
            return out

        w_sub = weights[valid_gt_rows]  # [Bv, K]
        gt_sub = tot_gt[valid_gt_rows]  # [Bv]
        w_pred = w_sub[:, clip_idx_arr]  # [Bv, N]
        tp_cum = np.cumsum(w_pred * tp_arr[None, :], axis=1)
        fp_cum = np.cumsum(w_pred * fp_arr[None, :], axis=1)
        has_any_pred = (tp_cum[:, -1] + fp_cum[:, -1]) > 0.0

        rec = tp_cum / gt_sub[:, None]
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        mpre = np.maximum.accumulate(prec[:, ::-1], axis=1)[:, ::-1]

        grid = cls.RECALL_GRID
        n_cols = rec.shape[1]
        sub_out = np.zeros(valid_gt_rows.size, dtype=np.float64)
        for r_i in range(valid_gt_rows.size):
            if not has_any_pred[r_i]:
                sub_out[r_i] = 0.0
                continue
            inds = np.searchsorted(rec[r_i], grid, side="left")
            ok = inds < n_cols
            if np.any(ok):
                sub_out[r_i] = float(mpre[r_i, inds[ok]].sum()) / 101.0
        out[valid_gt_rows] = sub_out
        return out

    def evaluate_resampled_curves(self, weights: np.ndarray, batch_size: int = 500) -> Dict[str, np.ndarray]:
        """
        Evaluate resampled rate and exact 101-point AP curves across all CRFs for `weights` [N_boot, K].
        Returns arrays of shape [N_boot, len(CRFS)].
        """
        n_boot = weights.shape[0]
        n_qps = len(self.crfs)
        res: Dict[str, np.ndarray] = {
            "total_bpp": np.zeros((n_boot, n_qps), dtype=np.float64),
            "p_frame_bpp": np.full((n_boot, n_qps), np.nan, dtype=np.float64),
            "gt_ap50_pct": np.full((n_boot, n_qps), np.nan, dtype=np.float64),
            "gt_ap50_95_pct": np.full((n_boot, n_qps), np.nan, dtype=np.float64),
        }
        if self.has_yolo:
            res["yolo_gt_ap50_pct"] = np.full((n_boot, n_qps), np.nan, dtype=np.float64)
            res["yolo_gt_ap50_95_pct"] = np.full((n_boot, n_qps), np.nan, dtype=np.float64)

        for q_i, crf in enumerate(self.crfs):
            denom_tot = np.maximum(weights @ self.total_pixels[crf], 1.0)
            res["total_bpp"][:, q_i] = (weights @ self.total_bits[crf]) / denom_tot
            if self.has_p_frames:
                denom_p = weights @ self.p_pixels[crf]
                valid_p = denom_p > 0.0
                res["p_frame_bpp"][valid_p, q_i] = (weights[valid_p] @ self.p_bits[crf]) / denom_p[valid_p]

            for prefix, table in (
                ("", self.frcnn_tables[crf]),
                *(("yolo_", self.yolo_tables[crf]) for _ in ([1] if self.has_yolo else [])),
            ):
                ap_per_thr = np.full((len(self.IOU_THRESHOLDS), n_boot), np.nan, dtype=np.float64)
                for t_idx, (tp_arr, fp_arr, c_arr) in enumerate(table["per_thr"]):
                    for start in range(0, n_boot, batch_size):
                        end = min(n_boot, start + batch_size)
                        ap_per_thr[t_idx, start:end] = self._weighted_ap_batch(
                            weights[start:end], table["clip_gt"], tp_arr, fp_arr, c_arr
                        )
                # Round matching FrozenFPNDetectorTeacher.compute_dataset_level_coco_metrics convention
                res[f"{prefix}gt_ap50_pct"][:, q_i] = np.round(ap_per_thr[0] * 100.0, 3)
                res[f"{prefix}gt_ap50_95_pct"][:, q_i] = np.round(np.mean(ap_per_thr, axis=0) * 100.0, 3)

        return res


def _compute_bdbr_samples(
    anchor_rates: np.ndarray,
    anchor_accs: np.ndarray,
    test_rates: np.ndarray,
    test_accs: np.ndarray,
) -> np.ndarray:
    """Compute BDBR (%) across bootstrap replicates [N_boot, len(CRFS)], returning NaN where undefined."""
    n_boot = anchor_rates.shape[0]
    samples = np.full(n_boot, np.nan, dtype=np.float64)
    if np.allclose(anchor_rates, test_rates) and np.allclose(anchor_accs, test_accs, equal_nan=True):
        # Anchor vs itself is identically 0.0% whenever curve has >=4 distinct finite points
        for i in range(n_boot):
            a_row = anchor_accs[i]
            r_row = anchor_rates[i]
            if np.isfinite(a_row).all() and np.isfinite(r_row).all() and np.all(r_row > 0) and len(np.unique(a_row)) >= 4:
                samples[i] = 0.0
        return samples

    for i in range(n_boot):
        try:
            val = bjontegaard_delta_rate(anchor_rates[i], anchor_accs[i], test_rates[i], test_accs[i])
            if math.isfinite(val):
                samples[i] = val
        except ValueError:
            continue
    return samples


def _summarize_bootstrap_samples(samples: np.ndarray) -> Optional[Dict[str, Any]]:
    valid = samples[np.isfinite(samples)]
    if valid.size == 0:
        return None
    ci_low, ci_high = np.percentile(valid, [2.5, 97.5])
    return {
        "ci95": [float(ci_low), float(ci_high)],
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0,
        "valid_resamples": int(valid.size),
        "total_resamples": int(samples.size),
    }


def build_bdbr_summary(
    per_qp: Dict[str, Dict[str, Any]],
    anchor: Dict[str, Dict[str, Any]],
    crfs: Sequence[int] = CRFS,
    bootstrap_curves: Optional[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]] = None,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    result: Dict[str, Any] = {
        "per_qp": per_qp,
        "bdbr_unavailable_reasons": {},
        "bootstrap_ci95": {},
        "curve_diagnostics": {},
    }
    raw_samples: Dict[str, np.ndarray] = {}

    accuracy_pairs = [("gt_ap50_pct", "gt_ap50_pct"), ("gt_ap50_95_pct", "gt_ap50_95_pct")]
    if all("yolo_gt_ap50_pct" in per_qp[str(crf)] and "yolo_gt_ap50_pct" in anchor[str(crf)] for crf in crfs):
        accuracy_pairs.extend(
            [("yolo_gt_ap50_pct", "yolo_gt_ap50_pct"), ("yolo_gt_ap50_95_pct", "yolo_gt_ap50_95_pct")]
        )

    for accuracy_key, _ in accuracy_pairs:
        vals = sorted(float(per_qp[str(crf)][accuracy_key]) for crf in crfs)
        min_gap = min(vals[i + 1] - vals[i] for i in range(len(vals) - 1))
        result["curve_diagnostics"][accuracy_key] = {
            "min_adjacent_ap_gap_pct": round(float(min_gap), 4),
            "near_flat_plateau_warning": bool(min_gap < 0.5),
            "interpolator": "scipy.interpolate.PchipInterpolator",
        }

    for rate_key, prefix in (("total_bpp", "tot"), ("p_frame_bpp", "p")):
        for accuracy_key, suffix in accuracy_pairs:
            field = f"bdbr_{prefix}_{suffix}"
            try:
                anchor_rates = [anchor[str(crf)][rate_key] for crf in crfs]
                test_rates = [per_qp[str(crf)][rate_key] for crf in crfs]
                if any(v is None for v in anchor_rates + test_rates):
                    raise ValueError(f"Missing {rate_key} operating points (e.g., I-only 1-frame clip)")
                value = bjontegaard_delta_rate(
                    anchor_rates,
                    [anchor[str(crf)][accuracy_key] for crf in crfs],
                    test_rates,
                    [per_qp[str(crf)][accuracy_key] for crf in crfs],
                )
                if not np.isfinite(value):
                    raise ValueError("No positive common accuracy interval")
                result[field] = value
            except (ValueError, TypeError, np.linalg.LinAlgError) as error:
                result[field] = None
                result["bdbr_unavailable_reasons"][field] = str(error)

            if bootstrap_curves is not None and result[field] is not None:
                anc_b, tst_b = bootstrap_curves
                if accuracy_key in anc_b and accuracy_key in tst_b:
                    b_samples = _compute_bdbr_samples(
                        anc_b[rate_key], anc_b[accuracy_key], tst_b[rate_key], tst_b[accuracy_key]
                    )
                    raw_samples[field] = b_samples
                    ci_info = _summarize_bootstrap_samples(b_samples)
                    if ci_info is not None:
                        result[f"{field}_ci95"] = ci_info["ci95"]
                        result["bootstrap_ci95"][field] = ci_info
                    else:
                        result[f"{field}_ci95"] = None
                else:
                    result[f"{field}_ci95"] = None
            else:
                result[f"{field}_ci95"] = None

    return result, raw_samples


def build_claim_falsification_audit(
    ablation_summary: Dict[str, Dict[str, Any]],
    overall_summary: Dict[str, Dict[str, Any]],
    ablation_boot_samples: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Any]:
    """
    Evaluates pre-registered claims C01..C05 against paired clip-block bootstrap 95% CIs (F18 fix).
    Separates structural artifact completeness (`eligibility.final_ready`) from scientific falsification.
    """
    claims_out: Dict[str, Any] = {}
    all_supported = True

    for claim_id, spec in CLAIM_FALSIFICATION_CRITERIA.items():
        metric = spec["metric"]
        variant = spec["variant"]
        comparator = spec["comparator"]
        threshold = float(spec["threshold_pct"])
        ci_cap = float(spec["require_ci95_upper_below"])

        var_entry = ablation_summary.get(variant, {})
        val_var = var_entry.get(metric)

        if comparator == "anchor_x265":
            delta_point = val_var
            ci_info = var_entry.get("bootstrap_ci95", {}).get(metric)
            ci95 = ci_info["ci95"] if ci_info else None
        else:
            comp_entry = ablation_summary.get(comparator, overall_summary.get(comparator, {}))
            val_comp = comp_entry.get(metric)
            if val_var is not None and val_comp is not None:
                delta_point = float(val_var) - float(val_comp)
            else:
                delta_point = None
            s_var = ablation_boot_samples.get(variant, {}).get(metric)
            s_comp = ablation_boot_samples.get(comparator, {}).get(metric)
            if s_var is not None and s_comp is not None:
                diff_info = _summarize_bootstrap_samples(s_var - s_comp)
                ci95 = diff_info["ci95"] if diff_info else None
            else:
                ci95 = None

        if delta_point is None or not math.isfinite(delta_point):
            status = "unavailable"
            all_supported = False
        elif delta_point >= threshold:
            status = "falsified"
            all_supported = False
        elif ci95 is not None and ci95[1] < ci_cap:
            status = "supported"
        else:
            status = "inconclusive_ci_crosses_threshold"
            all_supported = False

        claims_out[claim_id] = {
            "description": spec["description"],
            "metric": metric,
            "variant": variant,
            "comparator": comparator,
            "delta_bdbr_pct": round(float(delta_point), 4) if delta_point is not None else None,
            "ci95_pct": [round(float(ci95[0]), 4), round(float(ci95[1]), 4)] if ci95 is not None else None,
            "threshold_pct": threshold,
            "status": status,
        }

    return {
        "bootstrap_resamples": int(METRIC_POLICY.get("bootstrap_resamples", 10000)),
        "all_claims_supported": all_supported,
        "claims": claims_out,
    }


def _sequence_prefix(frame_id: str) -> str:
    return frame_id.split("/")[0]


def merge_eval_jsons_and_print_table(
    eval_dir: str,
    manifest_path: Any = None,
    output_path: Any = None,
    expected_shards: Optional[int] = None,
    n_bootstrap: int = 10000,
    bootstrap_seed: int = 2026,
) -> Dict[str, Any]:
    """
    Aggregates `eval_shard_*.json` across all shards, computes global 101-point AP,
    I/P/Total bpp, PCHIP BDBR (%), 10,000-resample paired clip-block bootstrap 95% CIs (`F15`),
    deterministic `per_sequence_summary` (`F09`), and `claim_falsification_audit` (`F18`).
    """
    if manifest_path is None:
        raise ValueError("Final evaluation requires --manifest; legacy clip-AP summaries are not final benchmarks")
    if type(n_bootstrap) is not int or n_bootstrap < 0:
        raise ValueError("n_bootstrap must be a non-negative integer")

    manifest, files = read_manifest_files(manifest_path, ".json", expected_shards=expected_shards)
    num_shards = int(manifest["expected_shards"])

    bundles: Dict[Tuple[int, str], Dict[str, str]] = {}
    common: Optional[Dict[str, Any]] = None
    hashes: Dict[str, str] = {}
    expected_ids: Optional[List[str]] = None
    grouped: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    spatial_hash: Optional[str] = None
    ground_truth: Dict[str, Tuple[Any, ...]] = {}

    for entry, path, digest in files:
        if not path.name.startswith("eval_shard_"):
            continue
        bundle = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(bundle, dict) or bundle.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Legacy or malformed evaluation schema (expected schema_version={SCHEMA_VERSION})")
        metadata = bundle.get("metadata", {})
        sid, variant = metadata.get("shard_id"), metadata.get("variant")
        if (
            sid != entry["shard_id"]
            or metadata.get("num_shards") != num_shards
            or variant not in VARIANTS
            or (sid, variant) in bundles
        ):
            raise ValueError("Invalid or duplicate evaluation shard identity")
        if path.name != f"eval_shard_{sid:02d}of{num_shards:02d}_{variant}.json":
            raise ValueError("Evaluation filename differs from identity")
        if metadata.get("crfs") != CRFS or metadata.get("dataset") != "MOT17" or metadata.get("split") != "test":
            raise ValueError("Evaluation operating points or held-out protocol mismatch")
        cohort = {key: metadata.get(key) for key in ("detector", "codec", "metric_policy", "preprocessing")}
        if any(value is None for value in cohort.values()):
            raise ValueError("Evaluation provenance is incomplete")
        detector = cohort["detector"]
        if detector.get("name") != "fasterrcnn_resnet50_fpn_v2" or detector.get("pretrained") is not True:
            raise ValueError("Final AP requires identified pretrained Faster R-CNN v2")
        if common is None:
            common = cohort
        elif common != cohort:
            raise ValueError("Evaluation cohorts have different detector/codec/metric/preprocessing settings")
        if any(cohort["codec"].get(key) != value for key, value in CODEC_CONFIGURATION.items()):
            raise ValueError("Evaluation changed preserved v3 codec configuration")
        if cohort["metric_policy"] != METRIC_POLICY or cohort["preprocessing"] != PREPROCESSING:
            raise ValueError("Evaluation metric/ignore/preprocessing policy differs from declared v3 protocol")
        validation = metadata.get("soup_validation", {})
        validate_release_validation(validation)

        ids = metadata.get("expected_frame_ids")
        if ids != EXPECTED_TEST_FRAME_IDS or metadata.get("excluded_tail_frame_ids") != EXCLUDED_TEST_TAIL_FRAME_IDS:
            raise ValueError(
                f"Final benchmark requires the complete {len(EXPECTED_TEST_FRAME_IDS)}-frame held-out protocol and explicit tail exclusions"
            )
        if (
            not isinstance(ids, list)
            or not ids
            or len(ids) != len(set(ids))
            or any(
                not isinstance(value, str) or not (value.startswith("MOT17-11/") or value.startswith("MOT17-13/"))
                for value in ids
            )
        ):
            raise ValueError("Invalid held-out physical frame manifest")
        if expected_ids is None:
            expected_ids = ids
        elif expected_ids != ids:
            raise ValueError("Expected physical frames differ across evaluation cohorts")

        model_hash = metadata.get("checkpoint_sha256")
        spatial = metadata.get("spatial_checkpoint_sha256")
        if any(
            not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in (model_hash, spatial)
        ):
            raise ValueError("Missing checkpoint hashes")
        if hashes.setdefault(variant, model_hash) != model_hash:
            raise ValueError("A variant evaluated different checkpoint weights across shards")
        if spatial_hash is None:
            spatial_hash = spatial
        elif spatial_hash != spatial:
            raise ValueError("Spatial baseline checkpoint differs across cohorts")

        bundles[sid, variant] = {"sha256": digest, "path": str(path.name)}
        records = bundle.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError("Empty evaluation shard")

        exp_h = PREPROCESSING.get("img_h", PREPROCESSING.get("img_size"))
        exp_w = PREPROCESSING.get("img_w", PREPROCESSING.get("img_size"))
        exp_clip_len = PREPROCESSING["clip_len"]
        exp_bframes = int(CODEC_CONFIGURATION.get("bframes", 0))

        for record in records:
            method, crf = record.get("method"), record.get("crf")
            if (
                record.get("shard_id") != sid
                or record.get("variant") != variant
                or method not in METHODS
                or crf not in CRFS
                or record.get("qp", crf) != crf
            ):
                raise ValueError("Record identity or operating point mismatch")
            frames = record.get("frame_ids")
            if not isinstance(frames, list) or not frames or len(frames) != len(set(frames)) or not set(frames).issubset(ids):
                raise ValueError("Record physical frames invalid")
            if len({_sequence_prefix(fid) for fid in frames}) != 1:
                raise ValueError("Clip record spans multiple sequences")
            sizes, types = record.get("packet_sizes"), record.get("frame_types")
            h, w = record.get("height"), record.get("width")
            if (
                type(h) is not int
                or type(w) is not int
                or h <= 0
                or w <= 0
                or not isinstance(sizes, list)
                or not isinstance(types, list)
                or len(sizes) != len(frames)
                or len(types) != len(frames)
            ):
                raise ValueError("Packet sizes or pixel denominators invalid")
            valid_structure = (
                types == ["I"] + ["P"] * (len(frames) - 1)
                if exp_bframes == 0
                else (types[0] == "I" and all(ft in ("P", "B") for ft in types[1:]))
            )
            if h != exp_h or w != exp_w or len(frames) != exp_clip_len or not valid_structure:
                raise ValueError("Record pixels, clip length, or frame structure differs from v3 evaluation protocol")
            if any(type(value) is not int or value <= 0 for value in sizes) or any(value not in ("I", "P", "B") for value in types):
                raise ValueError("Invalid packet byte size or frame type")
            restored = restore_raw_detections(record.get("raw_detections", {}), len(frames))
            group = grouped.setdefault((variant, method, crf), {"frames": set(), "records": [], "by_shard": {}})
            if group["frames"].intersection(frames):
                raise ValueError("Duplicate physical frame in one evaluation curve")
            group["frames"].update(frames)
            group["records"].append(record)
            group["by_shard"].setdefault(sid, set()).update(frames)
            for index, frame in enumerate(frames):
                gt = (restored["gt_boxes"][index].tolist(), restored["ignore_boxes"][index].tolist(), h, w)
                if ground_truth.setdefault(frame, gt) != gt:
                    raise ValueError("GT or spatial preprocessing differs for a paired physical frame")

    expected_pairs = {(sid, variant) for sid in range(num_shards) for variant in VARIANTS}
    if set(bundles) != expected_pairs:
        raise ValueError(f"Incomplete evaluation: {len(bundles)}/{len(expected_pairs)} shard-variant outputs")
    if hashes.get("spatial_only_scope") != spatial_hash:
        raise ValueError("Spatial reference hash differs from spatial variant soup")

    assert expected_ids is not None
    all_ids = set(expected_ids)
    sorted_records_map: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    reference_partition: Optional[Dict[int, set]] = None
    reference_clip_keys: Optional[List[Tuple[str, ...]]] = None

    for variant in VARIANTS:
        for method in METHODS:
            for crf in CRFS:
                group = grouped.get((variant, method, crf))
                if group is None or group["frames"] != all_ids or set(group["by_shard"]) != set(range(num_shards)):
                    raise ValueError(f"Incomplete paired frame coverage: {variant}/{method}/CRF{crf}")
                if reference_partition is None:
                    reference_partition = group["by_shard"]
                elif reference_partition != group["by_shard"]:
                    raise ValueError("Physical frame partition differs across methods, CRFs, or variants")
                recs_sorted = sorted(group["records"], key=lambda row: tuple(row["frame_ids"]))
                clip_keys = [tuple(r["frame_ids"]) for r in recs_sorted]
                if reference_clip_keys is None:
                    reference_clip_keys = clip_keys
                elif reference_clip_keys != clip_keys:
                    raise ValueError("Clip boundary alignment differs across methods, CRFs, or variants")
                sorted_records_map[(variant, method, crf)] = recs_sorted

    assert reference_clip_keys is not None
    num_clips_total = len(reference_clip_keys)
    rng = np.random.default_rng(bootstrap_seed)
    if n_bootstrap > 0 and num_clips_total >= 2:
        probs = np.full(num_clips_total, 1.0 / float(num_clips_total), dtype=np.float64)
        boot_weights = rng.multinomial(num_clips_total, probs, size=n_bootstrap).astype(np.float64)
    elif n_bootstrap > 0 and num_clips_total == 1:
        boot_weights = np.ones((1, 1), dtype=np.float64)
    else:
        boot_weights = None

    # Precompute resampled curves only for distinct (variant, method) curves (caching shared anchor/blur/spatial)
    boot_curve_cache: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}

    def _get_boot_curve(variant_name: str, method_name: str) -> Optional[Dict[str, np.ndarray]]:
        if boot_weights is None:
            return None
        cache_key = (
            ("shared", method_name)
            if method_name in ("anchor_x265", "deterministic_blur_ema", "spatial_scope_framewise")
            else (variant_name, method_name)
        )
        if cache_key not in boot_curve_cache:
            prepared = _PreparedClipBootstrapCurve(
                {crf: sorted_records_map[(variant_name, method_name, crf)] for crf in CRFS},
                CRFS,
            )
            boot_curve_cache[cache_key] = prepared.evaluate_resampled_curves(boot_weights)
        return boot_curve_cache[cache_key]

    summaries: Dict[str, Dict[str, Any]] = {}
    ablation_boot_samples: Dict[str, Dict[str, np.ndarray]] = {}
    for variant in VARIANTS:
        methods_per_qp: Dict[str, Dict[str, Any]] = {}
        for method in METHODS:
            methods_per_qp[method] = {
                str(crf): summarize_records(sorted_records_map[(variant, method, crf)])
                for crf in CRFS
            }
        anchor_boot = _get_boot_curve(variant, "anchor_x265")
        variant_summary: Dict[str, Any] = {}
        for method, per_qp in methods_per_qp.items():
            method_boot = _get_boot_curve(variant, method)
            pair_boot = (anchor_boot, method_boot) if (anchor_boot is not None and method_boot is not None) else None
            bdbr_dict, b_samples = build_bdbr_summary(per_qp, methods_per_qp["anchor_x265"], CRFS, bootstrap_curves=pair_boot)
            variant_summary[method] = bdbr_dict
            if method == "tscope_temporal_ctu_qp":
                ablation_boot_samples[variant] = b_samples
        summaries[variant] = variant_summary

    overall_methods_summary = summaries["full_tscope"]
    ablation_variants_summary = {variant: summaries[variant]["tscope_temporal_ctu_qp"] for variant in VARIANTS}

    # Deterministic per-sequence disaggregation inside merged_benchmarks.json (F09 fix)
    seq_names = sorted({_sequence_prefix(fid) for fid in expected_ids})
    per_sequence_summary: Dict[str, Any] = {}
    for seq_name in seq_names:
        seq_clip_indices = [
            idx for idx, fids in enumerate(reference_clip_keys) if _sequence_prefix(fids[0]) == seq_name
        ]
        seq_frame_count = sum(len(reference_clip_keys[idx]) for idx in seq_clip_indices)
        if n_bootstrap > 0 and len(seq_clip_indices) >= 2:
            k_seq = len(seq_clip_indices)
            seq_weights = rng.multinomial(k_seq, np.full(k_seq, 1.0 / float(k_seq)), size=n_bootstrap).astype(np.float64)
        elif n_bootstrap > 0 and len(seq_clip_indices) == 1:
            seq_weights = np.ones((1, 1), dtype=np.float64)
        else:
            seq_weights = None

        seq_methods_per_qp: Dict[str, Dict[str, Any]] = {}
        seq_boot_cache: Dict[Tuple[str, str], Optional[Dict[str, np.ndarray]]] = {}

        def _seq_boot(v_name: str, m_name: str) -> Optional[Dict[str, np.ndarray]]:
            if seq_weights is None:
                return None
            ck = (
                ("shared", m_name)
                if m_name in ("anchor_x265", "deterministic_blur_ema", "spatial_scope_framewise")
                else (v_name, m_name)
            )
            if ck not in seq_boot_cache:
                sub_recs = {
                    crf: [sorted_records_map[(v_name, m_name, crf)][i] for i in seq_clip_indices]
                    for crf in CRFS
                }
                seq_boot_cache[ck] = _PreparedClipBootstrapCurve(sub_recs, CRFS).evaluate_resampled_curves(seq_weights)
            return seq_boot_cache[ck]

        for method in METHODS:
            seq_methods_per_qp[method] = {
                str(crf): summarize_records(
                    [sorted_records_map[("full_tscope", method, crf)][i] for i in seq_clip_indices]
                )
                for crf in CRFS
            }
        seq_anc_boot = _seq_boot("full_tscope", "anchor_x265")
        seq_methods_out: Dict[str, Any] = {}
        for method, pqp in seq_methods_per_qp.items():
            m_boot = _seq_boot("full_tscope", method)
            pair_b = (seq_anc_boot, m_boot) if (seq_anc_boot is not None and m_boot is not None) else None
            seq_methods_out[method], _ = build_bdbr_summary(pqp, seq_methods_per_qp["anchor_x265"], CRFS, bootstrap_curves=pair_b)

        seq_variants_out: Dict[str, Any] = {}
        for variant in VARIANTS:
            v_pqp = {
                str(crf): summarize_records(
                    [sorted_records_map[(variant, "tscope_temporal_ctu_qp", crf)][i] for i in seq_clip_indices]
                )
                for crf in CRFS
            }
            v_boot = _seq_boot(variant, "tscope_temporal_ctu_qp")
            pair_b = (seq_anc_boot, v_boot) if (seq_anc_boot is not None and v_boot is not None) else None
            seq_variants_out[variant], _ = build_bdbr_summary(v_pqp, seq_methods_per_qp["anchor_x265"], CRFS, bootstrap_curves=pair_b)

        per_sequence_summary[seq_name] = {
            "frame_count": seq_frame_count,
            "clip_count": len(seq_clip_indices),
            "overall_methods_summary": seq_methods_out,
            "ablation_variants_summary": seq_variants_out,
        }

    claim_audit = build_claim_falsification_audit(
        ablation_variants_summary, overall_methods_summary, ablation_boot_samples
    )

    manifest_sha = sha256_file(manifest_path)
    result = {
        "schema_version": SCHEMA_VERSION,
        "crfs": list(CRFS),
        "eligibility": {
            "final_ready": True,
            "expected_shards": num_shards,
            "complete_shard_variants": len(expected_pairs),
            "frame_count": len(expected_ids),
            "clip_count": num_clips_total,
        },
        "configuration": common,
        "checkpoint_hashes": hashes,
        "expected_frame_ids": expected_ids,
        "overall_methods_summary": overall_methods_summary,
        "ablation_variants_summary": ablation_variants_summary,
        "per_sequence_summary": per_sequence_summary,
        "claim_falsification_audit": claim_audit,
        "variant_methods_summary": summaries,
        "source_files": list(bundles.values()),
        "source_manifest_sha256": manifest_sha,
    }

    destination = Path(output_path) if output_path else Path(eval_dir) / "merged_benchmarks.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)

    per_seq_dest = destination.parent / "per_sequence_benchmarks.json"
    if not per_seq_dest.exists():
        with per_seq_dest.open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": SCHEMA_VERSION,
                    "crfs": list(CRFS),
                    "source_manifest_sha256": manifest_sha,
                    "merged_benchmarks_sha256": sha256_file(destination),
                    "per_sequence_summary": per_sequence_summary,
                },
                handle,
                indent=2,
                allow_nan=False,
            )

    print("\n=== PAIRED DATASET-LEVEL PEDESTRIAN AP / PCHIP BDBR + 10K CLIP-BOOTSTRAP 95% CIs ===")
    for variant, summary in result["ablation_variants_summary"].items():
        tot_v = summary["bdbr_tot_gt_ap50_pct"]
        p_v = summary["bdbr_p_gt_ap50_pct"]
        tot_ci = summary.get("bdbr_tot_gt_ap50_pct_ci95")
        p_ci = summary.get("bdbr_p_gt_ap50_pct_ci95")
        tot_str = (
            f"{tot_v:+.2f}% [{tot_ci[0]:+.2f}, {tot_ci[1]:+.2f}]"
            if (tot_v is not None and tot_ci is not None)
            else (f"{tot_v:+.2f}%" if tot_v is not None else "N/A")
        )
        p_str = (
            f"{p_v:+.2f}% [{p_ci[0]:+.2f}, {p_ci[1]:+.2f}]"
            if (p_v is not None and p_ci is not None)
            else (f"{p_v:+.2f}%" if p_v is not None else "N/A")
        )
        print(f"{variant:<24} | Total BDBR: {tot_str:<30} | P-Frame BDBR: {p_str}")
    print(f"Verified final benchmark -> {destination}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Shard Checkpoints & Eval JSONs (T-SCOPE v3)")
    parser.add_argument("--eval_dir", type=str, default="./outputs")
    parser.add_argument("--ckpt_dir", type=str, default="./outputs")
    parser.add_argument("--merged_ckpt", type=str, default="./outputs/tscope_merged_soup.pt")
    parser.add_argument("--manifest", required=True, help="Checked isolated collection manifest")
    parser.add_argument("--action", choices=["soup", "eval"], required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=None)
    parser.add_argument(
        "--expected-shards",
        "--expected_shards",
        dest="expected_shards",
        type=int,
        default=None,
        help="Optional explicit shard count check (defaults to manifest expected_shards)",
    )
    parser.add_argument("--bootstrap_resamples", type=int, default=10000)
    args = parser.parse_args()

    if args.action == "eval":
        merge_eval_jsons_and_print_table(
            args.eval_dir,
            args.manifest,
            expected_shards=args.expected_shards,
            n_bootstrap=args.bootstrap_resamples,
        )
    else:
        merge_checkpoints_soup(
            args.ckpt_dir,
            args.merged_ckpt,
            args.manifest,
            args.variants,
            expected_shards=args.expected_shards,
        )


if __name__ == "__main__":
    main()
