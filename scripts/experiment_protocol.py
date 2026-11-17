"""Versioned contracts for the T-SCOPE v3 Q1 VCM experiment suite."""
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

SCHEMA_VERSION = 3
CRFS = [27, 32, 37, 42]
IMG_H = 384
IMG_W = 640
EVAL_IMAGE_SIZE: Tuple[int, int] = (IMG_H, IMG_W)
TRAIN_CLIP_LEN = 4
EVAL_GOP_LEN = 32

# Strictly de-confounded 5-variant ablation design (F07 & F15 fix: equal shard counts per variant).
VARIANTS = [
    "full_tscope",
    "wo_ctu_delta_qp",
    "spatial_only_scope",
    "iwa_gop_qp_reuse",
    "wo_x265_ste_proxy",
]

VARIANT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "full_tscope": {
        "lambda_temp_bg": 1.0,
        "lambda_temp_obj": 0.5,
        "max_delta_qp": 4.0,
        "use_real_x265_ste": True,
        "reuse_iframe_qp": False,
        "disable_temporal": False,
    },
    "wo_ctu_delta_qp": {
        "lambda_temp_bg": 1.0,
        "lambda_temp_obj": 0.5,
        "max_delta_qp": 0.0,
        "use_real_x265_ste": True,
        "reuse_iframe_qp": False,
        "disable_temporal": False,
    },
    "spatial_only_scope": {
        "lambda_temp_bg": 0.0,
        "lambda_temp_obj": 0.0,
        "max_delta_qp": 0.0,
        "use_real_x265_ste": True,
        "reuse_iframe_qp": False,
        "disable_temporal": True,
    },
    "iwa_gop_qp_reuse": {
        "lambda_temp_bg": 1.0,
        "lambda_temp_obj": 0.5,
        "max_delta_qp": 4.0,
        "use_real_x265_ste": True,
        "reuse_iframe_qp": True,
        "disable_temporal": False,
    },
    "wo_x265_ste_proxy": {
        "lambda_temp_bg": 1.0,
        "lambda_temp_obj": 0.5,
        "max_delta_qp": 4.0,
        "use_real_x265_ste": False,
        "reuse_iframe_qp": False,
        "disable_temporal": False,
    },
}

METHODS = [
    "anchor_x265",
    "deterministic_blur_ema",
    "spatial_scope_framewise",
    "tscope_temporal_ctu_qp",
    "oracle_gt_mask_upper_bound",
]

TRAIN_SEQUENCES = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-10"]
VAL_SEQUENCES = ["MOT17-09"]
MOT17_TEST_SEQUENCE_LENGTHS: Dict[str, int] = {
    "MOT17-11": 900,
    "MOT17-13": 750,
}


def expected_mot17_test_frames(clip_len: int = EVAL_GOP_LEN) -> Tuple[List[str], List[str]]:
    """Compute deterministic complete-clip held-out frame IDs and excluded tail frame IDs."""
    if type(clip_len) is not int or clip_len < 1:
        raise ValueError(f"clip_len must be a positive int, got {clip_len!r}")
    expected: List[str] = []
    excluded: List[str] = []
    for seq_name in ("MOT17-11", "MOT17-13"):
        total_frames = MOT17_TEST_SEQUENCE_LENGTHS[seq_name]
        usable = (total_frames // clip_len) * clip_len
        expected.extend(f"{seq_name}/{num:06d}" for num in range(1, usable + 1))
        excluded.extend(f"{seq_name}/{num:06d}" for num in range(usable + 1, total_frames + 1))
    return expected, excluded


EXPECTED_TEST_FRAME_IDS, EXCLUDED_TEST_TAIL_FRAME_IDS = expected_mot17_test_frames(EVAL_GOP_LEN)

PREPROCESSING = {
    "resize": "bilinear_aspect_preserving_16_9",
    "img_h": IMG_H,
    "img_w": IMG_W,
    "clip_len": EVAL_GOP_LEN,
    "stride": EVAL_GOP_LEN,
    "tail_policy": "complete_nonoverlapping_clips_only",
}

VALIDATION_PROTOCOL = {
    "full_clips": 131,
    "val_subsample": 2,
    "batch_size": 4,
    "severities": [0.2, 0.6],
    "img_h": IMG_H,
    "img_w": IMG_W,
    "clip_len": TRAIN_CLIP_LEN,
    "uniform_codec": "libx265_real_ste_yuv420p",
    "teacher": "fasterrcnn_resnet50_fpn_v2_single_teacher",
    "aggregation": "uniform_real_x265_criterion_plus_constituent_connectivity",
    "held_out_test_used": False,
}

CODEC_CONFIGURATION = {
    "encoder": "libx265",
    "supported_families": {"0": "libx265", "1": "libx264"},
    "pix_fmt": "yuv420p",
    "preset": "medium",
    "rate_control": "crf",
    "fps": 25,
    "aq_mode": 2,
    "bframes": 0,
    "ra_bframes": 4,
    "gop_len": EVAL_GOP_LEN,
    "qg_size": 16,
    "actual_roi_grid": [IMG_H // 16, IMG_W // 16],
    "positive_qoffset_cap": 0.14,
    "negative_qoffset_cap": -0.18,
    "bitrate_accounting": "mp4_video_packet_payload_excluding_container_and_global_headers",
}

METRIC_POLICY = {
    "name": "dataset_level_101_point_pedestrian_ap",
    "class_id": 1,
    "score_threshold": 0.01,
    "iou_thresholds": [round(0.50 + 0.05 * i, 2) for i in range(10)],
    "ignore_policy": "unmatched_prediction_suppressed_if_ignore_box_iou_at_evaluation_threshold",
    "bdbr_interpolator": "scipy.interpolate.PchipInterpolator_log10_rate",
    "bootstrap_resamples": 10000,
    "full_official_coco_equivalence": False,
}

# Versioned secondary VCM dataset & MPEG-VCM CTC protocol contract (F11 & F25 fix).
SFU_CTC_PROTOCOL = {
    "schema_version": SCHEMA_VERSION,
    "datasets": ["MOT20", "KITTI", "SFU-HW-Objects-v1"],
    "kaggle_slugs": {
        "mot17": "ahmedsamir1598/mot17challenge",
        "mot20": "ismailelbouknify/mot-20",
        "kitti": "leducnhuan/kitti-tracking",
        "hevc_b": "minhngt02/hevc-b",
    },
    "pix_fmt": "yuv420p",
    "gop_len": EVAL_GOP_LEN,
    "crfs": list(CRFS),
    "modes": {
        "low_delay_p": {"bframes": 0, "keyint": EVAL_GOP_LEN},
        "random_access": {"bframes": 4, "keyint": EVAL_GOP_LEN},
    },
}

# Pre-registered quantitative claim falsification criteria (F18 fix).
CLAIM_FALSIFICATION_CRITERIA: Dict[str, Dict[str, Any]] = {
    "C01_total_bdbr_vs_anchor": {
        "description": "Full T-SCOPE achieves statistically significant negative Total BDBR vs unedited libx265 on Faster R-CNN AP@50.",
        "metric": "bdbr_tot_gt_ap50_pct",
        "variant": "full_tscope",
        "comparator": "anchor_x265",
        "threshold_pct": 0.0,
        "require_ci95_upper_below": 0.0,
    },
    "C02_long_gop_p_frame_rate_reduction": {
        "description": "Motion-compensated temporal pre-editing (wo_ctu_delta_qp) achieves lower P-frame BDBR than framewise spatial_only_scope on 32-frame GOPs.",
        "metric": "bdbr_p_gt_ap50_pct",
        "variant": "wo_ctu_delta_qp",
        "comparator": "spatial_only_scope",
        "threshold_pct": 0.0,
        "require_ci95_upper_below": 0.0,
    },
    "C03_ctu_delta_qp_gain": {
        "description": "Full T-SCOPE (with 16x16 QG quantOffsets + P-carry) lowers Total BDBR relative to pixel-only wo_ctu_delta_qp.",
        "metric": "bdbr_tot_gt_ap50_pct",
        "variant": "full_tscope",
        "comparator": "wo_ctu_delta_qp",
        "threshold_pct": 0.0,
        "require_ci95_upper_below": 0.0,
    },
    "C04_real_codec_ste_gain": {
        "description": "Hybrid real-codec STE training (full_tscope) lowers Total BDBR relative to proxy-only wo_x265_ste_proxy under identical hyperparameters.",
        "metric": "bdbr_tot_gt_ap50_pct",
        "variant": "full_tscope",
        "comparator": "wo_x265_ste_proxy",
        "threshold_pct": 0.0,
        "require_ci95_upper_below": 0.0,
    },
    "C05_zero_shot_unseen_yolov8s_transfer": {
        "description": "Single-teacher Faster R-CNN-trained T-SCOPE transfers zero-shot to unseen YOLOv8s with negative Total BDBR.",
        "metric": "bdbr_tot_yolo_gt_ap50_pct",
        "variant": "full_tscope",
        "comparator": "anchor_x265",
        "threshold_pct": 0.0,
        "require_ci95_upper_below": 0.0,
    },
}


def sha256_file(path: Any) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_manifest_shard_directory(manifest_parent: Path, raw_dir: str) -> Path:
    """Resolve collection-relative shard directory safely without hardcoded author machine paths (F22 fix)."""
    if not isinstance(raw_dir, str) or not raw_dir.strip():
        raise ValueError("Missing shard directory in collection manifest")
    raw_path = Path(raw_dir)
    if ".." in raw_path.parts:
        raise ValueError("Unsafe parent traversal in manifest shard directory")
    if not raw_path.is_absolute() and ":" not in raw_dir:
        candidate = (manifest_parent / raw_path).resolve()
        if not candidate.is_relative_to(manifest_parent):
            raise ValueError("Manifest directory escapes collection")
        return candidate
    # Relocated-checkout fallback for legacy absolute paths whose leaf folder sits inside manifest_parent
    resolved_abs = raw_path.resolve()
    if resolved_abs.is_relative_to(manifest_parent) and resolved_abs.is_dir():
        return resolved_abs
    relocated = (manifest_parent / raw_path.name).resolve()
    if relocated.is_relative_to(manifest_parent) and relocated.is_dir():
        return relocated
    raise ValueError("Manifest directory escapes collection")


def read_manifest_files(
    manifest_path: Any,
    suffix: str,
    expected_shards: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[Tuple[Dict[str, Any], Path, str]]]:
    """Read and hash-verify manifested shard artifacts with parameterized shard count (F22 fix)."""
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest_expected = manifest.get("expected_shards")
    if type(manifest_expected) is not int or manifest_expected < 1 or not isinstance(manifest.get("shards"), list):
        raise ValueError("Manifest requires a positive integer expected_shards and shards list")
    if expected_shards is not None and manifest_expected != expected_shards:
        raise ValueError(f"Expected a {expected_shards}-shard collection manifest, got {manifest_expected}")

    files: List[Tuple[Dict[str, Any], Path, str]] = []
    seen_shards = set()
    for entry in manifest["shards"]:
        sid = entry.get("shard_id")
        if type(sid) is not int or not 0 <= sid < manifest_expected or sid in seen_shards:
            raise ValueError("Invalid or duplicate manifested shard")
        seen_shards.add(sid)
        if not entry.get("downloaded"):
            continue
        if str(entry.get("status", "")).upper().split(".")[-1] != "COMPLETE":
            raise ValueError(f"Shard {sid} is not complete")
        directory = _resolve_manifest_shard_directory(path.parent, entry.get("directory", ""))
        seen_files = set()
        for item in entry.get("files", []):
            relative = Path(item["path"])
            candidate = directory / relative
            if (
                relative.is_absolute()
                or ":" in str(relative)
                or ".." in relative.parts
                or candidate.is_symlink()
                or not candidate.resolve().is_relative_to(directory)
            ):
                raise ValueError("Unsafe manifested artifact path")
            if str(relative) in seen_files:
                raise ValueError("Duplicate manifested file")
            seen_files.add(str(relative))
            if candidate.suffix != suffix:
                continue
            if (
                not candidate.is_file()
                or candidate.stat().st_size != item.get("bytes")
                or sha256_file(candidate) != item.get("sha256")
            ):
                raise ValueError(f"Artifact hash or size mismatch: {candidate.name}")
            files.append((entry, candidate, item["sha256"]))
    return manifest, files


def verify_convergence_log(
    log: str,
    sid: int,
    variant: str,
    num_shards: Optional[int] = None,
) -> Dict[str, bool]:
    """Verify exact shard convergence and completion identity for parameterized shard topology."""
    if num_shards is not None:
        expected_finish = f"=== Shard {sid}/{num_shards} ({variant}) finished successfully! ==="
        expected_saved = (
            "Saved shard checkpoint" in log
            and f"tscope_{variant}_shard{sid:02d}of{num_shards:02d}.pt" in log
        )
        finish_ok = expected_finish in log or expected_saved
    else:
        finish_re = re.search(rf"=== Shard {sid}/\d+ \({re.escape(variant)}\) finished successfully! ===", log)
        saved_re = re.search(rf"Saved shard checkpoint.*tscope_{re.escape(variant)}_shard{sid:02d}of\d+\.pt", log)
        finish_ok = bool(finish_re or saved_re)

    if "Converged: no val improvement for " not in log or not finish_ok:
        raise ValueError("Log does not corroborate exact shard convergence and completion")
    return {"startup_visible": f"Variant={variant}" in log, "completion_identity_verified": True}


def validate_checkpoint(
    checkpoint: Dict[str, Any],
    sid: int,
    variant: str,
    expected_num_shards: Optional[int] = None,
):
    from models.temporal_scope import TemporalSCOPE

    if variant not in VARIANT_CONFIGS:
        raise ValueError(f"Unknown ablation variant: {variant!r}")
    num_shards = checkpoint.get("num_shards")
    if (
        checkpoint.get("shard_id") != sid
        or type(num_shards) is not int
        or num_shards < 1
        or (expected_num_shards is not None and num_shards != expected_num_shards)
        or checkpoint.get("variant") != variant
    ):
        raise ValueError("Checkpoint identity mismatch")
    if checkpoint.get("partial", False) or checkpoint.get("stopped_early") is not True:
        raise ValueError("Final soup requires a completed early-stopped checkpoint")
    if checkpoint.get("train_sequences") != TRAIN_SEQUENCES or checkpoint.get("val_sequences") != VAL_SEQUENCES:
        raise ValueError("Checkpoint split provenance mismatch")

    train_cfg = checkpoint.get("training_configuration", {})
    if train_cfg.get("use_yolov8s", False) is not False:
        raise ValueError("F04 violation: training checkpoint must use single-teacher Faster R-CNN only (use_yolov8s=False)")

    history = checkpoint.get("history", [])
    epochs, best_epoch = checkpoint.get("epochs_run", 0), checkpoint.get("best_epoch", 0)
    best_val = checkpoint.get("best_val_loss")
    min_patience = int(train_cfg.get("patience", 10) or 10)
    min_epochs = int(train_cfg.get("min_epochs", 12) or 12)
    if (
        type(epochs) is not int
        or type(best_epoch) is not int
        or len(history) != epochs
        or epochs < min_epochs
        or not 1 <= best_epoch <= epochs
        or epochs - best_epoch < min_patience
    ):
        raise ValueError("Invalid convergence history or patience")
    if not isinstance(best_val, (float, int)) or not math.isfinite(best_val):
        raise ValueError("Missing finite best validation loss")
    if [row.get("epoch") for row in history] != list(range(1, epochs + 1)):
        raise ValueError("Non-contiguous checkpoint history")
    if not math.isclose(history[best_epoch - 1].get("val_loss", float("nan")), best_val, abs_tol=1e-6):
        raise ValueError("Best validation loss differs from history")
    if any(
        not math.isfinite(row.get("loss_total", float("nan")))
        or ("val_loss" in row and not math.isfinite(row["val_loss"]))
        for row in history
    ):
        raise ValueError("Nonfinite checkpoint history")

    scale = float(VARIANT_CONFIGS[variant]["max_delta_qp"])
    num_families = int(train_cfg.get("model", {}).get("num_families", 2))
    if num_families != 2:
        raise ValueError(f"T-SCOPE v3 requires num_families=2 (libx265, libx264), got {num_families}")
    model = TemporalSCOPE(base_ch=48, num_families=num_families, cond_dim=64, qg_size=16, max_delta_qp=scale)
    state = checkpoint.get("state_dict", {})
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("Checkpoint parameter keys mismatch")
    for key, value in state.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != expected[key].shape
            or value.dtype != expected[key].dtype
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"Invalid checkpoint tensor: {key}")
    model.load_state_dict(state, strict=True)
    return model


def validate_soup_provenance(
    soup: Dict[str, Any],
    variant: str,
    require_validation: bool = False,
    expected_shards_per_variant: Optional[int] = None,
) -> Dict[str, Any]:
    if soup.get("schema_version") != SCHEMA_VERSION or soup.get("variant") != variant or variant not in VARIANTS:
        raise ValueError("Invalid same-variant soup schema (requires schema_version=3)")
    members = soup.get("constituents")
    if not isinstance(members, list) or len(members) < 1 or soup.get("merged_shards") != len(members):
        raise ValueError("Soup constituent membership is incomplete")
    if expected_shards_per_variant is not None and len(members) != expected_shards_per_variant:
        raise ValueError(
            f"Balanced shard contract requires {expected_shards_per_variant} shards for {variant}, got {len(members)}"
        )
    ids = [member.get("shard_id") for member in members if isinstance(member, dict)]
    if len(ids) != len(members) or len(set(ids)) != len(ids) or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("Soup constituent shard identities are invalid or duplicated")
    for member in members:
        for key in ("checkpoint_sha256", "log_sha256"):
            value = member.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Missing constituent hash provenance")
    if soup.get("strict_load_passed") is not True or soup.get("inference_smoke_passed") is not True:
        raise ValueError("Soup lacks strict load and inference smoke evidence")
    configuration = soup.get("configuration", {})
    if configuration.get("codec") != CODEC_CONFIGURATION or configuration.get("initialization_seed") != 2026:
        raise ValueError("Soup configuration differs from v3 training protocol")
    if configuration.get("use_yolov8s_in_training", False) is not False:
        raise ValueError("Soup violates single-teacher Faster R-CNN training contract (F04)")
    expected_scale = float(VARIANT_CONFIGS[variant]["max_delta_qp"])
    if configuration.get("max_delta_qp") != expected_scale:
        raise ValueError("Soup QG scale differs from v3 variant configuration")
    if require_validation:
        validate_release_validation(soup.get("validation", {}))
    return soup


def validate_release_validation(validation: Dict[str, Any]) -> Dict[str, Any]:
    if (
        validation.get("status") != "validated"
        or validation.get("release_ready") is not True
        or validation.get("split") != "MOT17-09"
    ):
        raise ValueError("Soup must pass MOT17-09 validation before final evaluation")
    uniform_loss = validation.get("uniform_loss_total", validation.get("loss_total"))
    if type(uniform_loss) not in (int, float) or not math.isfinite(uniform_loss):
        raise ValueError("Missing finite measured uniform soup validation loss")
    if validation.get("protocol") != VALIDATION_PROTOCOL:
        raise ValueError("Soup validation protocol differs from v3 protocol")
    fingerprints = validation.get("data_fingerprints", {})
    frames = fingerprints.get("frames", [])
    if not isinstance(frames, list) or [row.get("frame") for row in frames if isinstance(row, dict)] != list(range(1, 525)):
        raise ValueError("Soup validation requires the complete MOT17-09 frame fingerprints")
    hashes = [fingerprints.get("gt_sha256"), validation.get("source_soup_sha256")] + [row.get("sha256") for row in frames]
    if any(not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in hashes):
        raise ValueError("Invalid validation data or source soup hashes")
    if validation.get("selection") not in (
        "uniform_soup_no_test_selection",
        "greedy_soup_val_mot17_09_no_test_selection",
    ):
        raise ValueError("Validation must not use held-out test selection")
    return validation


def restore_raw_detections(raw: Dict[str, Any], count: int) -> Dict[str, List[torch.Tensor]]:
    result: Dict[str, List[torch.Tensor]] = {}
    required_keys = ("gt_boxes", "ignore_boxes", "pred_boxes", "pred_scores")
    optional_keys: List[str] = []
    if "yolo_pred_boxes" in raw or "yolo_pred_scores" in raw:
        optional_keys.extend(["yolo_pred_boxes", "yolo_pred_scores"])
    if "gt_track_ids" in raw:
        optional_keys.append("gt_track_ids")
    if "gt_classes" in raw:
        optional_keys.append("gt_classes")

    for key in required_keys + tuple(optional_keys):
        values = raw.get(key)
        if not isinstance(values, list) or len(values) != count:
            raise ValueError(f"Invalid raw detection frame count: {key}")
        tensors = []
        for value in values:
            if key in ("gt_track_ids", "gt_classes"):
                tensor = torch.tensor(value, dtype=torch.long)
                if tensor.ndim != 1:
                    raise ValueError(f"Invalid 1D integer metadata tensor: {key}")
            elif key in ("pred_scores", "yolo_pred_scores"):
                tensor = torch.tensor(value, dtype=torch.float32)
                if tensor.ndim != 1 or not bool(torch.isfinite(tensor).all()) or bool(((tensor < 0) | (tensor > 1)).any()):
                    raise ValueError("Invalid prediction scores")
            else:
                tensor = torch.tensor(value, dtype=torch.float32)
                if tensor.numel() == 0 and isinstance(value, list) and not value:
                    tensor = tensor.reshape(0, 4)
                if tensor.ndim != 2 or tensor.shape[1] != 4 or not bool(torch.isfinite(tensor).all()):
                    raise ValueError("Invalid box coordinates")
                if bool(((tensor[:, 2:] - tensor[:, :2]) <= 0).any()) or bool((tensor < 0).any()):
                    raise ValueError("Degenerate or negative box coordinates")
            tensors.append(tensor)
        result[key] = tensors

    if any(boxes.shape[0] != scores.shape[0] for boxes, scores in zip(result["pred_boxes"], result["pred_scores"])):
        raise ValueError("Prediction boxes and scores differ in count")
    if "yolo_pred_boxes" in result and any(
        boxes.shape[0] != scores.shape[0] for boxes, scores in zip(result["yolo_pred_boxes"], result["yolo_pred_scores"])
    ):
        raise ValueError("YOLO prediction boxes and scores differ in count")
    if "gt_track_ids" in result and any(
        boxes.shape[0] != tids.shape[0] for boxes, tids in zip(result["gt_boxes"], result["gt_track_ids"])
    ):
        raise ValueError("GT boxes and GT track IDs differ in count")
    if "gt_classes" in result and any(
        boxes.shape[0] != clss.shape[0] for boxes, clss in zip(result["gt_boxes"], result["gt_classes"])
    ):
        raise ValueError("GT boxes and GT classes differ in count")
    return result
