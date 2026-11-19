#!/usr/bin/env python3
"""
Reproduce & verify official `miccunifi/SCOPE` (Dani et al., 2026) on JPEG compression:
  1. `--mode coco_jpeg`: Exact `miccunifi/SCOPE/configs/eval.yaml` protocol on COCO 2017 validation
     (`Q in {10, 15, 20, 30, 40, 50}`, `codec_family='block_dct' -> 'diff_jpeg'`, exact YAML severities,
     `cv2.imencode` JPEG round-trip, `pycocotools` COCOeval AP@[.50:.95] & AP@50, Pareto + PCHIP BDBR).
  2. `--mode mot17_bridge`: Controlled diagnostic bridge on the exact 1,632 held-out MOT17 frames (640x384),
     comparing:
       (a) Direct JPEG vs. Official SCOPE + JPEG (`Q in {10, 15, 20, 30, 40, 50}`),
       (b) Direct libx265 All-Intra (`GOP=1`, `keyint=1`) vs. Official SCOPE + libx265 All-Intra (`CRF in {27,32,37,42}`).
     Isolates domain shift (COCO -> MOT17) from codec regime shift (JPEG -> HEVC Intra -> HEVC 32-frame GOP).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import subprocess
import sys
import time
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from scipy.interpolate import PchipInterpolator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Exact JPEG quality & conditioning severity pairs from miccunifi/SCOPE/configs/eval.yaml (lines 80-135)
SCOPE_OFFICIAL_JPEG_SPEC: List[Tuple[int, float]] = [
    (10, 0.908163),
    (15, 0.857143),
    (20, 0.806122),
    (30, 0.704082),
    (40, 0.602041),
    (50, 0.500000),
]

# Exact COCO80 -> COCO91 category ID mapping from miccunifi/SCOPE/scope/evaluation/categories.py
COCO80_TO_COCO91: Tuple[int, ...] = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
)
VALID_COCO91_SET = set(COCO80_TO_COCO91)


def _pareto_monotone_curve(rate: Sequence[float], quality: Sequence[float], eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """Exact replica of miccunifi/SCOPE/scope/evaluation/ablation/curves.py::pareto_monotone_curve."""
    r = np.asarray(rate, dtype=np.float64)
    q = np.asarray(quality, dtype=np.float64)
    valid = np.isfinite(r) & np.isfinite(q) & (r > 0)
    r, q = r[valid], q[valid]
    order = np.argsort(r)
    r, q = r[order], q[order]
    kept_r: List[float] = []
    kept_q: List[float] = []
    best_q = -math.inf
    for ri, qi in zip(r, q):
        if qi > best_q + eps:
            kept_r.append(float(ri))
            kept_q.append(float(qi))
            best_q = float(qi)
    return np.asarray(kept_r, dtype=np.float64), np.asarray(kept_q, dtype=np.float64)


def pchip_bdbr(
    rate_anchor: Sequence[float],
    acc_anchor: Sequence[float],
    rate_test: Sequence[float],
    acc_test: Sequence[float],
    min_points: int = 4,
) -> Optional[float]:
    """Exact Pareto-filtered PCHIP BDBR (%) matching miccunifi/SCOPE/scope/evaluation/ablation/curves.py::compute_bd_rate."""
    r1, a1 = _pareto_monotone_curve(rate_anchor, acc_anchor)
    r2, a2 = _pareto_monotone_curve(rate_test, acc_test)
    if len(r1) < min_points or len(r2) < min_points:
        return None
    lo, hi = float(max(a1[0], a2[0])), float(min(a1[-1], a2[-1]))
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo + 1e-6:
        return None
    i1 = float(PchipInterpolator(a1, np.log(r1)).integrate(lo, hi))
    i2 = float(PchipInterpolator(a2, np.log(r2)).integrate(lo, hi))
    return round(float((math.exp((i2 - i1) / (hi - lo)) - 1.0) * 100.0), 2)


def jpeg_encode_decode_bgr(img_bgr: np.ndarray, quality: int) -> Tuple[np.ndarray, int]:
    """Exact replica of miccunifi/SCOPE/scope/evaluation/utils.py::jpeg_encode_decode."""
    q = int(np.clip(quality, 1, 100))
    ok, encoded = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for quality={q}")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError(f"cv2.imdecode failed for quality={q}")
    return decoded, int(len(encoded))


@torch.no_grad()
def scope_preedit_bgr(
    img_bgr: np.ndarray,
    scope_model: torch.nn.Module,
    family_id: int,
    severity: float,
    device: torch.device,
) -> np.ndarray:
    """Exact replica of miccunifi/SCOPE/scope/evaluation/codec_transmissions.py::apply_scope_bgr_conditional."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(
        device=device, dtype=torch.float32
    )
    fam_t = torch.tensor([int(family_id)], device=device, dtype=torch.long)
    sev_t = torch.tensor([float(severity)], device=device, dtype=torch.float32)
    edited, *_ = scope_model(x, severity=sev_t, codec_family=fam_t)
    res_rgb = np.round(edited.detach().clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR)


def ensure_coco_val2017(coco_root: Path) -> Tuple[Path, Path]:
    """Locate mounted or download official COCO 2017 val images and instances_val2017.json."""
    search_roots = [coco_root, Path("/kaggle/input")]
    for sr in search_roots:
        if sr.exists():
            for ann in sr.rglob("instances_val2017.json"):
                for vdir in (
                    ann.parent.parent / "val2017",
                    ann.parent.parent / "images" / "val2017",
                    ann.parent / "val2017",
                ):
                    if vdir.is_dir() and len(list(vdir.glob("*.jpg"))) >= 4900:
                        print(f"[COCO] Found mounted COCO val2017 at {vdir} and {ann}", flush=True)
                        return vdir, ann

    coco_root.mkdir(parents=True, exist_ok=True)
    val_dir = coco_root / "val2017"
    ann_file = coco_root / "annotations" / "instances_val2017.json"

    if not (val_dir.is_dir() and len(list(val_dir.glob("*.jpg"))) >= 5000):
        zip_img = coco_root / "val2017.zip"
        print("[COCO] Downloading http://images.cocodataset.org/zips/val2017.zip ...", flush=True)
        subprocess.run(
            ["curl", "-L", "--retry", "3", "-o", str(zip_img), "http://images.cocodataset.org/zips/val2017.zip"],
            check=True,
        )
        with zipfile.ZipFile(zip_img, "r") as zf:
            zf.extractall(coco_root)
        zip_img.unlink(missing_ok=True)

    if not ann_file.is_file():
        zip_ann = coco_root / "annotations_trainval2017.zip"
        print("[COCO] Downloading http://images.cocodataset.org/annotations/annotations_trainval2017.zip ...", flush=True)
        subprocess.run(
            [
                "curl",
                "-L",
                "--retry",
                "3",
                "-o",
                str(zip_ann),
                "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            ],
            check=True,
        )
        with zipfile.ZipFile(zip_ann, "r") as zf:
            zf.extractall(coco_root)
        zip_ann.unlink(missing_ok=True)

    return val_dir, ann_file


def evaluate_coco_subset(coco_gt: Any, preds: List[Dict[str, Any]], img_ids: List[int]) -> Dict[str, float]:
    """Run official pycocotools COCOeval restricted to `img_ids`."""
    from pycocotools.cocoeval import COCOeval

    if not preds:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AP_small": 0.0, "AP_medium": 0.0, "AP_large": 0.0}
    with redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(preds)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.imgIds = list(img_ids)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    st = ev.stats
    return {
        "AP": round(float(st[0]) * 100.0, 3),
        "AP50": round(float(st[1]) * 100.0, 3),
        "AP75": round(float(st[2]) * 100.0, 3),
        "AP_small": round(float(st[3]) * 100.0, 3),
        "AP_medium": round(float(st[4]) * 100.0, 3),
        "AP_large": round(float(st[5]) * 100.0, 3),
    }


def load_ultralytics_yolo(preferred_names: Sequence[str]) -> Tuple[str, Any]:
    from ultralytics import YOLO

    last_err: Optional[Exception] = None
    for name in preferred_names:
        try:
            m = YOLO(name)
            print(f"[Detector] Loaded Ultralytics YOLO: {name}", flush=True)
            return name, m
        except Exception as exc:
            last_err = exc
            print(f"[Detector] Could not load {name} ({exc}), trying fallback...", flush=True)
    raise RuntimeError(f"Failed to load any Ultralytics model from {preferred_names}: {last_err}")


def run_coco_jpeg_verification(args: argparse.Namespace, scope_bundle: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    from pycocotools.coco import COCO

    val_dir, ann_file = ensure_coco_val2017(Path(args.coco_root))
    with redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(ann_file))

    all_img_ids = sorted(coco_gt.getImgIds())
    if args.max_images > 0 and args.max_images < len(all_img_ids):
        img_ids = all_img_ids[: args.max_images]
    else:
        img_ids = all_img_ids

    scope_model = scope_bundle["model"]
    fam_map = scope_bundle["fam_map"]
    jpeg_fam_name = "diff_jpeg" if "diff_jpeg" in fam_map else next(iter(fam_map))
    jpeg_fam_id = int(fam_map[jpeg_fam_name])
    print(f"[SCOPE] Loaded fam_map={fam_map}, using JPEG family={jpeg_fam_name!r} (id={jpeg_fam_id})", flush=True)

    req_dets = {x.strip().lower() for x in args.detectors.split(",") if x.strip()}
    frcnn = None
    yolo_name = None
    yolo_model = None
    rtdetr_model = None
    detectors: List[str] = []

    if "frcnn" in req_dets or "fasterrcnn" in req_dets:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

        # Exact TorchvisionCocoDetector settings from miccunifi/SCOPE/scope/evaluation/saved_multi_detector.py
        frcnn = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            box_score_thresh=0.001,
            box_nms_thresh=0.7,
            box_detections_per_img=300,
        ).to(device).eval()
        detectors.append("fasterrcnn_resnet50_fpn_v2")
        print("[Detector] Loaded TorchVision fasterrcnn_resnet50_fpn_v2 (conf=0.001, nms=0.7, max_det=300)", flush=True)

    if "yolo" in req_dets:
        yolo_name, yolo_model = load_ultralytics_yolo(["yolo26x.pt", "yolo11x.pt", "yolov8x.pt"])
        detectors.append(yolo_name)

    if "rtdetr" in req_dets or args.include_rtdetr:
        from ultralytics import RTDETR

        rtdetr_model = RTDETR("rtdetr-l.pt")
        detectors.append("rtdetr-l.pt")
        print("[Detector] Loaded Ultralytics RTDETR: rtdetr-l.pt", flush=True)

    if not detectors:
        raise ValueError(f"No valid detectors requested in --detectors={args.detectors!r}")

    methods = ["jpeg_direct", "scope_jpeg"]
    preds: Dict[str, Dict[int, Dict[str, List[Dict[str, Any]]]]] = {
        m: {q: {d: [] for d in detectors} for q, _ in SCOPE_OFFICIAL_JPEG_SPEC} for m in methods
    }
    bits_sum: Dict[str, Dict[int, int]] = {m: {q: 0 for q, _ in SCOPE_OFFICIAL_JPEG_SPEC} for m in methods}
    pix_sum: Dict[str, Dict[int, int]] = {m: {q: 0 for q, _ in SCOPE_OFFICIAL_JPEG_SPEC} for m in methods}
    bpp_img_sum: Dict[str, Dict[int, float]] = {m: {q: 0.0 for q, _ in SCOPE_OFFICIAL_JPEG_SPEC} for m in methods}
    img_count: Dict[str, Dict[int, int]] = {m: {q: 0 for q, _ in SCOPE_OFFICIAL_JPEG_SPEC} for m in methods}

    t0 = time.perf_counter()
    batch_size = 8
    for b_start in range(0, len(img_ids), batch_size):
        sub_ids = img_ids[b_start : b_start + batch_size]
        raw_bgrs: List[np.ndarray] = []
        valid_ids: List[int] = []
        for iid in sub_ids:
            info = coco_gt.loadImgs([iid])[0]
            p = val_dir / info["file_name"]
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is not None:
                raw_bgrs.append(im)
                valid_ids.append(iid)

        for q, sev in SCOPE_OFFICIAL_JPEG_SPEC:
            dec_direct_bgr: List[np.ndarray] = []
            dec_scope_bgr: List[np.ndarray] = []

            for im_bgr in raw_bgrs:
                h, w = im_bgr.shape[:2]
                npix = max(1, int(h * w))
                # 1. Direct JPEG
                d_im, nbytes_d = jpeg_encode_decode_bgr(im_bgr, q)
                dec_direct_bgr.append(d_im)
                bits_sum["jpeg_direct"][q] += nbytes_d * 8
                pix_sum["jpeg_direct"][q] += npix
                bpp_img_sum["jpeg_direct"][q] += (float(nbytes_d) * 8.0) / float(npix)
                img_count["jpeg_direct"][q] += 1

                # 2. SCOPE + JPEG
                ed_im = scope_preedit_bgr(im_bgr, scope_model, jpeg_fam_id, sev, device)
                s_im, nbytes_s = jpeg_encode_decode_bgr(ed_im, q)
                dec_scope_bgr.append(s_im)
                bits_sum["scope_jpeg"][q] += nbytes_s * 8
                pix_sum["scope_jpeg"][q] += npix
                bpp_img_sum["scope_jpeg"][q] += (float(nbytes_s) * 8.0) / float(npix)
                img_count["scope_jpeg"][q] += 1

            for m_key, bgr_list in (("jpeg_direct", dec_direct_bgr), ("scope_jpeg", dec_scope_bgr)):
                if frcnn is not None:
                    with torch.no_grad():
                        fr_inputs = [
                            torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).to(device, dtype=torch.float32)
                            / 255.0
                            for im in bgr_list
                        ]
                        fr_outs = frcnn(fr_inputs)
                    for iid, out in zip(valid_ids, fr_outs):
                        boxes = out["boxes"].detach().cpu().numpy()
                        scores = out["scores"].detach().cpu().numpy()
                        labels = out["labels"].detach().cpu().numpy().astype(int)
                        for (x1, y1, x2, y2), sc_val, lb in zip(boxes, scores, labels):
                            if int(lb) in VALID_COCO91_SET:
                                preds[m_key][q]["fasterrcnn_resnet50_fpn_v2"].append(
                                    {
                                        "image_id": int(iid),
                                        "category_id": int(lb),
                                        "bbox": [float(x1), float(y1), max(0.0, float(x2 - x1)), max(0.0, float(y2 - y1))],
                                        "score": float(sc_val),
                                    }
                                )

                if yolo_model is not None and yolo_name is not None:
                    yo_res = yolo_model.predict(
                        source=bgr_list,
                        conf=0.001,
                        iou=0.7,
                        max_det=300,
                        verbose=False,
                        device=0 if device.type == "cuda" else "cpu",
                    )
                    for iid, r in zip(valid_ids, yo_res):
                        if r.boxes is not None and len(r.boxes) > 0:
                            b_xyxy = r.boxes.xyxy.cpu().numpy()
                            confs = r.boxes.conf.cpu().numpy()
                            clss = r.boxes.cls.cpu().numpy().astype(int)
                            for (x1, y1, x2, y2), sc_val, c_id in zip(b_xyxy, confs, clss):
                                if 0 <= c_id < len(COCO80_TO_COCO91):
                                    preds[m_key][q][yolo_name].append(
                                        {
                                            "image_id": int(iid),
                                            "category_id": int(COCO80_TO_COCO91[c_id]),
                                            "bbox": [float(x1), float(y1), max(0.0, float(x2 - x1)), max(0.0, float(y2 - y1))],
                                            "score": float(sc_val),
                                        }
                                    )

                if rtdetr_model is not None:
                    rt_res = rtdetr_model.predict(
                        source=bgr_list,
                        conf=0.001,
                        iou=0.7,
                        max_det=300,
                        verbose=False,
                        device=0 if device.type == "cuda" else "cpu",
                    )
                    for iid, r in zip(valid_ids, rt_res):
                        if r.boxes is not None and len(r.boxes) > 0:
                            b_xyxy = r.boxes.xyxy.cpu().numpy()
                            confs = r.boxes.conf.cpu().numpy()
                            clss = r.boxes.cls.cpu().numpy().astype(int)
                            for (x1, y1, x2, y2), sc_val, c_id in zip(b_xyxy, confs, clss):
                                if 0 <= c_id < len(COCO80_TO_COCO91):
                                    preds[m_key][q]["rtdetr-l.pt"].append(
                                        {
                                            "image_id": int(iid),
                                            "category_id": int(COCO80_TO_COCO91[c_id]),
                                            "bbox": [float(x1), float(y1), max(0.0, float(x2 - x1)), max(0.0, float(y2 - y1))],
                                            "score": float(sc_val),
                                        }
                                    )

        done_n = min(b_start + batch_size, len(img_ids))
        if done_n % 200 == 0 or done_n == len(img_ids):
            elapsed = time.perf_counter() - t0
            print(f"[COCO JPEG] Processed {done_n}/{len(img_ids)} images in {elapsed:.1f}s", flush=True)

    summary: Dict[str, Any] = {
        "protocol": "miccunifi/SCOPE configs/eval.yaml exact JPEG verification",
        "dataset": "COCO 2017 validation",
        "evaluated_images": len(img_ids),
        "scope_ckpt_sha256": scope_bundle["ckpt_sha256"],
        "scope_codec_family": jpeg_fam_name,
        "jpeg_qualities": [q for q, _ in SCOPE_OFFICIAL_JPEG_SPEC],
        "severities": {str(q): s for q, s in SCOPE_OFFICIAL_JPEG_SPEC},
        "detectors": {},
    }

    for det in detectors:
        det_block: Dict[str, Any] = {"jpeg_direct": {}, "scope_jpeg": {}}
        for m_key in methods:
            for q, _ in SCOPE_OFFICIAL_JPEG_SPEC:
                ds_bpp = bits_sum[m_key][q] / max(1, pix_sum[m_key][q])
                avg_bpp = bpp_img_sum[m_key][q] / max(1, img_count[m_key][q])
                ap_dict = evaluate_coco_subset(coco_gt, preds[m_key][q][det], img_ids)
                det_block[m_key][str(q)] = {
                    "bpp": round(ds_bpp, 5),
                    "avg_bpp_per_image": round(avg_bpp, 5),
                    **ap_dict,
                }
        qs = [str(q) for q, _ in SCOPE_OFFICIAL_JPEG_SPEC]
        r_anc_ds = [det_block["jpeg_direct"][q]["bpp"] for q in qs]
        r_scp_ds = [det_block["scope_jpeg"][q]["bpp"] for q in qs]
        r_anc_avg = [det_block["jpeg_direct"][q]["avg_bpp_per_image"] for q in qs]
        r_scp_avg = [det_block["scope_jpeg"][q]["avg_bpp_per_image"] for q in qs]
        for metric_k in ("AP", "AP50", "AP75"):
            a_anc = [det_block["jpeg_direct"][q][metric_k] for q in qs]
            a_scp = [det_block["scope_jpeg"][q][metric_k] for q in qs]
            det_block[f"bdbr_{metric_k.lower()}_pct"] = pchip_bdbr(r_anc_ds, a_anc, r_scp_ds, a_scp)
            det_block[f"bdbr_{metric_k.lower()}_avg_img_bpp_pct"] = pchip_bdbr(r_anc_avg, a_anc, r_scp_avg, a_scp)
        summary["detectors"][det] = det_block
        print(
            f"[COCO RESULT | {det}] BDBR(AP@[.50:.95]) = {det_block['bdbr_ap_pct']}% "
            f"(avg_img_bpp: {det_block['bdbr_ap_avg_img_bpp_pct']}%) | "
            f"BDBR(AP@50) = {det_block['bdbr_ap50_pct']}%",
            flush=True,
        )

    return summary


@torch.no_grad()
def run_mot17_bridge_verification(args: argparse.Namespace, scope_bundle: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Evaluate Official SCOPE on held-out MOT17 (1,632 frames) under (1) JPEG and (2) HEVC All-Intra (GOP=1)."""
    from models.frozen_detector_fpn import FrozenFPNDetectorTeacher
    from scripts.eval_sota_and_tracking import apply_official_scope_clip
    from scripts.kaggle_shard_runner import (
        _call_evaluate_real_codec,
        _construct_real_dataset,
        _safe_find_mounted_dataset,
    )
    from video_codecs.hybrid_x265_ste import HybridX265STE

    mot_root = Path(args.mot17_root)
    if not (mot_root.exists() and list(mot_root.rglob("MOT17-11-FRCNN"))):
        discovered = _safe_find_mounted_dataset("mot17")
        if discovered is not None:
            mot_root = discovered
        else:
            for cand in Path("/kaggle/input").rglob("MOT17-11-FRCNN"):
                mot_root = cand.parent.parent if cand.parent.name == "train" else cand.parent
                break
    print(f"[MOT17 Bridge] Resolved MOT17 root -> {mot_root}", flush=True)

    ds = _construct_real_dataset(
        data_root=str(mot_root),
        shard_id=0,
        num_shards=1,
        clip_len=32,
        stride=32,
        img_hw=(384, 640),
        is_train=False,
        enforce_disjoint_split=True,
        split="test",
    )
    if len(ds) == 0:
        raise RuntimeError(f"Zero held-out MOT17 clips found under {mot_root}")
    print(f"[MOT17 Bridge] Loaded {len(ds)} clips ({len(ds) * 32} frames)", flush=True)

    fpn_teacher = FrozenFPNDetectorTeacher(pretrained=True, use_yolov8s=True).to(device).eval()
    codec = HybridX265STE(preset="medium", gop_len=1, pix_fmt="yuv420p", qg_size=16).to(device).eval()

    scope_model = scope_bundle["model"]
    fam_map = scope_bundle["fam_map"]
    jpeg_fam_id = int(fam_map.get("diff_jpeg", 0))

    jpeg_buckets: Dict[Tuple[str, int], Dict[str, Any]] = {
        (m, q): {"bits": 0, "pix": 0, "gt_b": [], "ig_b": [], "fr_p": [], "fr_s": [], "yo_p": [], "yo_s": []}
        for m in ("jpeg_direct", "scope_jpeg")
        for q, _ in SCOPE_OFFICIAL_JPEG_SPEC
    }
    crfs = [27, 32, 37, 42]
    intra_buckets: Dict[Tuple[str, int], Dict[str, Any]] = {
        (m, q): {"bits": 0, "pix": 0, "gt_b": [], "ig_b": [], "fr_p": [], "fr_s": [], "yo_p": [], "yo_s": []}
        for m in ("x265_intra_gop1", "scope_x265_intra_gop1")
        for q in crfs
    }

    t0 = time.perf_counter()
    for idx in range(len(ds)):
        sample = ds[idx]
        video = sample["video"].unsqueeze(0).to(device)  # [1, 32, 3, 384, 640]
        m_det = sample["m_det"].unsqueeze(0).to(device)
        gt_boxes = sample["gt_boxes"].unsqueeze(0).to(device)
        ignore_boxes = sample["ignore_boxes"].unsqueeze(0).to(device)
        b, t, c, h, w = video.shape
        clean_flat = video.view(b * t, c, h, w)

        clean_dets = []
        for st in range(0, clean_flat.shape[0], 4):
            sub_cd = fpn_teacher.detector(list(torch.clamp(clean_flat[st : st + 4], 0.0, 1.0)))
            clean_dets.extend({k: v.detach().cpu() for k, v in d.items()} for d in sub_cd)
        cached_clean_info = {"t_feats": fpn_teacher.extract_fast_pyramid(clean_flat), "clean_dets": clean_dets}

        # --- Part A: JPEG per frame ---
        frames_np_rgb = np.round(video[0].clamp(0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
        frames_bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames_np_rgb]

        for q, sev in SCOPE_OFFICIAL_JPEG_SPEC:
            d_tensors, s_tensors = [], []
            for f_bgr in frames_bgr:
                d_bgr, nb_d = jpeg_encode_decode_bgr(f_bgr, q)
                jpeg_buckets[("jpeg_direct", q)]["bits"] += nb_d * 8
                jpeg_buckets[("jpeg_direct", q)]["pix"] += h * w
                d_tensors.append(torch.from_numpy(cv2.cvtColor(d_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0)

                ed_bgr = scope_preedit_bgr(f_bgr, scope_model, jpeg_fam_id, sev, device)
                s_bgr, nb_s = jpeg_encode_decode_bgr(ed_bgr, q)
                jpeg_buckets[("scope_jpeg", q)]["bits"] += nb_s * 8
                jpeg_buckets[("scope_jpeg", q)]["pix"] += h * w
                s_tensors.append(torch.from_numpy(cv2.cvtColor(s_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0)

            for m_key, t_list in (("jpeg_direct", d_tensors), ("scope_jpeg", s_tensors)):
                recon_v = torch.stack(t_list, dim=0).unsqueeze(0).to(device)
                acc = fpn_teacher.evaluate_task_accuracy(
                    video,
                    recon_v,
                    m_det,
                    gt_boxes=gt_boxes,
                    ignore_boxes=ignore_boxes,
                    require_real_detector=True,
                    return_raw_detections=True,
                    cached_clean_info=cached_clean_info,
                )
                raw_d = acc["raw_detections"]
                bkt = jpeg_buckets[(m_key, q)]
                bkt["gt_b"].extend(raw_d["gt_boxes"])
                bkt["ig_b"].extend(raw_d["ignore_boxes"])
                bkt["fr_p"].extend(raw_d["pred_boxes"])
                bkt["fr_s"].extend(raw_d["pred_scores"])
                if "yolo_pred_boxes" in raw_d:
                    bkt["yo_p"].extend(raw_d["yolo_pred_boxes"])
                    bkt["yo_s"].extend(raw_d["yolo_pred_scores"])

        # --- Part B: libx265 All-Intra (gop_len=1, bframes=0) ---
        for crf in crfs:
            out_anc = _call_evaluate_real_codec(codec, video, crf, None, family_id=0, gop_len=1, bframes=0)
            v_scp = apply_official_scope_clip(video, crf, scope_bundle, chunk_size=8)
            out_scp = _call_evaluate_real_codec(codec, v_scp, crf, None, family_id=0, gop_len=1, bframes=0)

            for m_key, c_out in (("x265_intra_gop1", out_anc), ("scope_x265_intra_gop1", out_scp)):
                acc = fpn_teacher.evaluate_task_accuracy(
                    video,
                    c_out["video_recon"],
                    m_det,
                    gt_boxes=gt_boxes,
                    ignore_boxes=ignore_boxes,
                    require_real_detector=True,
                    return_raw_detections=True,
                    cached_clean_info=cached_clean_info,
                )
                raw_d = acc["raw_detections"]
                bkt = intra_buckets[(m_key, crf)]
                bkt["bits"] += sum(int(s) * 8 for s in c_out["packet_sizes"])
                bkt["pix"] += t * h * w
                bkt["gt_b"].extend(raw_d["gt_boxes"])
                bkt["ig_b"].extend(raw_d["ignore_boxes"])
                bkt["fr_p"].extend(raw_d["pred_boxes"])
                bkt["fr_s"].extend(raw_d["pred_scores"])
                if "yolo_pred_boxes" in raw_d:
                    bkt["yo_p"].extend(raw_d["yolo_pred_boxes"])
                    bkt["yo_s"].extend(raw_d["yolo_pred_scores"])

        if (idx + 1) % 5 == 0 or (idx + 1) == len(ds):
            print(f"[MOT17 Bridge] Clip {idx + 1}/{len(ds)} ({time.perf_counter() - t0:.1f}s)", flush=True)

    def _summarize_regime(buckets: Dict[Tuple[str, int], Dict[str, Any]], m_anc: str, m_scp: str, levels: List[int]) -> Dict[str, Any]:
        res: Dict[str, Any] = {m_anc: {}, m_scp: {}}
        for m_k in (m_anc, m_scp):
            for lv in levels:
                bkt = buckets[(m_k, lv)]
                bpp = bkt["bits"] / max(1, bkt["pix"])
                fr_ap = FrozenFPNDetectorTeacher.compute_dataset_level_coco_metrics(
                    bkt["gt_b"], bkt["fr_p"], bkt["fr_s"], all_ignore_boxes=bkt["ig_b"]
                )
                row_lv: Dict[str, Any] = {
                    "bpp": round(bpp, 5),
                    "frcnn_ap50": fr_ap["gt_ap50_pct"],
                    "frcnn_ap50_95": fr_ap["gt_ap50_95_pct"],
                }
                if bkt["yo_p"]:
                    yo_ap = FrozenFPNDetectorTeacher.compute_dataset_level_coco_metrics(
                        bkt["gt_b"], bkt["yo_p"], bkt["yo_s"], all_ignore_boxes=bkt["ig_b"]
                    )
                    row_lv["yolo_ap50"] = yo_ap["gt_ap50_pct"]
                    row_lv["yolo_ap50_95"] = yo_ap["gt_ap50_95_pct"]
                res[m_k][str(lv)] = row_lv
        lvs = [str(x) for x in levels]
        r1 = [res[m_anc][k]["bpp"] for k in lvs]
        r2 = [res[m_scp][k]["bpp"] for k in lvs]
        for mk in ("frcnn_ap50", "frcnn_ap50_95", "yolo_ap50", "yolo_ap50_95"):
            if all(mk in res[m_anc][k] and mk in res[m_scp][k] for k in lvs):
                res[f"bdbr_{mk}_pct"] = pchip_bdbr(
                    r1, [res[m_anc][k][mk] for k in lvs], r2, [res[m_scp][k][mk] for k in lvs]
                )
        return res

    jpeg_out = _summarize_regime(jpeg_buckets, "jpeg_direct", "scope_jpeg", [q for q, _ in SCOPE_OFFICIAL_JPEG_SPEC])
    intra_out = _summarize_regime(intra_buckets, "x265_intra_gop1", "scope_x265_intra_gop1", crfs)
    print(
        f"[MOT17 JPEG] FR50 BDBR={jpeg_out.get('bdbr_frcnn_ap50_pct')}% | "
        f"FR95 BDBR={jpeg_out.get('bdbr_frcnn_ap50_95_pct')}% | "
        f"YO50 BDBR={jpeg_out.get('bdbr_yolo_ap50_pct')}%",
        flush=True,
    )
    print(
        f"[MOT17 x265 Intra GOP=1] FR50 BDBR={intra_out.get('bdbr_frcnn_ap50_pct')}% | "
        f"FR95 BDBR={intra_out.get('bdbr_frcnn_ap50_95_pct')}% | "
        f"YO50 BDBR={intra_out.get('bdbr_yolo_ap50_pct')}%",
        flush=True,
    )
    return {
        "dataset": "Held-Out MOT17 (1,632 frames, 640x384)",
        "scope_ckpt_sha256": scope_bundle["ckpt_sha256"],
        "mot17_jpeg": jpeg_out,
        "mot17_x265_all_intra_gop1": intra_out,
    }


def load_official_scope_full(scope_repo_dir: str, scope_ckpt: str, device: torch.device) -> Dict[str, Any]:
    from scripts.eval_sota_and_tracking import try_setup_official_scope

    bundle = try_setup_official_scope(scope_ckpt=scope_ckpt, scope_repo_dir=scope_repo_dir, auto_download=True, device=device)
    if not bundle.get("available"):
        raise RuntimeError(f"Official SCOPE unavailable: {bundle.get('reason')}")
    if scope_repo_dir not in sys.path:
        sys.path.insert(0, scope_repo_dir)
    from scope.inference import load_scope_model

    _, fam_map = load_scope_model(bundle["ckpt_path"], device)
    bundle["fam_map"] = fam_map
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["coco_jpeg", "mot17_bridge", "both"], default="coco_jpeg")
    parser.add_argument("--coco_root", type=str, default="/tmp/coco2017")
    parser.add_argument("--mot17_root", type=str, default="/kaggle/input/mot17challenge")
    parser.add_argument("--max_images", type=int, default=5000, help="Number of COCO val2017 images (0 or 5000 = all)")
    parser.add_argument("--detectors", type=str, default="frcnn,yolo,rtdetr", help="Comma-separated: frcnn,yolo,rtdetr")
    parser.add_argument("--include_rtdetr", action="store_true", help="Also evaluate RT-DETR-l on COCO JPEG")
    parser.add_argument("--scope_repo_dir", type=str, default="/tmp/miccunifi_SCOPE")
    parser.add_argument("--scope_ckpt", type=str, default="")
    parser.add_argument("--output_json", type=Path, default=Path("scope_jpeg_verification.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[Verify SCOPE JPEG] Device={device}, mode={args.mode}, max_images={args.max_images}, detectors={args.detectors}",
        flush=True,
    )
    scope_bundle = load_official_scope_full(args.scope_repo_dir, args.scope_ckpt, device)

    report: Dict[str, Any] = {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if args.mode in ("coco_jpeg", "both"):
        report["coco_val2017_jpeg"] = run_coco_jpeg_verification(args, scope_bundle, device)
    if args.mode in ("mot17_bridge", "both"):
        report["mot17_diagnostic_bridge"] = run_mot17_bridge_verification(args, scope_bundle, device)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[Verify SCOPE JPEG] Saved report -> {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
