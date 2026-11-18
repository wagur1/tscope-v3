#!/usr/bin/env python3
"""
Zero-Shot Cross-Detector, Cross-Codec (libx265 & libx264), and Random-Access (IBBP)
BDBR Evaluation for T-SCOPE v3 (640x384 16:9, gop_len=32, CRFs [27, 32, 37, 42], PCHIP + 10k Bootstrap CIs).
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.frozen_detector_fpn import FrozenFPNDetectorTeacher
from models.temporal_scope import TemporalSCOPE
from scripts.experiment_protocol import CRFS, EVAL_GOP_LEN, IMG_H, IMG_W, SCHEMA_VERSION
from scripts.kaggle_shard_runner import _call_evaluate_real_codec, _construct_real_dataset, _safe_find_mounted_dataset
from scripts.merge_shards import _PreparedClipBootstrapCurve, _compute_bdbr_samples, _summarize_bootstrap_samples, bjontegaard_delta_rate
from video_codecs.hybrid_x265_ste import HybridX265STE


def sha256_file_full(path: Path) -> str:
    if not path.exists():
        return "none"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class CrossDetectorEvaluator(nn.Module):
    """
    Evaluates Zero-Shot Cross-Detector / Cross-Backbone Generalization of Single-Teacher T-SCOPE v3
    (trained ONLY with Frozen Faster R-CNN ResNet50-FPN v2, use_yolov8s=False in training; F04 fix):
      1. `fasterrcnn_r50_fpn_v2` : Two-Stage Anchor Detector (In-Domain Training Teacher)
      2. `yolov8s`               : One-Stage CSPDarknet Detector (Unseen Zero-Shot Detector)
      3. `retinanet_r50_fpn_v2`  : One-Stage Focal-Loss Dense Detector (Unseen Zero-Shot Detector)
      4. `fcos_r50_fpn`          : One-Stage Anchor-Free Detector (Unseen Zero-Shot Detector)
      5. `ssdlite_mobilenet_v3`  : Lightweight Edge Mobile Detector (Unseen Zero-Shot Detector)
      6. `swin_t_transformer`    : Vision Transformer Backbone (Zero-Shot Feature-Space SNR/Cosine ONLY)
    """

    BACKBONE_SPECS = [
        ("fasterrcnn_r50_fpn_v2", "Faster R-CNN (ResNet50-FPN v2)", "Two-Stage Anchor Detector (In-Domain Teacher)", False),
        ("yolov8s", "YOLOv8s (CSPDarknet)", "One-Stage Ultralytics Detector (Zero-Shot Unseen)", True),
        ("retinanet_r50_fpn_v2", "RetinaNet (ResNet50-FPN v2)", "One-Stage Focal-Loss Detector (Zero-Shot Unseen)", True),
        ("fcos_r50_fpn", "FCOS (ResNet50-FPN)", "One-Stage Anchor-Free Detector (Zero-Shot Unseen)", True),
        ("ssdlite_mobilenet_v3", "SSDLite (MobileNetV3-Large)", "Edge Mobile Detector (Zero-Shot Unseen)", True),
        ("swin_t_transformer", "Swin-T (Swin Transformer)", "Vision Transformer Feature Backbone (Zero-Shot)", True),
    ]

    def __init__(self, device: torch.device, include_yolo: bool = True) -> None:
        super().__init__()
        self.device = device
        from torchvision.models import Swin_T_Weights, swin_t
        from torchvision.models.detection import (
            FCOS_ResNet50_FPN_Weights,
            FasterRCNN_ResNet50_FPN_V2_Weights,
            RetinaNet_ResNet50_FPN_V2_Weights,
            SSDLite320_MobileNet_V3_Large_Weights,
            fasterrcnn_resnet50_fpn_v2,
            fcos_resnet50_fpn,
            retinanet_resnet50_fpn_v2,
            ssdlite320_mobilenet_v3_large,
        )

        print("[CrossDetector] Loading pretrained Faster R-CNN + 4 Zero-Shot Detectors + Swin-T ViT...")
        self.detectors: Dict[str, nn.Module] = {
            "fasterrcnn_r50_fpn_v2": fasterrcnn_resnet50_fpn_v2(
                weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            ).to(device).eval(),
            "retinanet_r50_fpn_v2": retinanet_resnet50_fpn_v2(
                weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
            ).to(device).eval(),
            "fcos_r50_fpn": fcos_resnet50_fpn(
                weights=FCOS_ResNet50_FPN_Weights.DEFAULT
            ).to(device).eval(),
            "ssdlite_mobilenet_v3": ssdlite320_mobilenet_v3_large(
                weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            ).to(device).eval(),
        }
        self.swin = swin_t(weights=Swin_T_Weights.DEFAULT).features.to(device).eval()

        self._yolo_holder: List[Any] = []
        self.has_yolov8s = False
        if include_yolo:
            try:
                from ultralytics import YOLO

                y_wrap = YOLO("yolov8s.pt")
                y_wrap.model.to(device).eval()
                for p in y_wrap.model.parameters():
                    p.requires_grad_(False)
                self._yolo_holder = [y_wrap]
                self.has_yolov8s = True
            except Exception as exc:
                print(f"[CrossDetector] YOLOv8s unavailable ({exc}); skipping yolov8s curve.")

        for m in list(self.detectors.values()) + [self.swin]:
            for p in m.parameters():
                p.requires_grad_(False)

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        )

    @property
    def yolo_wrapper(self) -> Any:
        return self._yolo_holder[0] if self._yolo_holder else None

    def active_backbone_specs(self) -> List[Tuple[str, str, str, bool]]:
        return [
            spec for spec in self.BACKBONE_SPECS if (spec[0] != "yolov8s" or self.has_yolov8s)
        ]

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    @torch.no_grad()
    def _extract_backbone_feats_chunked(
        self, name: str, imgs: torch.Tensor, chunk_size: int = 8
    ) -> Dict[str, torch.Tensor]:
        chunks_out: List[Dict[str, torch.Tensor]] = []
        for start in range(0, imgs.shape[0], chunk_size):
            sub = torch.clamp(imgs[start : start + chunk_size], 0.0, 1.0)
            if name in self.detectors:
                feats = dict(self.detectors[name].backbone(self._norm(sub)))
            elif name == "yolov8s" and self.yolo_wrapper is not None:
                ym = self.yolo_wrapper.model
                ym.eval()
                feats = {}
                y_list: List[torch.Tensor] = []
                cur = sub
                for idx, layer in enumerate(ym.model[:10]):
                    if layer.f != -1:
                        cur = (
                            y_list[layer.f]
                            if isinstance(layer.f, int)
                            else [cur if j == -1 else y_list[j] for j in layer.f]
                        )
                    cur = layer(cur)
                    y_list.append(cur)
                    if idx in (4, 6, 9):
                        feats[f"yolo_{idx}"] = cur
            elif name == "swin_t_transformer":
                feats = {}
                cur = self._norm(sub)
                for idx, layer in enumerate(self.swin):
                    cur = layer(cur)
                    if idx in (1, 3, 5, 7):
                        feats[f"s{idx}"] = cur.permute(0, 3, 1, 2).contiguous()
            else:
                raise ValueError(f"Unknown backbone: {name}")
            chunks_out.append({k: v.detach() for k, v in feats.items()})

        keys = list(chunks_out[0].keys())
        return {k: torch.cat([c[k] for c in chunks_out], dim=0) for k in keys}

    @torch.no_grad()
    def extract_clip_metrics_and_dets(
        self,
        backbone_name: str,
        clean_video: torch.Tensor,
        recon_video: torch.Tensor,
        m_det: torch.Tensor,
        gt_boxes: torch.Tensor,
        ignore_boxes: torch.Tensor,
        f_clean_cached: Optional[Dict[str, torch.Tensor]] = None,
        det_chunk_size: int = 8,
    ) -> Dict[str, Any]:
        b, t, c, h, w = clean_video.shape
        clean_flat = clean_video.view(b * t, c, h, w)
        recon_flat = recon_video.view(b * t, c, h, w)
        m_det_flat = m_det.view(b * t, 1, h, w)

        f_clean = (
            f_clean_cached
            if f_clean_cached is not None
            else self._extract_backbone_feats_chunked(backbone_name, clean_flat, chunk_size=det_chunk_size)
        )
        f_recon = self._extract_backbone_feats_chunked(backbone_name, recon_flat, chunk_size=det_chunk_size)

        cos_list: List[float] = []
        snr_list: List[float] = []
        for k, fc in f_clean.items():
            fr = f_recon[k]
            fc_dev = fc.to(fr.device)
            ml = F.interpolate(m_det_flat, size=fr.shape[-2:], mode="nearest")
            if float(ml.sum().item()) < 1.0:
                ml = torch.ones_like(ml)
            cos_map = F.cosine_similarity(fc_dev, fr, dim=1, eps=1e-6).unsqueeze(1)
            masked_cos = float(((ml * cos_map).sum() / (ml.sum() + 1e-6)).item())
            cos_list.append(masked_cos * 100.0)

            sig_pow = ((ml * fc_dev) ** 2).sum() / (ml.sum() * fr.shape[1] + 1e-6)
            err_pow = ((ml * (fc_dev - fr)) ** 2).sum() / (ml.sum() * fr.shape[1] + 1e-6)
            feat_snr_db = float(10.0 * torch.log10((sig_pow + 1e-6) / (err_pow + 1e-8)).item())
            snr_list.append(feat_snr_db)

        pred_boxes_list: List[torch.Tensor] = []
        pred_scores_list: List[torch.Tensor] = []
        gt_boxes_list: List[torch.Tensor] = []
        ig_boxes_list: List[torch.Tensor] = []

        gt_flat = gt_boxes.view(b * t, -1, 4)
        ig_flat = ignore_boxes.view(b * t, -1, 4)
        is_detector = backbone_name in self.detectors or (backbone_name == "yolov8s" and self.has_yolov8s)

        if backbone_name in self.detectors:
            det_model = self.detectors[backbone_name]
            recon_dets: List[Dict[str, torch.Tensor]] = []
            for start in range(0, recon_flat.shape[0], det_chunk_size):
                sub = torch.clamp(recon_flat[start : start + det_chunk_size], 0.0, 1.0)
                recon_dets.extend(det_model(list(sub)))
            for f_idx, r_det in enumerate(recon_dets):
                person_mask = (r_det["labels"] == 1) & (r_det["scores"] >= 0.01)
                pred_boxes_list.append(r_det["boxes"][person_mask].detach().cpu())
                pred_scores_list.append(r_det["scores"][person_mask].detach().cpu())
                f_gt = gt_flat[f_idx].detach().cpu()
                gt_boxes_list.append(f_gt[f_gt[:, 0] >= 0.0])
                f_ig = ig_flat[f_idx].detach().cpu()
                ig_boxes_list.append(f_ig[f_ig[:, 0] >= 0.0])
        elif backbone_name == "yolov8s" and self.yolo_wrapper is not None:
            for start in range(0, recon_flat.shape[0], det_chunk_size):
                sub = torch.clamp(recon_flat[start : start + det_chunk_size], 0.0, 1.0)
                y_res = self.yolo_wrapper.predict(
                    source=sub, imgsz=max(h, w), conf=0.01, classes=[0], verbose=False
                )
                for off, r_y in enumerate(y_res):
                    f_idx = start + off
                    pred_boxes_list.append(r_y.boxes.xyxy.detach().cpu())
                    pred_scores_list.append(r_y.boxes.conf.detach().cpu())
                    f_gt = gt_flat[f_idx].detach().cpu()
                    gt_boxes_list.append(f_gt[f_gt[:, 0] >= 0.0])
                    f_ig = ig_flat[f_idx].detach().cpu()
                    ig_boxes_list.append(f_ig[f_idx if False else 0] if False else f_ig[f_ig[:, 0] >= 0.0])

        return {
            "feat_snr_db": round(float(np.mean(snr_list)), 3),
            "fpn_cos_pct": round(float(np.mean(cos_list)), 3),
            "is_detector": is_detector,
            "pred_boxes": pred_boxes_list,
            "pred_scores": pred_scores_list,
            "gt_boxes": gt_boxes_list,
            "ignore_boxes": ig_boxes_list,
        }


def _load_v3_checkpoint_into_model(
    ckpt_path: Path, max_delta_qp: float, device: torch.device
) -> Tuple[TemporalSCOPE, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt.get("state_dict", ckpt)
    num_fam = int(ckpt.get("configuration", {}).get("num_families", 2))
    if "conditioner.family_emb.weight" in state:
        num_fam = int(state["conditioner.family_emb.weight"].shape[0])
    model = TemporalSCOPE(
        base_ch=48, num_families=num_fam, cond_dim=64, qg_size=16, max_delta_qp=max_delta_qp
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt


def _evaluate_regime(
    regime_name: str,
    family_id: int,
    bframes: int,
    gop_len: int,
    qps: Sequence[int],
    dataset: Any,
    model_tscope: TemporalSCOPE,
    model_spatial: TemporalSCOPE,
    codec: HybridX265STE,
    evaluator: CrossDetectorEvaluator,
    backbones_for_regime: List[Tuple[str, str, str, bool]],
    device: torch.device,
    n_bootstrap: int = 10000,
) -> Dict[str, Any]:
    methods = ["anchor", "spatial_scope_framewise", "tscope_temporal_ctu_qp"]
    # Per (backbone, method, qp) list of clip-level records for exact AP pooling + 10k clip-bootstrap CIs
    clip_records: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {
        (b_key, m, q): []
        for b_key, _, _, _ in backbones_for_regime
        for m in methods
        for q in qps
    }
    snr_cos_accum: Dict[Tuple[str, str, int], List[Tuple[float, float]]] = {
        (b_key, m, q): []
        for b_key, _, _, _ in backbones_for_regime
        for m in methods
        for q in qps
    }

    t0 = time.perf_counter()
    for idx in range(len(dataset)):
        sample = dataset[idx]
        video = sample["video"].unsqueeze(0).to(device)
        m_det = sample["m_det"].unsqueeze(0).to(device)
        gt_boxes = sample["gt_boxes"].unsqueeze(0).to(device)
        ignore_boxes = sample["ignore_boxes"].unsqueeze(0).to(device)
        b, t, c, h, w = video.shape
        clean_flat = video.view(b * t, c, h, w)

        clean_feats_clip: Dict[str, Dict[str, torch.Tensor]] = {}
        for b_key, _, _, _ in backbones_for_regime:
            clean_feats_clip[b_key] = evaluator._extract_backbone_feats_chunked(b_key, clean_flat)

        fam_t = torch.tensor([family_id], device=device, dtype=torch.long)
        for qp in qps:
            sev = torch.tensor([(qp - 22.0) / 25.0], device=device, dtype=torch.float32)

            base_out = _call_evaluate_real_codec(codec, video, qp, None, family_id=family_id, gop_len=gop_len, bframes=bframes)
            pred_sp = model_spatial(video, fam_t, sev, reuse_iframe_qp=False, disable_temporal=True)
            sp_out = _call_evaluate_real_codec(codec, pred_sp["i_edit"], qp, None, family_id=family_id, gop_len=gop_len, bframes=bframes)

            pred_ts = model_tscope(video, fam_t, sev, reuse_iframe_qp=False, disable_temporal=False)
            ts_out = _call_evaluate_real_codec(
                codec, pred_ts["i_edit"], qp, pred_ts["ctu_qp_offsets"], family_id=family_id, gop_len=gop_len, bframes=bframes
            )

            encoded_map = {
                "anchor": base_out,
                "spatial_scope_framewise": sp_out,
                "tscope_temporal_ctu_qp": ts_out,
            }

            for b_key, _, _, _ in backbones_for_regime:
                for m_name, c_out in encoded_map.items():
                    res = evaluator.extract_clip_metrics_and_dets(
                        b_key,
                        video,
                        c_out["video_recon"],
                        m_det,
                        gt_boxes,
                        ignore_boxes,
                        f_clean_cached=clean_feats_clip[b_key],
                    )
                    snr_cos_accum[(b_key, m_name, qp)].append((res["feat_snr_db"], res["fpn_cos_pct"]))
                    if res["is_detector"]:
                        clip_records[(b_key, m_name, qp)].append(
                            {
                                "frame_ids": sample["frame_ids"],
                                "packet_sizes": c_out["packet_sizes"],
                                "frame_types": c_out["frame_types"],
                                "height": h,
                                "width": w,
                                "raw_detections": {
                                    "gt_boxes": [x.tolist() for x in res["gt_boxes"]],
                                    "ignore_boxes": [x.tolist() for x in res["ignore_boxes"]],
                                    "pred_boxes": [x.tolist() for x in res["pred_boxes"]],
                                    "pred_scores": [x.tolist() for x in res["pred_scores"]],
                                },
                                "total_bpp": float(c_out["total_bpp"]),
                                "p_frame_bpp": float(c_out["p_frame_bpp"]),
                            }
                        )
                    else:
                        clip_records[(b_key, m_name, qp)].append(
                            {
                                "frame_ids": sample["frame_ids"],
                                "packet_sizes": c_out["packet_sizes"],
                                "frame_types": c_out["frame_types"],
                                "height": h,
                                "width": w,
                                "total_bpp": float(c_out["total_bpp"]),
                                "p_frame_bpp": float(c_out["p_frame_bpp"]),
                            }
                        )
        del clean_feats_clip
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (idx + 1) % 5 == 0 or (idx + 1) == len(dataset):
            print(f"[{regime_name}] Clip {idx + 1}/{len(dataset)} ({time.perf_counter() - t0:.1f}s)", flush=True)

    num_clips = len(dataset)
    rng = np.random.default_rng(2026)
    boot_weights = (
        rng.multinomial(num_clips, np.full(num_clips, 1.0 / float(num_clips)), size=n_bootstrap).astype(np.float64)
        if (n_bootstrap > 0 and num_clips >= 2)
        else None
    )

    def _safe_bdbr(r1, a1, r2, a2) -> Optional[float]:
        try:
            val = bjontegaard_delta_rate(r1, a1, r2, a2)
            return round(float(val), 2) if math.isfinite(val) else None
        except ValueError:
            return None

    regime_summary: Dict[str, Any] = {
        "encoder": "libx265" if family_id == 0 else "libx264",
        "family_id": family_id,
        "pix_fmt": "yuv420p",
        "gop_len": gop_len,
        "bframes": bframes,
        "crfs": list(qps),
        "backbones": {},
    }

    for b_key, b_label, b_paradigm, is_zero_shot in backbones_for_regime:
        is_det = b_key != "swin_t_transformer"
        per_qp_data: Dict[str, Dict[int, Dict[str, Any]]] = {m: {} for m in methods}
        boot_curves: Dict[str, Optional[Dict[str, np.ndarray]]] = {m: None for m in methods}

        for m_name in methods:
            if is_det and boot_weights is not None:
                prep = _PreparedClipBootstrapCurve(
                    {q: clip_records[(b_key, m_name, q)] for q in qps}, qps
                )
                boot_curves[m_name] = prep.evaluate_resampled_curves(boot_weights)

            for q in qps:
                recs = clip_records[(b_key, m_name, q)]
                snrs = [pair[0] for pair in snr_cos_accum[(b_key, m_name, q)]]
                coss = [pair[1] for pair in snr_cos_accum[(b_key, m_name, q)]]
                tot_bits = sum(sum(int(s) * 8 for s in r["packet_sizes"]) for r in recs)
                tot_pix = sum(len(r["frame_ids"]) * int(r["height"]) * int(r["width"]) for r in recs)
                p_bits = sum(
                    sum(int(s) * 8 for s, ft in zip(r["packet_sizes"], r["frame_types"]) if ft != "I")
                    for r in recs
                )
                p_pix = sum(
                    sum(1 for ft in r["frame_types"] if ft != "I") * int(r["height"]) * int(r["width"])
                    for r in recs
                )

                if is_det:
                    all_gt = [
                        torch.tensor(box, dtype=torch.float32).reshape(-1, 4)
                        for r in recs
                        for box in r["raw_detections"]["gt_boxes"]
                    ]
                    all_ig = [
                        torch.tensor(box, dtype=torch.float32).reshape(-1, 4)
                        for r in recs
                        for box in r["raw_detections"]["ignore_boxes"]
                    ]
                    all_pb = [
                        torch.tensor(box, dtype=torch.float32).reshape(-1, 4)
                        for r in recs
                        for box in r["raw_detections"]["pred_boxes"]
                    ]
                    all_ps = [
                        torch.tensor(sc, dtype=torch.float32)
                        for r in recs
                        for sc in r["raw_detections"]["pred_scores"]
                    ]
                    coco_m = FrozenFPNDetectorTeacher.compute_dataset_level_coco_metrics(
                        all_gt, all_pb, all_ps, all_ignore_boxes=all_ig
                    )
                    gt_ap50 = coco_m["gt_ap50_pct"]
                    gt_ap95 = coco_m["gt_ap50_95_pct"]
                else:
                    gt_ap50 = None
                    gt_ap95 = None

                per_qp_data[m_name][q] = {
                    "total_bpp": round(tot_bits / max(1, tot_pix), 5),
                    "p_bpp": round(p_bits / max(1, p_pix), 5) if p_pix > 0 else None,
                    "feat_snr_db": round(float(np.mean(snrs)), 3),
                    "fpn_cos_pct": round(float(np.mean(coss)), 3),
                    "gt_ap50_pct": gt_ap50,
                    "gt_ap50_95_pct": gt_ap95,
                }

        ar = [per_qp_data["anchor"][q]["total_bpp"] for q in qps]
        ap = [per_qp_data["anchor"][q]["p_bpp"] for q in qps]
        asnr = [per_qp_data["anchor"][q]["feat_snr_db"] for q in qps]

        sr = [per_qp_data["spatial_scope_framewise"][q]["total_bpp"] for q in qps]
        sp = [per_qp_data["spatial_scope_framewise"][q]["p_bpp"] for q in qps]
        ssnr = [per_qp_data["spatial_scope_framewise"][q]["feat_snr_db"] for q in qps]

        tr = [per_qp_data["tscope_temporal_ctu_qp"][q]["total_bpp"] for q in qps]
        tp = [per_qp_data["tscope_temporal_ctu_qp"][q]["p_bpp"] for q in qps]
        tsnr = [per_qp_data["tscope_temporal_ctu_qp"][q]["feat_snr_db"] for q in qps]

        entry_out: Dict[str, Any] = {
            "label": b_label,
            "paradigm": b_paradigm,
            "zero_shot_unseen_in_training": is_zero_shot,
            "is_object_detector": is_det,
            "spatial_scope_total_bdbr_feat_snr_pct": _safe_bdbr(ar, asnr, sr, ssnr),
            "spatial_scope_p_bdbr_feat_snr_pct": _safe_bdbr(ap, asnr, sp, ssnr) if all(x is not None for x in ap + sp) else None,
            "tscope_total_bdbr_feat_snr_pct": _safe_bdbr(ar, asnr, tr, tsnr),
            "tscope_p_bdbr_feat_snr_pct": _safe_bdbr(ap, asnr, tp, tsnr) if all(x is not None for x in ap + tp) else None,
            "per_qp": {
                m_name: {str(q): per_qp_data[m_name][q] for q in qps}
                for m_name in methods
            },
        }

        if is_det:
            aap50 = [per_qp_data["anchor"][q]["gt_ap50_pct"] for q in qps]
            sap50 = [per_qp_data["spatial_scope_framewise"][q]["gt_ap50_pct"] for q in qps]
            tap50 = [per_qp_data["tscope_temporal_ctu_qp"][q]["gt_ap50_pct"] for q in qps]
            aap95 = [per_qp_data["anchor"][q]["gt_ap50_95_pct"] for q in qps]
            sap95 = [per_qp_data["spatial_scope_framewise"][q]["gt_ap50_95_pct"] for q in qps]
            tap95 = [per_qp_data["tscope_temporal_ctu_qp"][q]["gt_ap50_95_pct"] for q in qps]

            entry_out["spatial_total_bdbr_gt_ap50_pct"] = _safe_bdbr(ar, aap50, sr, sap50)
            entry_out["tscope_total_bdbr_gt_ap50_pct"] = _safe_bdbr(ar, aap50, tr, tap50)
            entry_out["tscope_p_bdbr_gt_ap50_pct"] = (
                _safe_bdbr(ap, aap50, tp, tap50) if all(x is not None for x in ap + tp) else None
            )
            entry_out["spatial_total_bdbr_gt_ap50_95_pct"] = _safe_bdbr(ar, aap95, sr, sap95)
            entry_out["tscope_total_bdbr_gt_ap50_95_pct"] = _safe_bdbr(ar, aap95, tr, tap95)
            entry_out["tscope_p_bdbr_gt_ap50_95_pct"] = (
                _safe_bdbr(ap, aap95, tp, tap95) if all(x is not None for x in ap + tp) else None
            )

            anc_b = boot_curves["anchor"]
            ts_b = boot_curves["tscope_temporal_ctu_qp"]
            if anc_b is not None and ts_b is not None:
                ci_ap50 = _summarize_bootstrap_samples(
                    _compute_bdbr_samples(anc_b["total_bpp"], anc_b["gt_ap50_pct"], ts_b["total_bpp"], ts_b["gt_ap50_pct"])
                )
                ci_ap95 = _summarize_bootstrap_samples(
                    _compute_bdbr_samples(anc_b["total_bpp"], anc_b["gt_ap50_95_pct"], ts_b["total_bpp"], ts_b["gt_ap50_95_pct"])
                )
                entry_out["tscope_total_bdbr_gt_ap50_ci95"] = ci_ap50["ci95"] if ci_ap50 else None
                entry_out["tscope_total_bdbr_gt_ap50_95_ci95"] = ci_ap95["ci95"] if ci_ap95 else None

        regime_summary["backbones"][b_key] = entry_out
        print(
            f"  [{regime_name}] {b_label:<32} | ZeroShot={str(is_zero_shot):<5} | "
            f"FeatSNR BDBR={entry_out['tscope_total_bdbr_feat_snr_pct']} | "
            f"GT AP@50 BDBR={entry_out.get('tscope_total_bdbr_gt_ap50_pct')}"
        )

    return regime_summary


ALL_REGIMES = (
    "hevc_x265_low_delay_p",
    "avc_x264_low_delay_p",
    "hevc_x265_random_access_ibbp",
)


def merge_regime_json_files(input_paths: Sequence[Path], output_path: Path) -> Dict[str, Any]:
    """Merge independent cross-backbone regime JSON bundles into one cross_backbone_bdbr.json."""
    if not input_paths:
        raise ValueError("No regime JSON files supplied for merge")
    merged_regimes: Dict[str, Any] = {}
    base_doc: Optional[Dict[str, Any]] = None
    for path in input_paths:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        if base_doc is None:
            base_doc = doc
        elif doc.get("provenance", {}).get("tscope_ckpt_sha256") != base_doc.get("provenance", {}).get("tscope_ckpt_sha256"):
            raise ValueError(f"Checkpoint SHA mismatch when merging {path}")
        for r_key, r_val in doc.get("regimes", {}).items():
            merged_regimes[r_key] = r_val

    assert base_doc is not None
    ordered_regimes = {k: merged_regimes[k] for k in ALL_REGIMES if k in merged_regimes}
    for k, v in merged_regimes.items():
        if k not in ordered_regimes:
            ordered_regimes[k] = v

    primary_key = "hevc_x265_low_delay_p" if "hevc_x265_low_delay_p" in ordered_regimes else next(iter(ordered_regimes))
    out_doc = {
        "schema_version": base_doc.get("schema_version", SCHEMA_VERSION),
        "provenance": {
            **base_doc.get("provenance", {}),
            "merged_from_regime_files": [str(Path(p).name) for p in input_paths],
            "regimes_included": list(ordered_regimes.keys()),
        },
        "regimes": ordered_regimes,
        "backbones": ordered_regimes[primary_key]["backbones"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, indent=2, allow_nan=False)
    print(f"Merged {len(ordered_regimes)} regime(s) -> {output_path}")
    return out_doc


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-Shot Cross-Detector & Cross-Codec BDBR Evaluation (T-SCOPE v3)")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--soup_dir", type=str, default="./release_ckpts")
    parser.add_argument("--tscope_ckpt", type=str, default="")
    parser.add_argument("--spatial_ckpt", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--img_h", type=int, default=IMG_H)
    parser.add_argument("--img_w", type=int, default=IMG_W)
    parser.add_argument("--gop_len", type=int, default=EVAL_GOP_LEN)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--eval_x264", action="store_true", default=False, help="Also evaluate cross-codec libx264 (family_id=1)")
    parser.add_argument("--eval_ra", action="store_true", default=False, help="Also evaluate Random Access IBBP (bframes=4)")
    parser.add_argument("--regimes", nargs="+", choices=list(ALL_REGIMES), default=None, help="Explicit subset of regimes to run")
    parser.add_argument("--merge_regime_jsons", nargs="+", default=None, help="Merge partial regime JSONs into cross_backbone_bdbr.json")
    parser.add_argument("--bootstrap_resamples", type=int, default=10000)
    args = parser.parse_args()

    if args.merge_regime_jsons:
        out_p = Path(args.output_json) if args.output_json else Path(args.output_dir) / "cross_backbone_bdbr.json"
        merge_regime_json_files([Path(p) for p in args.merge_regime_jsons], out_p)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = (args.data_root or "").strip()
    if data_root.lower() in ("", "auto", "none"):
        mounted = _safe_find_mounted_dataset("mot17")
        data_root = str(mounted) if mounted is not None else "./data/MOT17_sample"

    soup_dir = Path(args.soup_dir)
    tscope_ckpt_p = Path(args.tscope_ckpt) if args.tscope_ckpt else soup_dir / "tscope_soup_full_tscope.pt"
    spatial_ckpt_p = Path(args.spatial_ckpt) if args.spatial_ckpt else soup_dir / "tscope_soup_spatial_only_scope.pt"
    if not tscope_ckpt_p.exists() or not spatial_ckpt_p.exists():
        raise FileNotFoundError(
            f"Validated v3 soup checkpoints required: {tscope_ckpt_p} and {spatial_ckpt_p}"
        )

    dataset = _construct_real_dataset(
        data_root=data_root,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        clip_len=args.gop_len,
        stride=args.gop_len,
        img_hw=(args.img_h, args.img_w),
        is_train=False,
        enforce_disjoint_split=True,
        split="test",
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No held-out MOT17 test clips discovered at {data_root} with gop_len={args.gop_len}")

    model_tscope, ts_meta = _load_v3_checkpoint_into_model(tscope_ckpt_p, max_delta_qp=4.0, device=device)
    model_spatial, _ = _load_v3_checkpoint_into_model(spatial_ckpt_p, max_delta_qp=0.0, device=device)
    if ts_meta.get("configuration", {}).get("use_yolov8s_in_training", False) is not False:
        raise ValueError("F04 violation: cross-detector transfer requires single-teacher Faster R-CNN checkpoint!")

    codec = HybridX265STE(use_real_x265_in_forward=True, preset="medium").to(device)
    evaluator = CrossDetectorEvaluator(device, include_yolo=True)
    all_specs = evaluator.active_backbone_specs()

    selected_regimes = set(args.regimes) if args.regimes else {
        "hevc_x265_low_delay_p",
        *(["avc_x264_low_delay_p"] if args.eval_x264 else []),
        *(["hevc_x265_random_access_ibbp"] if args.eval_ra else []),
    }

    regimes: Dict[str, Any] = {}
    if "hevc_x265_low_delay_p" in selected_regimes:
        regimes["hevc_x265_low_delay_p"] = _evaluate_regime(
            regime_name="HEVC-LDP (x265, fam=0, bframes=0)",
            family_id=0,
            bframes=0,
            gop_len=args.gop_len,
            qps=CRFS,
            dataset=dataset,
            model_tscope=model_tscope,
            model_spatial=model_spatial,
            codec=codec,
            evaluator=evaluator,
            backbones_for_regime=all_specs,
            device=device,
            n_bootstrap=args.bootstrap_resamples,
        )

    primary_And_yolo_specs = [s for s in all_specs if s[0] in ("fasterrcnn_r50_fpn_v2", "yolov8s", "retinanet_r50_fpn_v2")]
    if "avc_x264_low_delay_p" in selected_regimes:
        regimes["avc_x264_low_delay_p"] = _evaluate_regime(
            regime_name="AVC-LDP (x264, fam=1, bframes=0)",
            family_id=1,
            bframes=0,
            gop_len=args.gop_len,
            qps=CRFS,
            dataset=dataset,
            model_tscope=model_tscope,
            model_spatial=model_spatial,
            codec=codec,
            evaluator=evaluator,
            backbones_for_regime=primary_And_yolo_specs,
            device=device,
            n_bootstrap=args.bootstrap_resamples,
        )

    if "hevc_x265_random_access_ibbp" in selected_regimes:
        regimes["hevc_x265_random_access_ibbp"] = _evaluate_regime(
            regime_name="HEVC-RA (x265, fam=0, bframes=4)",
            family_id=0,
            bframes=4,
            gop_len=args.gop_len,
            qps=CRFS,
            dataset=dataset,
            model_tscope=model_tscope,
            model_spatial=model_spatial,
            codec=codec,
            evaluator=evaluator,
            backbones_for_regime=primary_And_yolo_specs,
            device=device,
            n_bootstrap=args.bootstrap_resamples,
        )

    primary_regime_key = (
        "hevc_x265_low_delay_p" if "hevc_x265_low_delay_p" in regimes else next(iter(regimes))
    )
    test_seqs = sorted(set(c["seq_name"] for c in dataset.clips))
    summary_results: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "held_out_test_sequences": test_seqs,
            "img_h": args.img_h,
            "img_w": args.img_w,
            "gop_len": args.gop_len,
            "crfs": list(CRFS),
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "training_teacher": "fasterrcnn_resnet50_fpn_v2_only",
            "unseen_zero_shot_detectors": ["yolov8s", "retinanet_r50_fpn_v2", "fcos_r50_fpn", "ssdlite_mobilenet_v3"],
            "tscope_ckpt_sha256": sha256_file_full(tscope_ckpt_p),
            "spatial_ckpt_sha256": sha256_file_full(spatial_ckpt_p),
            "bdbr_interpolator": "scipy.interpolate.PchipInterpolator",
            "bootstrap_resamples": args.bootstrap_resamples,
            "gt_used_at_inference": False,
        },
        "regimes": regimes,
        # Top-level alias to primary HEVC LDP backbones for seamless table consumption
        "backbones": regimes[primary_regime_key]["backbones"],
    }

    out_path = (
        Path(args.output_json)
        if args.output_json
        else Path(args.output_dir)
        / (
            f"cross_backbone_bdbr_shard{args.shard_id:02d}of{args.num_shards:02d}.json"
            if args.num_shards > 1
            else "cross_backbone_bdbr.json"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2, allow_nan=False)
    print(f"\nSaved v3 cross-detector & cross-codec generalization bundle -> {out_path}")


if __name__ == "__main__":
    main()
