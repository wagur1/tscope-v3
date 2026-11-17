#!/usr/bin/env python3
"""
T-SCOPE v3 Q1 Benchmark Suite (`eval_sota_and_tracking.py`):
  1. External & Standard SOTA Baselines on Held-Out MOT17 (640x384, gop_len=32, yuv420p, CRFs [27,32,37,42]):
     - `anchor_x265` (Standard HEVC Main 4:2:0)
     - `official_miccunifi_scope` (Dani et al. 2026 official `miccunifi/SCOPE` pretrained `best_val.pt` + libx265)
     - `mpeg_vcm_spatial_resampling` (MPEG-VCM Reduced-Resolution Coding + Bicubic Restoration)
     - `detector_roi_codec_addroi` (Lightweight Detector Boxes -> libx265 `addroi` without pixel pre-editing)
     - `compressed_video_finetuned_frcnn` (Faster R-CNN v2 fine-tuned on compressed disjoint MOT17 train clips)
     - `tscope_spatial_only`, `tscope_iwa_gop_qp_reuse`, `tscope_full` (Ours)
  2. Multi-Object Tracking (`ByteTrackerKalman` -> CLEAR-MOT `MOTA`, `IDF1`, `HOTA`, `IDSW` & Tracking BDBR)
  3. Secondary VCM Datasets (`MOT20` Crowded Pedestrian + `KITTI` Multi-Class Autonomous Driving + `SFU` YUV)
  4. Complexity & Runtime Table (T4 GPU Mean/P95 Latency, IRLS vs U-Net split, FPS, Peak VRAM, GMACs, Codec Time)
"""
import argparse
import copy
import inspect
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.frozen_detector_fpn import FrozenFPNDetectorTeacher
from models.temporal_scope import TemporalSCOPE, estimate_global_affine_irls
from scripts.experiment_protocol import CRFS, EVAL_GOP_LEN, IMG_H, IMG_W, SCHEMA_VERSION, TRAIN_CLIP_LEN, sha256_file
from scripts.kaggle_shard_runner import _call_evaluate_real_codec, _construct_real_dataset, _safe_find_mounted_dataset
from scripts.merge_shards import (
    _PreparedClipBootstrapCurve,
    _compute_bdbr_samples,
    _summarize_bootstrap_samples,
    bjontegaard_delta_rate,
)
from vcm_datasets.vcm_video_datasets import RealVCMVideoDataset
from video_codecs.hybrid_x265_ste import HybridX265STE


# ======================================================================================
# 1. Exact NumPy/SciPy IoU, ByteTrack (Kalman-IoU), and CLEAR-MOT + IDF1 + HOTA Metrics
# ======================================================================================


def _np_box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix [N, M] between XYXY float64 arrays."""
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float64)
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = np.maximum(0.0, boxes1[:, 2] - boxes1[:, 0]) * np.maximum(0.0, boxes1[:, 3] - boxes1[:, 1])
    area2 = np.maximum(0.0, boxes2[:, 2] - boxes2[:, 0]) * np.maximum(0.0, boxes2[:, 3] - boxes2[:, 1])
    union = np.maximum(area1[:, None] + area2[None, :] - inter, 1e-9)
    return inter / union


class _KalmanBoxTracklet:
    """Constant-velocity 8-state Kalman tracklet [cx, cy, a, h, vx, vy, va, vh] for ByteTrack."""

    _F = np.eye(8, dtype=np.float64)
    for _i in range(4):
        _F[_i, _i + 4] = 1.0
    _H = np.eye(4, 8, dtype=np.float64)

    def __init__(self, xyxy: np.ndarray, score: float, track_id: int, frame_id: int) -> None:
        self.track_id = int(track_id)
        self.score = float(score)
        self.frame_id = int(frame_id)
        self.hits = 1
        self.time_since_update = 0
        self.activated = True

        z = self._xyxy_to_cxcyah(xyxy)
        self.mean = np.zeros(8, dtype=np.float64)
        self.mean[:4] = z
        std = np.array(
            [2.0 * 0.05 * z[3], 2.0 * 0.05 * z[3], 1e-2, 2.0 * 0.05 * z[3],
             10.0 * 0.00625 * z[3], 10.0 * 0.00625 * z[3], 1e-5, 10.0 * 0.00625 * z[3]],
            dtype=np.float64,
        )
        self.cov = np.diag(np.square(std))

    @staticmethod
    def _xyxy_to_cxcyah(xyxy: np.ndarray) -> np.ndarray:
        w = max(1.0, float(xyxy[2] - xyxy[0]))
        h = max(1.0, float(xyxy[3] - xyxy[1]))
        cx = float(xyxy[0]) + 0.5 * w
        cy = float(xyxy[1]) + 0.5 * h
        return np.array([cx, cy, w / h, h], dtype=np.float64)

    def to_xyxy(self) -> np.ndarray:
        cx, cy, a, h = self.mean[:4]
        h = max(1.0, float(h))
        w = max(1.0, float(a) * h)
        return np.array([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dtype=np.float64)

    def predict(self) -> None:
        if self.time_since_update > 0:
            self.mean[7] = 0.0
        h = max(1.0, float(self.mean[3]))
        q_std = np.array(
            [0.05 * h, 0.05 * h, 1e-2, 0.05 * h, 0.00625 * h, 0.00625 * h, 1e-5, 0.00625 * h],
            dtype=np.float64,
        )
        self.mean = self._F @ self.mean
        self.cov = self._F @ self.cov @ self._F.T + np.diag(np.square(q_std))
        self.time_since_update += 1

    def update(self, xyxy: np.ndarray, score: float, frame_id: int) -> None:
        z = self._xyxy_to_cxcyah(xyxy)
        h = max(1.0, float(self.mean[3]))
        r_std = np.array([0.05 * h, 0.05 * h, 1e-1, 0.05 * h], dtype=np.float64)
        proj_mean = self._H @ self.mean
        proj_cov = self._H @ self.cov @ self._H.T + np.diag(np.square(r_std))
        kalman_gain = np.linalg.solve(proj_cov, self._H @ self.cov).T
        self.mean = self.mean + kalman_gain @ (z - proj_mean)
        self.cov = self.cov - kalman_gain @ proj_cov @ kalman_gain.T
        self.score = float(score)
        self.frame_id = int(frame_id)
        self.hits += 1
        self.time_since_update = 0
        self.activated = True


class ByteTrackerKalman:
    """
    Deterministic two-stage ByteTrack (Zhang et al., ECCV 2022) associating both
    high-confidence and occluded low-confidence detections via Kalman + Hungarian IoU.
    """

    def __init__(
        self,
        high_thresh: float = 0.50,
        low_thresh: float = 0.10,
        init_thresh: float = 0.55,
        match_iou_high: float = 0.20,
        match_iou_low: float = 0.30,
        max_time_lost: int = 30,
    ) -> None:
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.init_thresh = init_thresh
        self.match_iou_high = match_iou_high
        self.match_iou_low = match_iou_low
        self.max_time_lost = max_time_lost
        self.reset()

    def reset(self) -> None:
        self.frame_id = 0
        self.next_id = 1
        self.tracks: List[_KalmanBoxTracklet] = []

    @staticmethod
    def _associate(
        tracks: List[_KalmanBoxTracklet], det_boxes: np.ndarray, min_iou: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not tracks or det_boxes.shape[0] == 0:
            return [], list(range(len(tracks))), list(range(det_boxes.shape[0]))
        t_boxes = np.stack([t.to_xyxy() for t in tracks], axis=0)
        ious = _np_box_iou(t_boxes, det_boxes)
        cost = 1.0 - ious
        r_ind, c_ind = linear_sum_assignment(cost)
        matches: List[Tuple[int, int]] = []
        matched_t = set()
        matched_d = set()
        for r, c in zip(r_ind, c_ind):
            if float(ious[r, c]) >= min_iou:
                matches.append((int(r), int(c)))
                matched_t.add(int(r))
                matched_d.add(int(c))
        unmatched_t = [i for i in range(len(tracks)) if i not in matched_t]
        unmatched_d = [j for j in range(det_boxes.shape[0]) if j not in matched_d]
        return matches, unmatched_t, unmatched_d

    def update(self, boxes: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.frame_id += 1
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)

        for trk in self.tracks:
            trk.predict()

        high_mask = scores >= self.high_thresh
        low_mask = (scores >= self.low_thresh) & (~high_mask)
        high_boxes, high_scores = boxes[high_mask], scores[high_mask]
        low_boxes, low_scores = boxes[low_mask], scores[low_mask]

        # Stage 1: match high-score detections against all tracked + recently lost tracklets
        matches_1, u_trk_1, u_det_high = self._associate(self.tracks, high_boxes, self.match_iou_high)
        for t_idx, d_idx in matches_1:
            self.tracks[t_idx].update(high_boxes[d_idx], float(high_scores[d_idx]), self.frame_id)

        # Stage 2: match low-score detections against remaining recently active tracklets
        rem_active_indices = [i for i in u_trk_1 if self.tracks[i].time_since_update == 1]
        rem_active_tracks = [self.tracks[i] for i in rem_active_indices]
        matches_2, _, _ = self._associate(rem_active_tracks, low_boxes, self.match_iou_low)
        for local_t_idx, d_idx in matches_2:
            orig_idx = rem_active_indices[local_t_idx]
            self.tracks[orig_idx].update(low_boxes[d_idx], float(low_scores[d_idx]), self.frame_id)

        # Spawn new tracks from unmatched high-confidence detections
        for d_idx in u_det_high:
            if float(high_scores[d_idx]) >= self.init_thresh:
                self.tracks.append(
                    _KalmanBoxTracklet(high_boxes[d_idx], float(high_scores[d_idx]), self.next_id, self.frame_id)
                )
                self.next_id += 1

        # Prune expired tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_time_lost]

        active_out = [
            (t.to_xyxy(), t.track_id)
            for t in self.tracks
            if t.time_since_update == 0 and (t.hits >= 2 or self.frame_id <= 2)
        ]
        if not active_out:
            return np.zeros((0, 4), dtype=np.float64), np.zeros((0,), dtype=np.int64)
        out_boxes = np.stack([pair[0] for pair in active_out], axis=0)
        out_ids = np.array([pair[1] for pair in active_out], dtype=np.int64)
        return out_boxes, out_ids


def compute_mot_tracking_metrics(
    gt_boxes_seq: Sequence[np.ndarray],
    gt_ids_seq: Sequence[np.ndarray],
    pred_boxes_seq: Sequence[np.ndarray],
    pred_ids_seq: Sequence[np.ndarray],
    ignore_boxes_seq: Optional[Sequence[np.ndarray]] = None,
    iou_thresh: float = 0.50,
) -> Dict[str, Any]:
    """
    Computes exact MOTChallenge CLEAR-MOT (`MOTA`, `MOTP`, `IDSW`, `FP`, `FN`),
    global bipartite `IDF1` (Ristani et al., ECCV 2016), and 19-threshold `HOTA`
    (Luiten et al., IJCV 2021) for a single contiguous video sequence.
    """
    n_frames = len(gt_boxes_seq)
    if not (len(gt_ids_seq) == len(pred_boxes_seq) == len(pred_ids_seq) == n_frames):
        raise ValueError("Tracking metric sequences must have identical frame count")

    # Filter ignore-region false positives consistently per frame
    filtered_preds: List[Tuple[np.ndarray, np.ndarray]] = []
    for f_idx in range(n_frames):
        gb = np.asarray(gt_boxes_seq[f_idx], dtype=np.float64).reshape(-1, 4)
        pb = np.asarray(pred_boxes_seq[f_idx], dtype=np.float64).reshape(-1, 4)
        pids = np.asarray(pred_ids_seq[f_idx], dtype=np.int64).reshape(-1)
        if ignore_boxes_seq is not None and pb.shape[0] > 0:
            ig = np.asarray(ignore_boxes_seq[f_idx], dtype=np.float64).reshape(-1, 4)
            if ig.shape[0] > 0:
                gt_max = _np_box_iou(pb, gb).max(axis=1) if gb.shape[0] > 0 else np.zeros(pb.shape[0])
                ig_max = _np_box_iou(pb, ig).max(axis=1)
                keep = (gt_max >= iou_thresh) | (ig_max < iou_thresh)
                pb, pids = pb[keep], pids[keep]
        filtered_preds.append((pb, pids))

    # 1. CLEAR-MOT (MOTA, MOTP, IDSW, FP, FN) with correspondence carry-over
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_idsw = 0
    iou_sum = 0.0
    last_match_for_gt: Dict[int, int] = {}

    unique_gt_ids = sorted({int(x) for arr in gt_ids_seq for x in np.asarray(arr).reshape(-1).tolist()})
    unique_pred_ids = sorted({int(x) for _, pids in filtered_preds for x in pids.tolist()})
    gt_id_to_idx = {gid: i for i, gid in enumerate(unique_gt_ids)}
    pred_id_to_idx = {pid: i for i, pid in enumerate(unique_pred_ids)}
    n_u_gt = len(unique_gt_ids)
    n_u_pred = len(unique_pred_ids)
    gt_id_frame_counts = np.zeros(n_u_gt, dtype=np.int64)
    pred_id_frame_counts = np.zeros(n_u_pred, dtype=np.int64)

    # Precompute framewise IoUs for IDF1 & HOTA
    frame_ious: List[np.ndarray] = []
    frame_g_idx: List[np.ndarray] = []
    frame_p_idx: List[np.ndarray] = []

    for f_idx in range(n_frames):
        gb = np.asarray(gt_boxes_seq[f_idx], dtype=np.float64).reshape(-1, 4)
        gids = np.asarray(gt_ids_seq[f_idx], dtype=np.int64).reshape(-1)
        pb, pids = filtered_preds[f_idx]

        g_indices = np.array([gt_id_to_idx[int(g)] for g in gids], dtype=np.int64)
        p_indices = np.array([pred_id_to_idx[int(p)] for p in pids], dtype=np.int64)
        for gi in g_indices:
            gt_id_frame_counts[gi] += 1
        for pi in p_indices:
            pred_id_frame_counts[pi] += 1

        ious = _np_box_iou(gb, pb)
        frame_ious.append(ious)
        frame_g_idx.append(g_indices)
        frame_p_idx.append(p_indices)

        ng, np_f = gb.shape[0], pb.shape[0]
        total_gt += ng
        if ng == 0:
            total_fp += np_f
            continue
        if np_f == 0:
            total_fn += ng
            continue

        matched_g = set()
        matched_p = set()
        pairs: List[Tuple[int, int]] = []

        # MOTChallenge step 1: preserve previous frame's active correspondence if IoU >= iou_thresh
        for gi_local, gid in enumerate(gids.tolist()):
            prev_pid = last_match_for_gt.get(gid)
            if prev_pid is not None:
                cand_p = [pi_local for pi_local, pid in enumerate(pids.tolist()) if pid == prev_pid and pi_local not in matched_p]
                if cand_p and float(ious[gi_local, cand_p[0]]) >= iou_thresh:
                    pairs.append((gi_local, cand_p[0]))
                    matched_g.add(gi_local)
                    matched_p.add(cand_p[0])

        # MOTChallenge step 2: Hungarian matching on remaining unmatched GT & predictions
        rem_g = [i for i in range(ng) if i not in matched_g]
        rem_p = [j for j in range(np_f) if j not in matched_p]
        if rem_g and rem_p:
            sub_iou = ious[np.ix_(rem_g, rem_p)]
            cost = np.where(sub_iou >= iou_thresh, 1.0 - sub_iou, 1e6)
            r_ind, c_ind = linear_sum_assignment(cost)
            for r, c in zip(r_ind, c_ind):
                if float(sub_iou[r, c]) >= iou_thresh:
                    gi_local, pi_local = rem_g[r], rem_p[c]
                    pairs.append((gi_local, pi_local))
                    matched_g.add(gi_local)
                    matched_p.add(pi_local)

        for gi_local, pi_local in pairs:
            gid = int(gids[gi_local])
            pid = int(pids[pi_local])
            if gid in last_match_for_gt and last_match_for_gt[gid] != pid:
                total_idsw += 1
            last_match_for_gt[gid] = pid
            total_tp += 1
            iou_sum += float(ious[gi_local, pi_local])

        total_fn += ng - len(matched_g)
        total_fp += np_f - len(matched_p)

    mota = (1.0 - float(total_fn + total_fp + total_idsw) / float(max(1, total_gt))) * 100.0 if total_gt > 0 else float("nan")
    motp = (iou_sum / float(max(1, total_tp))) * 100.0 if total_tp > 0 else 0.0

    # 2. Exact Global Bipartite IDF1 (Ristani et al., ECCV 2016)
    if n_u_gt > 0 and n_u_pred > 0:
        pair_matches = np.zeros((n_u_gt, n_u_pred), dtype=np.int64)
        for f_idx in range(n_frames):
            ious = frame_ious[f_idx]
            if ious.size == 0:
                continue
            valid = ious >= iou_thresh
            if not np.any(valid):
                continue
            g_arr = frame_g_idx[f_idx]
            p_arr = frame_p_idx[f_idx]
            rs, cs = np.where(valid)
            for r, c in zip(rs.tolist(), cs.tolist()):
                pair_matches[g_arr[r], p_arr[c]] += 1
        # Minimize IDFP + IDFN = total_gt + total_pred - 2 * IDTP -> maximize pair_matches
        r_id, c_id = linear_sum_assignment(-pair_matches)
        idtp = int(sum(pair_matches[r, c] for r, c in zip(r_id, c_id)))
    else:
        idtp = 0
    total_pred_dets = int(pred_id_frame_counts.sum())
    idfp = max(0, total_pred_dets - idtp)
    idfn = max(0, total_gt - idtp)
    idf1 = (2.0 * float(idtp) / float(max(1, 2 * idtp + idfp + idfn))) * 100.0 if total_gt > 0 else float("nan")

    # 3. Exact 19-Threshold HOTA (Luiten et al., IJCV 2021)
    alphas = [round(0.05 * i, 2) for i in range(1, 20)]
    hota_alphas: List[float] = []
    deta_alphas: List[float] = []
    assa_alphas: List[float] = []

    if total_gt > 0:
        for alpha in alphas:
            if n_u_gt == 0 or n_u_pred == 0:
                hota_alphas.append(0.0)
                deta_alphas.append(0.0)
                assa_alphas.append(0.0)
                continue

            # Potential matches matrix for global alignment prior
            pot_matches = np.zeros((n_u_gt, n_u_pred), dtype=np.float64)
            for f_idx in range(n_frames):
                ious = frame_ious[f_idx]
                if ious.size == 0:
                    continue
                rs, cs = np.where(ious >= alpha)
                g_arr, p_arr = frame_g_idx[f_idx], frame_p_idx[f_idx]
                for r, c in zip(rs.tolist(), cs.tolist()):
                    pot_matches[g_arr[r], p_arr[c]] += float(ious[r, c])
            denom_pot = (
                gt_id_frame_counts[:, None] + pred_id_frame_counts[None, :] - pot_matches
            ).clip(min=1e-9)
            global_align = pot_matches / denom_pot

            tpa_mat = np.zeros((n_u_gt, n_u_pred), dtype=np.int64)
            tp_a = 0
            fp_a = 0
            fn_a = 0
            matched_pairs_alpha: List[Tuple[int, int]] = []

            for f_idx in range(n_frames):
                ious = frame_ious[f_idx]
                g_arr, p_arr = frame_g_idx[f_idx], frame_p_idx[f_idx]
                ng, np_f = g_arr.size, p_arr.size
                if ng == 0:
                    fp_a += np_f
                    continue
                if np_f == 0:
                    fn_a += ng
                    continue
                score_mat = ious * (1e-5 + global_align[np.ix_(g_arr, p_arr)])
                cost_mat = np.where(ious >= alpha, -score_mat, 1e6)
                r_ind, c_ind = linear_sum_assignment(cost_mat)
                m_cnt = 0
                for r, c in zip(r_ind, c_ind):
                    if float(ious[r, c]) >= alpha:
                        gi, pi = int(g_arr[r]), int(p_arr[c])
                        tpa_mat[gi, pi] += 1
                        matched_pairs_alpha.append((gi, pi))
                        m_cnt += 1
                tp_a += m_cnt
                fn_a += ng - m_cnt
                fp_a += np_f - m_cnt

            if tp_a == 0:
                hota_alphas.append(0.0)
                deta_alphas.append(0.0)
                assa_alphas.append(0.0)
            else:
                deta = float(tp_a) / float(tp_a + fn_a + fp_a)
                ass_scores = [
                    float(tpa_mat[gi, pi])
                    / float(max(1, int(gt_id_frame_counts[gi] + pred_id_frame_counts[pi] - tpa_mat[gi, pi])))
                    for gi, pi in matched_pairs_alpha
                ]
                assa = float(np.mean(ass_scores))
                deta_alphas.append(deta)
                assa_alphas.append(assa)
                hota_alphas.append(math.sqrt(max(0.0, deta * assa)))

    hota = float(np.mean(hota_alphas)) * 100.0 if hota_alphas else float("nan")
    deta_mean = float(np.mean(deta_alphas)) * 100.0 if deta_alphas else float("nan")
    assa_mean = float(np.mean(assa_alphas)) * 100.0 if assa_alphas else float("nan")

    return {
        "mota_pct": round(mota, 3) if math.isfinite(mota) else None,
        "idf1_pct": round(idf1, 3) if math.isfinite(idf1) else None,
        "hota_pct": round(hota, 3) if math.isfinite(hota) else None,
        "deta_pct": round(deta_mean, 3) if math.isfinite(deta_mean) else None,
        "assa_pct": round(assa_mean, 3) if math.isfinite(assa_mean) else None,
        "motp_pct": round(motp, 3),
        "idsw": int(total_idsw),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "gt": int(total_gt),
        "idtp": int(idtp),
        "idfp": int(idfp),
        "idfn": int(idfn),
        "hota_per_alpha": hota_alphas,
    }


def pool_mot_tracking_summaries(seq_metrics: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pool CLEAR-MOT, IDF1, and GT-weighted HOTA across multiple video sequences."""
    tot_gt = sum(int(m["gt"]) for m in seq_metrics)
    tot_tp = sum(int(m["tp"]) for m in seq_metrics)
    tot_fp = sum(int(m["fp"]) for m in seq_metrics)
    tot_fn = sum(int(m["fn"]) for m in seq_metrics)
    tot_idsw = sum(int(m["idsw"]) for m in seq_metrics)
    tot_idtp = sum(int(m["idtp"]) for m in seq_metrics)
    tot_idfp = sum(int(m["idfp"]) for m in seq_metrics)
    tot_idfn = sum(int(m["idfn"]) for m in seq_metrics)

    mota = (1.0 - float(tot_fn + tot_fp + tot_idsw) / float(max(1, tot_gt))) * 100.0 if tot_gt > 0 else None
    idf1 = (2.0 * float(tot_idtp) / float(max(1, 2 * tot_idtp + tot_idfp + tot_idfn))) * 100.0 if tot_gt > 0 else None
    valid_hota = [(float(m["hota_pct"]), int(m["gt"])) for m in seq_metrics if m.get("hota_pct") is not None and int(m["gt"]) > 0]
    hota = (
        sum(val * w for val, w in valid_hota) / float(sum(w for _, w in valid_hota))
        if valid_hota
        else None
    )
    return {
        "mota_pct": round(mota, 3) if mota is not None else None,
        "idf1_pct": round(idf1, 3) if idf1 is not None else None,
        "hota_pct": round(hota, 3) if hota is not None else None,
        "idsw": tot_idsw,
        "tp": tot_tp,
        "fp": tot_fp,
        "fn": tot_fn,
        "gt": tot_gt,
    }


# ======================================================================================
# 2. External & Standard SOTA Pre-Processors / Codec-ROI / Compressed-Video Fine-Tuning
# ======================================================================================


def try_setup_official_scope(
    scope_ckpt: str,
    scope_repo_dir: str,
    auto_download: bool,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Loads official Dani et al. 2026 `miccunifi/SCOPE` (`scope/inference.py` + `best_val.pt`).
    Uses official BPG/HEVC conditioning (`codec_family='intra_smooth'`, `severity = qp / 51.0`).
    """
    repo_path = Path(scope_repo_dir)
    if auto_download and not (repo_path / "scope" / "inference.py").is_file():
        try:
            if repo_path.exists():
                shutil.rmtree(repo_path, ignore_errors=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/miccunifi/SCOPE.git", str(repo_path)],
                check=True,
                timeout=120,
            )
        except Exception as exc:
            return {"available": False, "reason": f"Failed to clone miccunifi/SCOPE: {exc}"}

    ckpt_path = Path(scope_ckpt) if scope_ckpt else None
    if (ckpt_path is None or not ckpt_path.is_file()) and auto_download:
        weights_dir = Path("/tmp/miccunifi_scope_weights")
        if not weights_dir.exists() or not list(weights_dir.rglob("*.pt")):
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "gdown"], check=False, timeout=90)
                weights_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "gdown",
                        "--folder",
                        "--remaining-ok",
                        "https://drive.google.com/drive/folders/1YbB1M0Bdp1QZl-aoGNWCCsJRMgzROGES",
                        "-O",
                        str(weights_dir),
                    ],
                    check=True,
                    timeout=300,
                )
            except Exception as exc:
                return {"available": False, "reason": f"Failed to download miccunifi/SCOPE weights from Google Drive: {exc}"}
        cand_pts = sorted(weights_dir.rglob("best_val.pt")) or sorted(weights_dir.rglob("*.pt"))
        full_pts = [p for p in cand_pts if "FULL" in str(p).upper()]
        if full_pts:
            ckpt_path = full_pts[0]
        elif cand_pts:
            ckpt_path = cand_pts[0]

    if ckpt_path is None or not ckpt_path.is_file() or not (repo_path / "scope" / "inference.py").is_file():
        return {
            "available": False,
            "reason": "Official miccunifi/SCOPE repo or pretrained best_val.pt not present (pass --auto_download_scope on internet-enabled runner)",
        }

    try:
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        from scope.inference import load_scope_model

        scope_model, fam_map = load_scope_model(str(ckpt_path), device)
        fam_name = (
            "intra_smooth"
            if "intra_smooth" in fam_map
            else ("bpg_like" if "bpg_like" in fam_map else next(iter(fam_map)))
        )
        return {
            "available": True,
            "model": scope_model,
            "family_id": int(fam_map[fam_name]),
            "family_name": fam_name,
            "ckpt_path": str(ckpt_path),
            "ckpt_sha256": sha256_file(ckpt_path),
        }
    except Exception as exc:
        return {"available": False, "reason": f"Failed to initialize miccunifi/SCOPE model: {exc}"}


@torch.no_grad()
def apply_official_scope_clip(
    video: torch.Tensor, qp: int, scope_bundle: Dict[str, Any], chunk_size: int = 8
) -> torch.Tensor:
    """Apply official `miccunifi/SCOPE` frame-by-frame with `severity = qp / 51.0`."""
    b, t, c, h, w = video.shape
    flat = torch.clamp(video.view(b * t, c, h, w), 0.0, 1.0)
    scope_model = scope_bundle["model"]
    fam_id = int(scope_bundle["family_id"])
    sev_scalar = min(1.0, max(0.0, float(qp) / 51.0))

    outs = []
    for start in range(0, flat.shape[0], chunk_size):
        sub = flat[start : start + chunk_size]
        bsz = sub.shape[0]
        sev = torch.full((bsz,), sev_scalar, device=sub.device, dtype=sub.dtype)
        fam = torch.full((bsz,), fam_id, device=sub.device, dtype=torch.long)
        edited, *_ = scope_model(sub, severity=sev, codec_family=fam)
        outs.append(torch.clamp(edited, 0.0, 1.0))
    return torch.cat(outs, dim=0).view(b, t, c, h, w)


@torch.no_grad()
def encode_mpeg_vcm_spatial_resampling(
    video: torch.Tensor,
    qp: int,
    codec: HybridX265STE,
    gop_len: int,
    bframes: int = 0,
) -> Dict[str, Any]:
    """
    Standard JVET / MPEG-VCM Spatial-Resampling baseline:
      1. Anti-aliased downscale to 16px-aligned reduced resolution `(H_s, W_s)` per CRF.
      2. Encode `(H_s, W_s)` with real `libx265` (`yuv420p`, `gop_len`).
      3. Express bitstream `total_bpp` and `p_frame_bpp` strictly in bits per ORIGINAL `(H * W)` pixel!
      4. Upscale decoded clip back to `(H, W)` via bicubic interpolation before machine vision inference.
    """
    scale_by_qp = {27: 0.85, 32: 0.75, 37: 0.65, 42: 0.55}
    scale = scale_by_qp.get(qp, max(0.50, min(0.90, 1.0 - 0.015 * (qp - 20))))
    b, t, c, h, w = video.shape
    hs = max(32, int(round((h * scale) / 16.0)) * 16)
    ws = max(32, int(round((w * scale) / 16.0)) * 16)

    down = F.interpolate(
        video.view(b * t, c, h, w), size=(hs, ws), mode="bilinear", align_corners=False, antialias=True
    ).view(b, t, c, hs, ws)
    c_out = _call_evaluate_real_codec(codec, down, qp, None, family_id=0, gop_len=gop_len, bframes=bframes)
    up_recon = torch.clamp(
        F.interpolate(
            c_out["video_recon"].view(b * t, c, hs, ws), size=(h, w), mode="bicubic", align_corners=False
        ),
        0.0,
        1.0,
    ).view(b, t, c, h, w)

    area_ratio = float(hs * ws) / float(h * w)
    out = dict(c_out)
    out["video_recon"] = up_recon
    out["height"] = h
    out["width"] = w
    out["total_bpp"] = float(c_out["total_bpp"]) * area_ratio
    out["i_frame_bpp"] = float(c_out["i_frame_bpp"]) * area_ratio
    out["p_frame_bpp"] = float(c_out["p_frame_bpp"]) * area_ratio
    out["rate_per_frame"] = c_out["rate_per_frame"] * area_ratio
    return out


@torch.no_grad()
def build_detector_roi_dqp_offsets(
    video: torch.Tensor,
    roi_detector_fn: Any,
    qg_size: int = 16,
    bg_dqp: float = 4.0,
    context_pad: int = 12,
) -> torch.Tensor:
    """
    Detector-ROI Codec Control baseline (`F06`):
    Runs an object detector on clean frames, builds binary foreground boxes + halo mask,
    and emits non-negative background `+DeltaQP` (`[B, T, H//16, W//16]`) with ZERO pixel modification.
    """
    b, t, c, h, w = video.shape
    flat = torch.clamp(video.view(b * t, c, h, w), 0.0, 1.0)
    boxes_per_frame: List[torch.Tensor] = roi_detector_fn(flat)
    fg_mask = torch.zeros((b * t, 1, h, w), device=video.device, dtype=torch.float32)
    for f_idx, boxes in enumerate(boxes_per_frame):
        for box in boxes:
            x1 = max(0, int(float(box[0]) - context_pad))
            y1 = max(0, int(float(box[1]) - context_pad))
            x2 = min(w, int(float(box[2]) + context_pad))
            y2 = min(h, int(float(box[3]) + context_pad))
            if x2 > x1 and y2 > y1:
                fg_mask[f_idx, 0, y1:y2, x1:x2] = 1.0
    qg_fg = F.max_pool2d(fg_mask, kernel_size=qg_size, stride=qg_size)
    # 3x3 QG halo guard matching T-SCOPE v3 so +dQP never clips box boundaries
    qg_halo = F.max_pool2d(qg_fg, kernel_size=3, stride=1, padding=1)
    dqp = bg_dqp * (1.0 - qg_halo[:, 0])
    return dqp.view(b, t, qg_halo.shape[-2], qg_halo.shape[-1])


def train_compressed_video_finetuned_frcnn(
    data_root: str,
    img_hw: Tuple[int, int],
    codec: HybridX265STE,
    device: torch.device,
    epochs: int = 2,
    max_clips_per_epoch: int = 80,
) -> nn.Module:
    """
    Compressed-Video Fine-Tuned Detector baseline (`F06`):
    Adapts Faster R-CNN ResNet50-FPN v2 (FPN + RPN + ROI box head) strictly on `anchor_x265`-compressed
    disjoint `MOT17` train sequences (`02, 04, 05, 10`) across `CRFS=[27, 32, 37, 42]`. Zero test leakage.
    """
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

    det = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT).to(device)
    for name, p in det.named_parameters():
        p.requires_grad_(not name.startswith("backbone.body."))

    train_ds = _construct_real_dataset(
        data_root=data_root,
        shard_id=0,
        num_shards=1,
        clip_len=TRAIN_CLIP_LEN,
        stride=TRAIN_CLIP_LEN * 2,
        img_hw=img_hw,
        is_train=True,
        enforce_disjoint_split=True,
        split="train",
    )
    if len(train_ds) == 0:
        det.eval()
        return det

    trainable_params = [p for p in det.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
    rng = np.random.default_rng(2026)

    det.train()
    for ep in range(1, epochs + 1):
        indices = rng.permutation(len(train_ds))[: min(len(train_ds), max_clips_per_epoch)]
        ep_loss = 0.0
        steps = 0
        for idx in indices.tolist():
            sample = train_ds[int(idx)]
            video = sample["video"].unsqueeze(0).to(device)
            gt_all = sample["gt_boxes"]  # [T, 256, 4]
            qp = int(CRFS[steps % len(CRFS)])
            with torch.no_grad():
                enc = _call_evaluate_real_codec(codec, video, qp, None, family_id=0, gop_len=video.shape[1], bframes=0)
                recon = enc["video_recon"][0]  # [T, 3, H, W]

            imgs_step: List[torch.Tensor] = []
            targets_step: List[Dict[str, torch.Tensor]] = []
            for f_i in range(recon.shape[0]):
                valid_gt = gt_all[f_i][gt_all[f_i][:, 0] >= 0.0]
                if valid_gt.shape[0] == 0:
                    continue
                imgs_step.append(torch.clamp(recon[f_i], 0.0, 1.0))
                targets_step.append(
                    {
                        "boxes": valid_gt.to(device),
                        "labels": torch.ones((valid_gt.shape[0],), dtype=torch.int64, device=device),
                    }
                )
            if not imgs_step:
                continue

            optimizer.zero_grad(set_to_none=True)
            loss_dict = det(imgs_step[:2], targets_step[:2])
            loss = sum(loss_dict.values())
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 2.0)
                optimizer.step()
                ep_loss += float(loss.item())
                steps += 1
        print(f"  [CompressedFineTune FRCNN] epoch {ep}/{epochs} | steps={steps} | mean_loss={ep_loss / max(1, steps):.4f}", flush=True)

    det.eval()
    for p in det.parameters():
        p.requires_grad_(False)
    return det


# ======================================================================================
# 3. Multi-Class Dataset AP Helper (for KITTI / SFU-HW-Objects-v1)
# ======================================================================================


def compute_multiclass_dataset_coco_metrics(
    all_gt_boxes: Sequence[torch.Tensor],
    all_gt_classes: Sequence[torch.Tensor],
    all_pred_boxes: Sequence[torch.Tensor],
    all_pred_labels: Sequence[torch.Tensor],
    all_pred_scores: Sequence[torch.Tensor],
    all_ignore_boxes: Optional[Sequence[torch.Tensor]] = None,
    candidate_classes: Sequence[int] = (1, 2, 3, 4, 6, 8),
) -> Dict[str, Any]:
    """
    Computes multi-class dataset-level 101-point COCO mAP@50 and mAP@[.50:.95]
    across all active COCO classes present in ground truth (`F11` & `F25`).
    """
    per_class: Dict[str, Dict[str, float]] = {}
    ap50_vals: List[float] = []
    ap95_vals: List[float] = []

    for cls_id in candidate_classes:
        cls_gt: List[torch.Tensor] = []
        cls_pb: List[torch.Tensor] = []
        cls_ps: List[torch.Tensor] = []
        tot_c_gt = 0
        for f_i in range(len(all_gt_boxes)):
            g_b = all_gt_boxes[f_i]
            g_c = all_gt_classes[f_i]
            m_g = g_c == cls_id
            sub_g = g_b[m_g] if g_b.shape[0] > 0 else torch.zeros((0, 4), dtype=torch.float32)
            tot_c_gt += int(sub_g.shape[0])
            cls_gt.append(sub_g)

            p_b = all_pred_boxes[f_i]
            p_l = all_pred_labels[f_i]
            p_s = all_pred_scores[f_i]
            m_p = p_l == cls_id
            cls_pb.append(p_b[m_p] if p_b.shape[0] > 0 else torch.zeros((0, 4), dtype=torch.float32))
            cls_ps.append(p_s[m_p] if p_s.numel() > 0 else torch.zeros((0,), dtype=torch.float32))

        if tot_c_gt == 0:
            continue
        res_c = FrozenFPNDetectorTeacher.compute_dataset_level_coco_metrics(
            cls_gt, cls_pb, cls_ps, all_ignore_boxes=list(all_ignore_boxes) if all_ignore_boxes is not None else None
        )
        if math.isfinite(res_c["gt_ap50_pct"]) and math.isfinite(res_c["gt_ap50_95_pct"]):
            per_class[str(cls_id)] = {
                "gt_count": tot_c_gt,
                "ap50_pct": res_c["gt_ap50_pct"],
                "ap50_95_pct": res_c["gt_ap50_95_pct"],
            }
            ap50_vals.append(res_c["gt_ap50_pct"])
            ap95_vals.append(res_c["gt_ap50_95_pct"])

    return {
        "map50_pct": round(float(np.mean(ap50_vals)), 3) if ap50_vals else float("nan"),
        "map50_95_pct": round(float(np.mean(ap95_vals)), 3) if ap95_vals else float("nan"),
        "active_classes": per_class,
    }


# ======================================================================================
# 4. Complexity, Latency, Peak VRAM, GMACs & Codec Wall-Clock Profiler (F16, F17, F24)
# ======================================================================================


def _estimate_tensor_gmacs(model: nn.Module, forward_fn: Any, num_frames: int) -> Optional[float]:
    try:
        from torch.utils.flop_counter import FlopCounterMode

        with FlopCounterMode(display=False) as fc:
            forward_fn()
        # 1 MAC = 2 FLOPs
        return round(float(fc.get_total_flops()) / (2.0 * 1e9 * max(1, num_frames)), 4)
    except Exception:
        if hasattr(model, "estimate_gmacs"):
            try:
                return float(model.estimate_gmacs())
            except Exception:
                return None
        return None


@torch.no_grad()
def benchmark_complexity_and_runtime(
    models_dict: Dict[str, TemporalSCOPE],
    scope_bundle: Dict[str, Any],
    codec: HybridX265STE,
    device: torch.device,
    img_hw: Tuple[int, int] = (IMG_H, IMG_W),
    gop_len: int = EVAL_GOP_LEN,
    warmup_iters: int = 3,
    timed_iters: int = 12,
) -> Dict[str, Any]:
    """
    Profiles neural pre-processing latency (mean/p95 ms/frame, FPS, IRLS vs U-Net split,
    peak GPU VRAM MB, GMACs/frame) and real CPU `libx265`/`libx264` encoding wall-clock time.
    """
    h, w = img_hw
    gen = torch.Generator().manual_seed(2026)
    # Realistic textured moving clip so IRLS, ConvGRU, and x265 ME perform realistic work
    base_f = 0.25 + 0.50 * torch.rand((1, 1, 3, h, w), generator=gen)
    clip = torch.clamp(base_f + 0.03 * torch.randn((1, gop_len, 3, h, w), generator=gen), 0.0, 1.0).to(device)
    fam0 = torch.tensor([0], device=device, dtype=torch.long)
    sev = torch.tensor([0.40], device=device, dtype=torch.float32)

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _time_callable(fn: Any) -> Tuple[float, float, float]:
        for _ in range(warmup_iters):
            fn()
        _sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        latencies_ms: List[float] = []
        for _ in range(timed_iters):
            _sync()
            t0 = time.perf_counter()
            fn()
            _sync()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0 / float(gop_len))
        peak_mb = (
            float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
            if device.type == "cuda"
            else 0.0
        )
        return float(np.mean(latencies_ms)), float(np.percentile(latencies_ms, 95)), peak_mb

    # 1. Isolate 6-DOF Cauchy-IRLS motion estimation latency
    prev_flat = clip[:, :-1].reshape(gop_len - 1, 3, h, w)
    curr_flat = clip[:, 1:].reshape(gop_len - 1, 3, h, w)
    irls_mean_ms, irls_p95_ms, irls_vram_mb = _time_callable(
        lambda: estimate_global_affine_irls(prev_flat, curr_flat)
    )

    neural_profiles: Dict[str, Any] = {
        "cauchy_irls_6dof_motion_estimator_only": {
            "mean_ms_per_frame": round(irls_mean_ms, 3),
            "p95_ms_per_frame": round(irls_p95_ms, 3),
            "fps": round(1000.0 / max(1e-6, irls_mean_ms), 2),
            "peak_vram_mb": round(irls_vram_mb, 2),
        }
    }

    variant_modes = [
        ("full_tscope", models_dict["full_tscope"], False, False),
        ("iwa_gop_qp_reuse", models_dict["iwa_gop_qp_reuse"], True, False),
        ("spatial_only_scope", models_dict["spatial_only_scope"], False, True),
    ]
    for label, m_inst, reuse_qp, dis_temp in variant_modes:
        params_m = sum(p.numel() for p in m_inst.parameters()) / 1e6
        fn_run = lambda m=m_inst, r=reuse_qp, d=dis_temp: m(clip, fam0, sev, reuse_iframe_qp=r, disable_temporal=d)
        mean_ms, p95_ms, peak_mb = _time_callable(fn_run)
        gmacs = _estimate_tensor_gmacs(m_inst, fn_run, gop_len)
        fwd_stats = copy.deepcopy(getattr(m_inst, "last_forward_stats", None))
        neural_profiles[label] = {
            "params_millions": round(params_m, 3),
            "gmacs_per_frame": gmacs,
            "mean_ms_per_frame": round(mean_ms, 3),
            "p95_ms_per_frame": round(p95_ms, 3),
            "unet_plus_heads_ms_per_frame_est": round(max(0.0, mean_ms - (0.0 if dis_temp else irls_mean_ms)), 3),
            "fps": round(1000.0 / max(1e-6, mean_ms), 2),
            "peak_vram_mb": round(peak_mb, 2),
            "forward_stats": fwd_stats,
        }

    if scope_bundle.get("available"):
        s_model = scope_bundle["model"]
        s_params = sum(p.numel() for p in s_model.parameters()) / 1e6
        fn_scope = lambda: apply_official_scope_clip(clip, 32, scope_bundle, chunk_size=8)
        s_mean_ms, s_p95_ms, s_peak_mb = _time_callable(fn_scope)
        s_gmacs = _estimate_tensor_gmacs(s_model, fn_scope, gop_len)
        neural_profiles["official_miccunifi_scope"] = {
            "params_millions": round(s_params, 3),
            "gmacs_per_frame": s_gmacs,
            "mean_ms_per_frame": round(s_mean_ms, 3),
            "p95_ms_per_frame": round(s_p95_ms, 3),
            "fps": round(1000.0 / max(1e-6, s_mean_ms), 2),
            "peak_vram_mb": round(s_peak_mb, 2),
        }

    # 2. CPU Codec Wall-Clock Encoding Time (CRF 32, gop_len=32, yuv420p)
    out_full = models_dict["full_tscope"](clip, fam0, sev, reuse_iframe_qp=False, disable_temporal=False)
    out_iwa = models_dict["iwa_gop_qp_reuse"](clip, fam0, sev, reuse_iframe_qp=True, disable_temporal=False)
    # Ensure non-trivial background +dQP for realistic filtergraph timing
    dqp_dyn = out_full["ctu_qp_offsets"].clone()
    if float(dqp_dyn.abs().max().item()) < 0.2:
        dqp_dyn[:, :, :4, :4] = 3.0
        dqp_dyn[:, -1, :4, :4] = 3.5
    dqp_stat = out_iwa["ctu_qp_offsets"].clone()
    if float(dqp_stat.abs().max().item()) < 0.2:
        dqp_stat[:, :, :4, :4] = 3.0

    codec_profiles: Dict[str, Any] = {}
    codec_cases = [
        ("anchor_libx265_unedited", clip, None, 0),
        ("tscope_preedit_libx265_no_addroi", out_full["i_edit"], None, 0),
        ("tscope_iwa_libx265_static_addroi", out_iwa["i_edit"], dqp_stat, 0),
        ("tscope_full_libx265_dynamic_addroi", out_full["i_edit"], dqp_dyn, 0),
        ("anchor_libx264_unedited", clip, None, 1),
        ("tscope_full_libx264_dynamic_addroi", out_full["i_edit"], dqp_dyn, 1),
    ]
    enc_runs = 3
    for c_label, v_in, qp_in, fam_id in codec_cases:
        times_ms: List[float] = []
        last_bpp = 0.0
        for _ in range(enc_runs):
            t0 = time.perf_counter()
            c_res = _call_evaluate_real_codec(codec, v_in, 32, qp_in, family_id=fam_id, gop_len=gop_len, bframes=0)
            times_ms.append((time.perf_counter() - t0) * 1000.0 / float(gop_len))
            last_bpp = float(c_res["total_bpp"])
        m_ms = float(np.mean(times_ms))
        codec_profiles[c_label] = {
            "encoder": "libx265" if fam_id == 0 else "libx264",
            "mean_encode_decode_ms_per_frame": round(m_ms, 3),
            "encode_decode_fps": round(1000.0 / max(1e-6, m_ms), 2),
            "crf32_total_bpp": round(last_bpp, 5),
        }

    full_gmacs = neural_profiles["full_tscope"].get("gmacs_per_frame")
    iwa_gmacs = neural_profiles["iwa_gop_qp_reuse"].get("gmacs_per_frame")
    gmacs_saving_pct = (
        round((1.0 - float(iwa_gmacs) / float(full_gmacs)) * 100.0, 2)
        if (full_gmacs and iwa_gmacs and full_gmacs > 0)
        else None
    )

    return {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "resolution": f"{w}x{h}",
        "gop_len": gop_len,
        "iwa_vs_full_gmacs_reduction_pct": gmacs_saving_pct,
        "neural_pre_editors": neural_profiles,
        "codec_wall_clock": codec_profiles,
    }


# ======================================================================================
# 5. Main End-to-End Orchestration (`sota`, `tracking`, `second_dataset`, `complexity`)
# ======================================================================================


def _load_soup_or_fresh(soup_dir: Path, variant: str, max_dqp: float, device: torch.device) -> Tuple[TemporalSCOPE, Optional[str]]:
    path = soup_dir / f"tscope_soup_{variant}.pt"
    model = TemporalSCOPE(base_ch=48, num_families=2, cond_dim=64, qg_size=16, max_delta_qp=max_dqp).to(device)
    if path.is_file():
        ckpt = torch.load(path, map_location=device, weights_only=True)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=True)
        model.eval()
        return model, sha256_file(path)
    raise FileNotFoundError(f"Missing required v3 soup checkpoint: {path}")


def _safe_bdbr(r1: Sequence[float], a1: Sequence[float], r2: Sequence[float], a2: Sequence[float]) -> Optional[float]:
    try:
        val = bjontegaard_delta_rate(r1, a1, r2, a2)
        return round(float(val), 2) if math.isfinite(val) else None
    except ValueError:
        return None


@torch.no_grad()
def run_sota_and_tracking_on_mot17(
    args: argparse.Namespace,
    data_root: str,
    models_dict: Dict[str, TemporalSCOPE],
    scope_bundle: Dict[str, Any],
    codec: HybridX265STE,
    fpn_teacher: FrozenFPNDetectorTeacher,
    device: torch.device,
    run_sota: bool,
    run_tracking: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    img_hw = (args.img_h, args.img_w)
    dataset = _construct_real_dataset(
        data_root=data_root,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        clip_len=args.gop_len,
        stride=args.gop_len,
        img_hw=img_hw,
        is_train=False,
        enforce_disjoint_split=True,
        split="test",
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No held-out MOT17 test clips found at {data_root} with gop_len={args.gop_len}")

    # Build lightweight detector-ROI box extractor & compressed-video fine-tuned detector when SOTA is active
    roi_det_fn = None
    finetuned_frcnn = None
    if run_sota:
        if fpn_teacher.use_yolov8s and fpn_teacher.yolo_wrapper is not None:
            def _yolo_roi_fn(flat_imgs: torch.Tensor) -> List[torch.Tensor]:
                res = fpn_teacher.yolo_wrapper.predict(
                    source=flat_imgs, imgsz=max(img_hw), conf=0.25, classes=[0], verbose=False
                )
                return [r.boxes.xyxy.detach() for r in res]
            roi_det_fn = _yolo_roi_fn
        else:
            def _frcnn_roi_fn(flat_imgs: torch.Tensor) -> List[torch.Tensor]:
                outs = []
                for st in range(0, flat_imgs.shape[0], 8):
                    dets = fpn_teacher.detector(list(flat_imgs[st : st + 8]))
                    for d in dets:
                        m = (d["labels"] == 1) & (d["scores"] >= 0.35)
                        outs.append(d["boxes"][m].detach())
                return outs
            roi_det_fn = _frcnn_roi_fn

        if args.finetune_det_epochs > 0:
            print("[SOTA] Fine-tuning Faster R-CNN v2 baseline on anchor_x265-compressed disjoint MOT17 train clips...", flush=True)
            with torch.enable_grad():
                finetuned_frcnn = train_compressed_video_finetuned_frcnn(
                    data_root=data_root,
                    img_hw=img_hw,
                    codec=codec,
                    device=device,
                    epochs=args.finetune_det_epochs,
                )

    active_methods = ["anchor_x265", "tscope_spatial_only", "tscope_iwa_gop_qp_reuse", "tscope_full"]
    if run_sota:
        active_methods.extend(["mpeg_vcm_spatial_resampling", "detector_roi_codec_addroi"])
        if scope_bundle.get("available"):
            active_methods.append("official_miccunifi_scope")
        if finetuned_frcnn is not None:
            active_methods.append("compressed_video_finetuned_frcnn")

    clip_records: Dict[Tuple[str, int], List[Dict[str, Any]]] = {(m, q): [] for m in active_methods for q in CRFS}
    # Sequence-ordered per-frame detections for ByteTrack tracking evaluation
    seq_frame_dets: Dict[Tuple[str, int, str], List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}

    t0 = time.perf_counter()
    for idx in range(len(dataset)):
        sample = dataset[idx]
        seq_name = sample["seq_name"].split("-FRCNN")[0]
        video = sample["video"].unsqueeze(0).to(device)
        m_det = sample["m_det"].unsqueeze(0).to(device)
        gt_boxes = sample["gt_boxes"].unsqueeze(0).to(device)
        ignore_boxes = sample["ignore_boxes"].unsqueeze(0).to(device)
        gt_tids = sample.get("gt_track_ids")
        b, t, c, h, w = video.shape
        clean_flat = video.view(b * t, c, h, w)

        clean_dets = []
        for st in range(0, clean_flat.shape[0], 8):
            clean_dets.extend(fpn_teacher.detector(list(torch.clamp(clean_flat[st : st + 8], 0.0, 1.0))))
        cached_clean_info = {"t_feats": fpn_teacher.extract_fast_pyramid(clean_flat), "clean_dets": clean_dets}

        roi_dqp_clip = build_detector_roi_dqp_offsets(video, roi_det_fn) if (run_sota and roi_det_fn is not None) else None
        fam0 = torch.tensor([0], device=device, dtype=torch.long)

        for qp in CRFS:
            sev = torch.tensor([(qp - 22.0) / 25.0], device=device, dtype=torch.float32)

            encoded_by_method: Dict[str, Dict[str, Any]] = {}
            encoded_by_method["anchor_x265"] = _call_evaluate_real_codec(
                codec, video, qp, None, family_id=0, gop_len=args.gop_len, bframes=args.bframes
            )

            p_sp = models_dict["spatial_only_scope"](video, fam0, sev, reuse_iframe_qp=False, disable_temporal=True)
            encoded_by_method["tscope_spatial_only"] = _call_evaluate_real_codec(
                codec, p_sp["i_edit"], qp, None, family_id=0, gop_len=args.gop_len, bframes=args.bframes
            )

            p_iwa = models_dict["iwa_gop_qp_reuse"](video, fam0, sev, reuse_iframe_qp=True, disable_temporal=False)
            encoded_by_method["tscope_iwa_gop_qp_reuse"] = _call_evaluate_real_codec(
                codec, p_iwa["i_edit"], qp, p_iwa["ctu_qp_offsets"], family_id=0, gop_len=args.gop_len, bframes=args.bframes
            )

            p_full = models_dict["full_tscope"](video, fam0, sev, reuse_iframe_qp=False, disable_temporal=False)
            encoded_by_method["tscope_full"] = _call_evaluate_real_codec(
                codec, p_full["i_edit"], qp, p_full["ctu_qp_offsets"], family_id=0, gop_len=args.gop_len, bframes=args.bframes
            )

            if run_sota:
                encoded_by_method["mpeg_vcm_spatial_resampling"] = encode_mpeg_vcm_spatial_resampling(
                    video, qp, codec, gop_len=args.gop_len, bframes=args.bframes
                )
                if roi_dqp_clip is not None:
                    encoded_by_method["detector_roi_codec_addroi"] = _call_evaluate_real_codec(
                        codec, video, qp, roi_dqp_clip, family_id=0, gop_len=args.gop_len, bframes=args.bframes
                    )
                if scope_bundle.get("available"):
                    v_scope = apply_official_scope_clip(video, qp, scope_bundle, chunk_size=8)
                    encoded_by_method["official_miccunifi_scope"] = _call_evaluate_real_codec(
                        codec, v_scope, qp, None, family_id=0, gop_len=args.gop_len, bframes=args.bframes
                    )
                if finetuned_frcnn is not None:
                    # Uses exact same anchor_x265 bitstream, evaluated with compressed-domain fine-tuned detector!
                    encoded_by_method["compressed_video_finetuned_frcnn"] = encoded_by_method["anchor_x265"]

            for m_name, c_out in encoded_by_method.items():
                if m_name == "compressed_video_finetuned_frcnn" and finetuned_frcnn is not None:
                    recon_flat = c_out["video_recon"].view(b * t, c, h, w)
                    ft_dets = []
                    for st in range(0, recon_flat.shape[0], 8):
                        ft_dets.extend(finetuned_frcnn(list(torch.clamp(recon_flat[st : st + 8], 0.0, 1.0))))
                    gt_f = gt_boxes[0].detach().cpu()
                    ig_f = ignore_boxes[0].detach().cpu()
                    raw_d = {"gt_boxes": [], "ignore_boxes": [], "pred_boxes": [], "pred_scores": []}
                    for f_i, d_out in enumerate(ft_dets):
                        pm = (d_out["labels"] == 1) & (d_out["scores"] >= 0.01)
                        raw_d["pred_boxes"].append(d_out["boxes"][pm].detach().cpu())
                        raw_d["pred_scores"].append(d_out["scores"][pm].detach().cpu())
                        raw_d["gt_boxes"].append(gt_f[f_i][gt_f[f_i][:, 0] >= 0.0])
                        raw_d["ignore_boxes"].append(ig_f[f_i][ig_f[f_i][:, 0] >= 0.0])
                else:
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

                raw_keys = ["gt_boxes", "ignore_boxes", "pred_boxes", "pred_scores"]
                if "yolo_pred_boxes" in raw_d:
                    raw_keys.extend(["yolo_pred_boxes", "yolo_pred_scores"])
                clip_records[(m_name, qp)].append(
                    {
                        "frame_ids": sample["frame_ids"],
                        "packet_sizes": c_out["packet_sizes"],
                        "frame_types": c_out["frame_types"],
                        "height": h,
                        "width": w,
                        "raw_detections": {k: [v.tolist() for v in raw_d[k]] for k in raw_keys},
                    }
                )

                if run_tracking and gt_tids is not None:
                    gt_f_cpu = gt_boxes[0].detach().cpu()
                    tids_cpu = gt_tids.detach().cpu()
                    for f_i, fid_str in enumerate(sample["frame_ids"]):
                        f_num = int(fid_str.split("/")[-1])
                        v_mask = gt_f_cpu[f_i][:, 0] >= 0.0
                        gb_np = raw_d["gt_boxes"][f_i].numpy()
                        gid_np = tids_cpu[f_i][v_mask].numpy().astype(np.int64)
                        ig_np = raw_d["ignore_boxes"][f_i].numpy()
                        pb_np = raw_d["pred_boxes"][f_i].numpy()
                        ps_np = raw_d["pred_scores"][f_i].numpy()
                        seq_frame_dets.setdefault((m_name, qp, seq_name), []).append(
                            (f_num, gb_np, gid_np, ig_np, pb_np, ps_np)
                        )

        if (idx + 1) % 3 == 0 or (idx + 1) == len(dataset):
            print(f"[SOTA+Tracking] Clip {idx + 1}/{len(dataset)} ({time.perf_counter() - t0:.1f}s)", flush=True)

    # Summarize SOTA AP & BDBR + 10k clip-bootstrap CIs
    num_clips = len(dataset)
    rng = np.random.default_rng(2026)
    boot_w = (
        rng.multinomial(num_clips, np.full(num_clips, 1.0 / float(num_clips)), size=args.bootstrap_resamples).astype(np.float64)
        if (args.bootstrap_resamples > 0 and num_clips >= 2)
        else None
    )

    sota_methods_out: Dict[str, Any] = {}
    boot_curves: Dict[str, Optional[Dict[str, np.ndarray]]] = {}
    for m_name in active_methods:
        if boot_w is not None:
            boot_curves[m_name] = _PreparedClipBootstrapCurve(
                {q: clip_records[(m_name, q)] for q in CRFS}, CRFS
            ).evaluate_resampled_curves(boot_w)
        else:
            boot_curves[m_name] = None

        pqp: Dict[str, Any] = {}
        for q in CRFS:
            recs = clip_records[(m_name, q)]
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
            all_gt = [torch.tensor(b, dtype=torch.float32).reshape(-1, 4) for r in recs for b in r["raw_detections"]["gt_boxes"]]
            all_ig = [torch.tensor(b, dtype=torch.float32).reshape(-1, 4) for r in recs for b in r["raw_detections"]["ignore_boxes"]]
            all_pb = [torch.tensor(b, dtype=torch.float32).reshape(-1, 4) for r in recs for b in r["raw_detections"]["pred_boxes"]]
            all_ps = [torch.tensor(s, dtype=torch.float32) for r in recs for s in r["raw_detections"]["pred_scores"]]
            coco_m = FrozenFPNDetectorTeacher.compute_dataset_level_coco_metrics(all_gt, all_pb, all_ps, all_ignore_boxes=all_ig)
            row_q: Dict[str, Any] = {
                "total_bpp": round(tot_bits / max(1, tot_pix), 5),
                "p_frame_bpp": round(p_bits / max(1, p_pix), 5) if p_pix > 0 else None,
                **coco_m,
            }
            if all("yolo_pred_boxes" in r["raw_detections"] for r in recs):
                all_yb = [torch.tensor(b, dtype=torch.float32).reshape(-1, 4) for r in recs for b in r["raw_detections"]["yolo_pred_boxes"]]
                all_ys = [torch.tensor(s, dtype=torch.float32) for r in recs for s in r["raw_detections"]["yolo_pred_scores"]]
                y_m = FrozenFPNDetectorTeacher.compute_dataset_level_coco_metrics(all_gt, all_yb, all_ys, all_ignore_boxes=all_ig)
                row_q["yolo_gt_ap50_pct"] = y_m["gt_ap50_pct"]
                row_q["yolo_gt_ap50_95_pct"] = y_m["gt_ap50_95_pct"]
            pqp[str(q)] = row_q
        sota_methods_out[m_name] = {"per_qp": pqp}

    anc_pqp = sota_methods_out["anchor_x265"]["per_qp"]
    ar = [anc_pqp[str(q)]["total_bpp"] for q in CRFS]
    ap = [anc_pqp[str(q)]["p_frame_bpp"] for q in CRFS]
    a50 = [anc_pqp[str(q)]["gt_ap50_pct"] for q in CRFS]
    a95 = [anc_pqp[str(q)]["gt_ap50_95_pct"] for q in CRFS]

    for m_name, entry in sota_methods_out.items():
        mpqp = entry["per_qp"]
        mr = [mpqp[str(q)]["total_bpp"] for q in CRFS]
        mp = [mpqp[str(q)]["p_frame_bpp"] for q in CRFS]
        m50 = [mpqp[str(q)]["gt_ap50_pct"] for q in CRFS]
        m95 = [mpqp[str(q)]["gt_ap50_95_pct"] for q in CRFS]
        entry["bdbr_tot_gt_ap50_pct"] = 0.0 if m_name == "anchor_x265" else _safe_bdbr(ar, a50, mr, m50)
        entry["bdbr_p_gt_ap50_pct"] = (
            0.0 if m_name == "anchor_x265" else (_safe_bdbr(ap, a50, mp, m50) if all(x is not None for x in ap + mp) else None)
        )
        entry["bdbr_tot_gt_ap50_95_pct"] = 0.0 if m_name == "anchor_x265" else _safe_bdbr(ar, a95, mr, m95)
        entry["bdbr_p_gt_ap50_95_pct"] = (
            0.0 if m_name == "anchor_x265" else (_safe_bdbr(ap, a95, mp, m95) if all(x is not None for x in ap + mp) else None)
        )
        if boot_curves["anchor_x265"] is not None and boot_curves[m_name] is not None and m_name != "anchor_x265":
            ci50 = _summarize_bootstrap_samples(
                _compute_bdbr_samples(
                    boot_curves["anchor_x265"]["total_bpp"],
                    boot_curves["anchor_x265"]["gt_ap50_pct"],
                    boot_curves[m_name]["total_bpp"],
                    boot_curves[m_name]["gt_ap50_pct"],
                )
            )
            entry["bdbr_tot_gt_ap50_ci95"] = ci50["ci95"] if ci50 else None

    sota_report = {
        "external_repositories_status": {
            "miccunifi_SCOPE": {
                "repo": "https://github.com/miccunifi/SCOPE",
                "executed": bool(scope_bundle.get("available")),
                "details": scope_bundle.get("ckpt_sha256") if scope_bundle.get("available") else scope_bundle.get("reason"),
            },
            "XingtongGe_PreprocessingICM": {
                "repo": "https://github.com/XingtongGe/PreprocessingICM",
                "executed": False,
                "reason": "Public repository provides NPP_Codec/models/Preprocessor.py source only with zero published pretrained weights; requires training from scratch on legacy mmcv==1.7.0 + mmdet==2.19.1 + bpgenc.",
            },
        },
        "methods": sota_methods_out,
    }

    # Summarize ByteTrack Tracking Metrics (MOTA, IDF1, HOTA, IDSW & Tracking BDBR)
    tracking_report: Dict[str, Any] = {}
    if run_tracking and seq_frame_dets:
        seqs_present = sorted({k[2] for k in seq_frame_dets})
        for m_name in active_methods:
            per_qp_trk: Dict[str, Any] = {}
            for q in CRFS:
                seq_summaries = []
                per_seq_q: Dict[str, Any] = {}
                for seq_name in seqs_present:
                    frames_data = sorted(seq_frame_dets.get((m_name, q, seq_name), []), key=lambda row: row[0])
                    if not frames_data:
                        continue
                    tracker = ByteTrackerKalman()
                    gb_seq, gid_seq, ig_seq, tb_seq, tid_seq = [], [], [], [], []
                    for _, gb_np, gid_np, ig_np, pb_np, ps_np in frames_data:
                        trk_boxes, trk_ids = tracker.update(pb_np, ps_np)
                        gb_seq.append(gb_np)
                        gid_seq.append(gid_np)
                        ig_seq.append(ig_np)
                        tb_seq.append(trk_boxes)
                        tid_seq.append(trk_ids)
                    seq_m = compute_mot_tracking_metrics(gb_seq, gid_seq, tb_seq, tid_seq, ignore_boxes_seq=ig_seq)
                    seq_m_slim = {k: v for k, v in seq_m.items() if k != "hota_per_alpha"}
                    per_seq_q[seq_name] = seq_m_slim
                    seq_summaries.append(seq_m)
                pooled = pool_mot_tracking_summaries(seq_summaries)
                pooled["total_bpp"] = sota_methods_out[m_name]["per_qp"][str(q)]["total_bpp"]
                pooled["p_frame_bpp"] = sota_methods_out[m_name]["per_qp"][str(q)]["p_frame_bpp"]
                pooled["per_sequence"] = per_seq_q
                per_qp_trk[str(q)] = pooled
            tracking_report[m_name] = {"per_qp": per_qp_trk}

        anc_trk = tracking_report["anchor_x265"]["per_qp"]
        ar_t = [anc_trk[str(q)]["total_bpp"] for q in CRFS]
        a_mota = [anc_trk[str(q)]["mota_pct"] for q in CRFS]
        a_idf1 = [anc_trk[str(q)]["idf1_pct"] for q in CRFS]
        a_hota = [anc_trk[str(q)]["hota_pct"] for q in CRFS]

        for m_name, t_entry in tracking_report.items():
            mpqp = t_entry["per_qp"]
            mr_t = [mpqp[str(q)]["total_bpp"] for q in CRFS]
            m_mota = [mpqp[str(q)]["mota_pct"] for q in CRFS]
            m_idf1 = [mpqp[str(q)]["idf1_pct"] for q in CRFS]
            m_hota = [mpqp[str(q)]["hota_pct"] for q in CRFS]
            t_entry["bdbr_tot_mota_pct"] = (
                0.0 if m_name == "anchor_x265" else (_safe_bdbr(ar_t, a_mota, mr_t, m_mota) if all(x is not None for x in a_mota + m_mota) else None)
            )
            t_entry["bdbr_tot_idf1_pct"] = (
                0.0 if m_name == "anchor_x265" else (_safe_bdbr(ar_t, a_idf1, mr_t, m_idf1) if all(x is not None for x in a_idf1 + m_idf1) else None)
            )
            t_entry["bdbr_tot_hota_pct"] = (
                0.0 if m_name == "anchor_x265" else (_safe_bdbr(ar_t, a_hota, mr_t, m_hota) if all(x is not None for x in a_hota + m_hota) else None)
            )

    return sota_report, tracking_report


def _infer_secondary_dataset_tag(root: Path) -> str:
    """Infer canonical secondary dataset key ('mot20', 'kitti', 'sfu') from an explicit single-dataset path."""
    lower = str(root).replace("\\", "/").lower()
    if "mot20" in lower or "mot-20" in lower:
        return "mot20"
    if "kitti" in lower:
        return "kitti"
    if "hevc-b" in lower or "sfu" in lower:
        return "sfu"
    if (root / "label_02").is_dir() and (root / "image_02").is_dir():
        return "kitti"
    if root.is_dir() and any(root.glob("*.yuv")):
        return "sfu"
    for child in sorted(root.iterdir()) if root.is_dir() else ():
        cname = child.name.upper()
        if cname.startswith("MOT20-") and (child / "img1").is_dir():
            return "mot20"
    if root.is_dir():
        has_mot20 = any(
            gt_file.parent.parent.name.upper().startswith("MOT20-")
            for gt_file in sorted(root.rglob("gt.txt"))
        )
        has_kitti = any(
            lbl_dir.is_dir() and (lbl_dir.parent / "image_02").is_dir()
            for lbl_dir in sorted(root.rglob("label_02"))
        )
        has_sfu = any(root.rglob("*.yuv"))
        if int(has_mot20) + int(has_kitti) + int(has_sfu) == 1:
            if has_mot20:
                return "mot20"
            if has_kitti:
                return "kitti"
            if has_sfu:
                return "sfu"
    return "custom_second_dataset"


def _discover_secondary_dataset_candidates(second_data_root: str) -> List[Tuple[str, Path]]:
    """
    Resolves canonical secondary dataset `(tag, path)` pairs:
      1. If an explicit `--second_data_root` is provided, maps it to its canonical tag
         (`mot20` / `kitti` / `sfu` / multi-dataset parent, falling back to `custom_second_dataset`).
      2. Always appends any additional mounted secondary datasets (`mot20`, `kitti`, `sfu`)
         whose resolved path is not already queued so co-mounted MOT20 + KITTI both run!
    """
    candidates: List[Tuple[str, Path]] = []
    seen_resolved = set()
    seen_tags = set()

    def _add(tag: str, p: Path) -> None:
        rp = p.resolve()
        if rp in seen_resolved or tag in seen_tags:
            return
        seen_resolved.add(rp)
        seen_tags.add(tag)
        candidates.append((tag, p))

    sec_arg = (second_data_root or "").strip()
    if sec_arg.lower() not in ("", "auto", "none") and Path(sec_arg).exists():
        sec_path = Path(sec_arg)
        inferred = _infer_secondary_dataset_tag(sec_path)
        if inferred != "custom_second_dataset":
            _add(inferred, sec_path)
        else:
            sub_found: Dict[str, Path] = {}
            try:
                from scripts.prepare_kaggle_datasets import _scan_kitti_root, _scan_mot_train_roots, _scan_yuv_root

                mot_sub = _scan_mot_train_roots(sec_path)
                if "mot20" in mot_sub:
                    sub_found["mot20"] = mot_sub["mot20"]
                kitti_sub = _scan_kitti_root(sec_path)
                if kitti_sub is not None:
                    sub_found["kitti"] = kitti_sub
                sfu_sub = _scan_yuv_root(sec_path)
                if sfu_sub is not None:
                    sub_found["sfu"] = sfu_sub
            except Exception:
                sub_found = {}
            added_sub = False
            for sub_tag in ("mot20", "kitti", "sfu"):
                if sub_tag in sub_found and sub_found[sub_tag].exists():
                    _add(sub_tag, sub_found[sub_tag])
                    added_sub = True
            if not added_sub:
                _add("custom_second_dataset", sec_path)

    for ds_tag in ("mot20", "kitti", "sfu"):
        found = _safe_find_mounted_dataset(ds_tag)
        if found is not None and found.exists():
            _add(ds_tag, found)

    return candidates


@torch.no_grad()
def evaluate_secondary_vcm_datasets(
    args: argparse.Namespace,
    models_dict: Dict[str, TemporalSCOPE],
    codec: HybridX265STE,
    fpn_teacher: FrozenFPNDetectorTeacher,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluates T-SCOPE v3 zero-shot on secondary VCM datasets (`MOT20` crowded tracking & `KITTI` multi-class driving).
    """
    candidates = _discover_secondary_dataset_candidates(args.second_data_root)

    if not candidates:
        return {"status": "skipped", "reason": "No secondary dataset mounted (mount mot-20 or kitti-tracking via --extra_dataset_slugs)"}

    results: Dict[str, Any] = {}
    img_hw = (args.img_h, args.img_w)
    fam0 = torch.tensor([0], device=device, dtype=torch.long)

    for ds_tag, ds_root in candidates:
        sig = inspect.signature(RealVCMVideoDataset.__init__)
        ds_kwargs: Dict[str, Any] = {
            "data_root": str(ds_root),
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "clip_len": args.gop_len,
            "stride": args.gop_len,
            "img_size": img_hw if (img_hw[0] != img_hw[1] or "img_h" in sig.parameters) else img_hw[0],
            "is_train": False,
            "enforce_disjoint_split": False,
            "split": "test",
        }
        if "img_h" in sig.parameters and "img_w" in sig.parameters:
            ds_kwargs["img_h"], ds_kwargs["img_w"] = img_hw
        if "dataset_filter" in sig.parameters and ds_tag in ("mot20", "kitti", "sfu"):
            ds_kwargs["dataset_filter"] = ds_tag
        if "label_root" in sig.parameters and args.sfu_label_root:
            ds_kwargs["label_root"] = args.sfu_label_root

        sec_ds = RealVCMVideoDataset(**ds_kwargs)
        if len(sec_ds) == 0:
            results[ds_tag] = {"status": "empty", "data_root": str(ds_root)}
            continue

        # Cap evaluation to a representative deterministic slice (up to 30 complete 32-frame GOP clips = 960 frames)
        # so multi-dataset evaluation completes cleanly inside a single Kaggle T4 session
        max_clips = min(len(sec_ds), args.max_second_dataset_clips)
        methods = ["anchor_x265", "tscope_spatial_only", "tscope_iwa_gop_qp_reuse", "tscope_full"]
        buckets: Dict[Tuple[str, int], Dict[str, Any]] = {
            (m, q): {
                "tot_bits": 0,
                "tot_pix": 0,
                "p_bits": 0,
                "p_pix": 0,
                "gt_boxes": [],
                "gt_classes": [],
                "ig_boxes": [],
                "pred_boxes": [],
                "pred_labels": [],
                "pred_scores": [],
            }
            for m in methods
            for q in CRFS
        }

        for idx in range(max_clips):
            sample = sec_ds[idx]
            video = sample["video"].unsqueeze(0).to(device)
            gt_b_all = sample["gt_boxes"]  # [T, 256, 4]
            ig_b_all = sample["ignore_boxes"]
            gt_cls_all = sample.get("gt_classes", sample.get("gt_labels"))
            b, t, c, h, w = video.shape

            for qp in CRFS:
                sev = torch.tensor([(qp - 22.0) / 25.0], device=device, dtype=torch.float32)
                c_anc = _call_evaluate_real_codec(codec, video, qp, None, family_id=0, gop_len=args.gop_len, bframes=0)
                p_sp = models_dict["spatial_only_scope"](video, fam0, sev, reuse_iframe_qp=False, disable_temporal=True)
                c_sp = _call_evaluate_real_codec(codec, p_sp["i_edit"], qp, None, family_id=0, gop_len=args.gop_len, bframes=0)
                p_iwa = models_dict["iwa_gop_qp_reuse"](video, fam0, sev, reuse_iframe_qp=True, disable_temporal=False)
                c_iwa = _call_evaluate_real_codec(codec, p_iwa["i_edit"], qp, p_iwa["ctu_qp_offsets"], family_id=0, gop_len=args.gop_len, bframes=0)
                p_full = models_dict["full_tscope"](video, fam0, sev, reuse_iframe_qp=False, disable_temporal=False)
                c_full = _call_evaluate_real_codec(codec, p_full["i_edit"], qp, p_full["ctu_qp_offsets"], family_id=0, gop_len=args.gop_len, bframes=0)

                for m_name, c_out in [
                    ("anchor_x265", c_anc),
                    ("tscope_spatial_only", c_sp),
                    ("tscope_iwa_gop_qp_reuse", c_iwa),
                    ("tscope_full", c_full),
                ]:
                    bkt = buckets[(m_name, qp)]
                    area = h * w
                    for sz, ft in zip(c_out["packet_sizes"], c_out["frame_types"]):
                        bkt["tot_bits"] += int(sz) * 8
                        bkt["tot_pix"] += area
                        if ft != "I":
                            bkt["p_bits"] += int(sz) * 8
                            bkt["p_pix"] += area

                    recon_flat = c_out["video_recon"].view(b * t, c, h, w)
                    dets = []
                    for st in range(0, recon_flat.shape[0], 8):
                        dets.extend(fpn_teacher.detector(list(torch.clamp(recon_flat[st : st + 8], 0.0, 1.0))))
                    for f_i, d_f in enumerate(dets):
                        v_m = gt_b_all[f_i][:, 0] >= 0.0
                        bkt["gt_boxes"].append(gt_b_all[f_i][v_m].detach().cpu())
                        if gt_cls_all is not None:
                            bkt["gt_classes"].append(gt_cls_all[f_i][v_m].detach().cpu())
                        else:
                            bkt["gt_classes"].append(torch.ones(( int(v_m.sum().item()),), dtype=torch.long))
                        ig_m = ig_b_all[f_i][:, 0] >= 0.0
                        bkt["ig_boxes"].append(ig_b_all[f_i][ig_m].detach().cpu())

                        sc_m = d_f["scores"] >= 0.01
                        bkt["pred_boxes"].append(d_f["boxes"][sc_m].detach().cpu())
                        bkt["pred_labels"].append(d_f["labels"][sc_m].detach().cpu())
                        bkt["pred_scores"].append(d_f["scores"][sc_m].detach().cpu())

        ds_summary: Dict[str, Any] = {"clips_evaluated": max_clips, "frames_evaluated": max_clips * args.gop_len, "methods": {}}
        for m_name in methods:
            pqp = {}
            for q in CRFS:
                bkt = buckets[(m_name, q)]
                mc = compute_multiclass_dataset_coco_metrics(
                    bkt["gt_boxes"],
                    bkt["gt_classes"],
                    bkt["pred_boxes"],
                    bkt["pred_labels"],
                    bkt["pred_scores"],
                    bkt["ig_boxes"],
                )
                pqp[str(q)] = {
                    "total_bpp": round(bkt["tot_bits"] / max(1, bkt["tot_pix"]), 5),
                    "p_frame_bpp": round(bkt["p_bits"] / max(1, bkt["p_pix"]), 5) if bkt["p_pix"] > 0 else None,
                    "map50_pct": mc["map50_pct"],
                    "map50_95_pct": mc["map50_95_pct"],
                    "active_classes": mc["active_classes"],
                }
            ds_summary["methods"][m_name] = {"per_qp": pqp}

        anc_p = ds_summary["methods"]["anchor_x265"]["per_qp"]
        ar = [anc_p[str(q)]["total_bpp"] for q in CRFS]
        ap = [anc_p[str(q)]["p_frame_bpp"] for q in CRFS]
        a50 = [anc_p[str(q)]["map50_pct"] for q in CRFS]
        a95 = [anc_p[str(q)]["map50_95_pct"] for q in CRFS]
        for m_name, m_ent in ds_summary["methods"].items():
            mp_d = m_ent["per_qp"]
            mr = [mp_d[str(q)]["total_bpp"] for q in CRFS]
            mp = [mp_d[str(q)]["p_frame_bpp"] for q in CRFS]
            m50 = [mp_d[str(q)]["map50_pct"] for q in CRFS]
            m95 = [mp_d[str(q)]["map50_95_pct"] for q in CRFS]
            m_ent["bdbr_tot_map50_pct"] = 0.0 if m_name == "anchor_x265" else _safe_bdbr(ar, a50, mr, m50)
            m_ent["bdbr_p_map50_pct"] = (
                0.0 if m_name == "anchor_x265" else (_safe_bdbr(ap, a50, mp, m50) if all(x is not None for x in ap + mp) else None)
            )
            m_ent["bdbr_tot_map50_95_pct"] = 0.0 if m_name == "anchor_x265" else _safe_bdbr(ar, a95, mr, m95)
        results[ds_tag] = ds_summary
        print(f"[2nd Dataset: {ds_tag}] tscope_full mAP@50 BDBR = {ds_summary['methods']['tscope_full']['bdbr_tot_map50_pct']}", flush=True)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="T-SCOPE v3 External SOTA, Tracking, 2nd Dataset & Complexity Suite")
    parser.add_argument("--data_root", type=str, default="", help="MOT17 root (auto-detects Kaggle mount if omitted)")
    parser.add_argument("--second_data_root", type=str, default="", help="Optional MOT20 / KITTI / SFU root")
    parser.add_argument("--sfu_label_root", type=str, default="", help="Optional SFU YOLO label directory")
    parser.add_argument("--soup_dir", type=str, default="./release_ckpts")
    parser.add_argument("--scope_ckpt", type=str, default="", help="Optional path to official miccunifi/SCOPE best_val.pt")
    parser.add_argument("--scope_repo_dir", type=str, default="/tmp/miccunifi_SCOPE")
    parser.add_argument("--auto_download_scope", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--img_h", type=int, default=IMG_H)
    parser.add_argument("--img_w", type=int, default=IMG_W)
    parser.add_argument("--gop_len", type=int, default=EVAL_GOP_LEN)
    parser.add_argument("--bframes", type=int, default=0)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--finetune_det_epochs", type=int, default=2)
    parser.add_argument("--max_second_dataset_clips", type=int, default=30)
    parser.add_argument("--bootstrap_resamples", type=int, default=10000)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["sota", "tracking", "second_dataset", "complexity", "all"],
        default=["all"],
    )
    args = parser.parse_args()

    selected_tasks = set(args.tasks)
    if "all" in selected_tasks:
        selected_tasks = {"sota", "tracking", "second_dataset", "complexity"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = (args.data_root or "").strip()
    if data_root.lower() in ("", "auto", "none"):
        mounted = _safe_find_mounted_dataset("mot17")
        data_root = str(mounted) if mounted is not None else "./data/MOT17_sample"

    soup_dir = Path(args.soup_dir)
    models_dict: Dict[str, TemporalSCOPE] = {}
    soup_hashes: Dict[str, Optional[str]] = {}
    for variant, max_dqp in (
        ("full_tscope", 4.0),
        ("iwa_gop_qp_reuse", 4.0),
        ("spatial_only_scope", 0.0),
    ):
        m_v, h_v = _load_soup_or_fresh(soup_dir, variant, max_dqp, device)
        models_dict[variant] = m_v
        soup_hashes[variant] = h_v

    codec = HybridX265STE(use_real_x265_in_forward=True, preset="medium").to(device)
    scope_bundle = (
        try_setup_official_scope(args.scope_ckpt, args.scope_repo_dir, args.auto_download_scope, device)
        if ("sota" in selected_tasks or "complexity" in selected_tasks)
        else {"available": False, "reason": "Not requested"}
    )

    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "img_h": args.img_h,
            "img_w": args.img_w,
            "gop_len": args.gop_len,
            "bframes": args.bframes,
            "crfs": list(CRFS),
            "soup_sha256": soup_hashes,
            "tasks_executed": sorted(selected_tasks),
            "bdbr_interpolator": "scipy.interpolate.PchipInterpolator",
        },
    }

    fpn_teacher = None
    if selected_tasks.intersection({"sota", "tracking", "second_dataset"}):
        fpn_teacher = FrozenFPNDetectorTeacher(pretrained=True, use_yolov8s=True).to(device)
        if not fpn_teacher.use_torchvision or fpn_teacher.detector is None:
            raise RuntimeError("Pretrained Faster R-CNN ResNet50-FPN v2 is required for SOTA/tracking evaluation")

    if "sota" in selected_tasks or "tracking" in selected_tasks:
        assert fpn_teacher is not None
        sota_res, trk_res = run_sota_and_tracking_on_mot17(
            args=args,
            data_root=data_root,
            models_dict=models_dict,
            scope_bundle=scope_bundle,
            codec=codec,
            fpn_teacher=fpn_teacher,
            device=device,
            run_sota=("sota" in selected_tasks),
            run_tracking=("tracking" in selected_tasks),
        )
        if "sota" in selected_tasks:
            report["external_sota_baselines"] = sota_res
        if "tracking" in selected_tasks:
            report["mot17_tracking_metrics"] = trk_res

    if "second_dataset" in selected_tasks:
        assert fpn_teacher is not None
        report["second_dataset_vcm_summary"] = evaluate_secondary_vcm_datasets(
            args=args,
            models_dict=models_dict,
            codec=codec,
            fpn_teacher=fpn_teacher,
            device=device,
        )

    if "complexity" in selected_tasks:
        report["complexity_runtime_table"] = benchmark_complexity_and_runtime(
            models_dict=models_dict,
            scope_bundle=scope_bundle,
            codec=codec,
            device=device,
            img_hw=(args.img_h, args.img_w),
            gop_len=args.gop_len,
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = (
        f"sota_and_tracking_report_shard{args.shard_id:02d}of{args.num_shards:02d}.json"
        if args.num_shards > 1
        else "sota_and_tracking_report.json"
    )
    out_path = out_dir / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(f"\nSaved complete SOTA, Tracking, 2nd-Dataset & Complexity report -> {out_path}")


if __name__ == "__main__":
    main()
