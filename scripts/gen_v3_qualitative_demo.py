#!/usr/bin/env python3
"""
Generate Q1 Qualitative Figure (fig_qualitative_v3.pdf/.png) and Side-by-Side
Demo Videos (demo_*_crf37.mp4) from verified T-SCOPE v3 soup checkpoints.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.frozen_detector_fpn import FrozenFPNDetectorTeacher
from scripts.eval_cross_backbone_bdbr import _load_v3_checkpoint_into_model
from scripts.experiment_protocol import IMG_H, IMG_W
from scripts.kaggle_shard_runner import (
    _call_evaluate_real_codec,
    _construct_real_dataset,
    _safe_find_mounted_dataset,
)
from video_codecs.hybrid_x265_ste import HybridX265STE


def _to_uint8_hwc(img_chw: torch.Tensor) -> np.ndarray:
    arr = (img_chw.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round()
    return np.clip(arr, 0, 255).astype(np.uint8)


def _draw_boxes_pil(
    arr_hwc: np.ndarray,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    score_thr: float = 0.50,
    color: Tuple[int, int, int] = (0, 230, 118),
) -> Image.Image:
    im = Image.fromarray(arr_hwc)
    draw = ImageDraw.Draw(im)
    b_np = boxes.detach().cpu().numpy()
    s_np = scores.detach().cpu().numpy()
    for box, sc in zip(b_np, s_np):
        if float(sc) < score_thr:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    return im


def _colorize_heatmap(map_hw: np.ndarray, cmap_name: str = "inferno") -> np.ndarray:
    import matplotlib

    cmap = matplotlib.colormaps[cmap_name]
    rgba = cmap(np.clip(map_hw, 0.0, 1.0))
    return (rgba[:, :, :3] * 255.0).round().astype(np.uint8)


def _overlay_dqp_grid(img_hwc: np.ndarray, dqp_qg: np.ndarray, max_dqp: float = 7.14) -> np.ndarray:
    h, w, _ = img_hwc.shape
    dqp_t = torch.from_numpy(dqp_qg).float().unsqueeze(0).unsqueeze(0)
    up = F.interpolate(dqp_t, size=(h, w), mode="nearest").squeeze().numpy()
    norm = np.clip(up / max(max_dqp, 1e-6), 0.0, 1.0)
    heat = _colorize_heatmap(norm, "YlOrRd").astype(np.float32)
    base = img_hwc.astype(np.float32)
    alpha = 0.45 * (norm > 0.02)[..., None]
    blend = base * (1.0 - alpha) + heat * alpha
    return np.clip(blend.round(), 0, 255).astype(np.uint8)


def _write_mp4_from_frames(frames_hwc: List[np.ndarray], out_mp4: Path, fps: int = 10) -> None:
    if not frames_hwc:
        return
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        for i, arr in enumerate(frames_hwc):
            Image.fromarray(arr).save(td_p / f"f_{i:04d}.png")
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(td_p / "f_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(out_mp4),
        ]
        subprocess.run(cmd, check=True)


def _pick_clip_indices(dataset: Any, targets: Tuple[str, ...]) -> List[Tuple[str, int]]:
    picked: List[Tuple[str, int]] = []
    seen = set()
    for t_prefix in targets:
        for idx, clip in enumerate(dataset.clips):
            sname = str(clip.get("seq_name", ""))
            if sname.startswith(t_prefix) and t_prefix not in seen:
                picked.append((sname, idx))
                seen.add(t_prefix)
                break
    if not picked and len(dataset) > 0:
        picked.append((str(dataset.clips[0].get("seq_name", "sample")), 0))
    return picked


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate T-SCOPE v3 qualitative figure & side-by-side MP4 demos")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--soup_dir", type=str, default="./release_ckpts")
    parser.add_argument("--output_dir", type=str, default="./outputs_demo")
    parser.add_argument("--crf", type=int, default=37)
    parser.add_argument("--gop_len", type=int, default=32)
    parser.add_argument("--frame_idx", type=int, default=16)
    parser.add_argument("--score_thr", type=float, default=0.50)
    parser.add_argument("--smoke_test", action="store_true", help="Allow train/sample split for local smoke checks")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = (args.data_root or "").strip()
    if data_root.lower() in ("", "auto", "none"):
        mounted = _safe_find_mounted_dataset("mot17")
        data_root = str(mounted) if mounted is not None else str(ROOT_DIR / "data" / "MOT17_sample")

    soup_dir = Path(args.soup_dir)
    full_ckpt = soup_dir / "tscope_soup_full_tscope.pt"
    iwa_ckpt = soup_dir / "tscope_soup_iwa_gop_qp_reuse.pt"
    model_full, _ = _load_v3_checkpoint_into_model(full_ckpt, max_delta_qp=4.0, device=device)
    model_iwa, _ = _load_v3_checkpoint_into_model(iwa_ckpt, max_delta_qp=4.0, device=device)

    codec = HybridX265STE(use_real_x265_in_forward=True, preset="medium").to(device)
    teacher = FrozenFPNDetectorTeacher(pretrained=True, use_yolov8s=False).to(device).eval()

    split = "train" if args.smoke_test else "test"
    dataset = _construct_real_dataset(
        data_root=data_root,
        shard_id=0,
        num_shards=1,
        clip_len=args.gop_len,
        stride=args.gop_len,
        img_hw=(IMG_H, IMG_W),
        is_train=args.smoke_test,
        enforce_disjoint_split=not args.smoke_test,
        split=split,
    )
    targets = ("MOT17-11", "MOT17-13") if not args.smoke_test else ("MOT17-01", "MOT17-02")
    selected = _pick_clip_indices(dataset, targets)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fam0 = torch.tensor([0], device=device, dtype=torch.long)
    sev = torch.tensor([(args.crf - 22.0) / 25.0], device=device, dtype=torch.float32)

    panel_rows: List[Dict[str, Any]] = []
    summary_stats: Dict[str, Any] = {}

    for seq_name, ds_idx in selected:
        sample = dataset[ds_idx]
        video = sample["video"].unsqueeze(0).to(device)
        t_len = video.shape[1]
        t_show = min(args.frame_idx, t_len - 1)

        pred_full = model_full(video, fam0, sev, reuse_iframe_qp=False, disable_temporal=False)
        pred_iwa = model_iwa(video, fam0, sev, reuse_iframe_qp=True, disable_temporal=False)

        anc_out = _call_evaluate_real_codec(codec, video, args.crf, None, family_id=0, gop_len=args.gop_len, bframes=0)
        full_out = _call_evaluate_real_codec(
            codec, pred_full["i_edit"], args.crf, pred_full["ctu_qp_offsets"], family_id=0, gop_len=args.gop_len, bframes=0
        )
        iwa_out = _call_evaluate_real_codec(
            codec, pred_iwa["i_edit"], args.crf, pred_iwa["ctu_qp_offsets"], family_id=0, gop_len=args.gop_len, bframes=0
        )

        # Run Faster R-CNN detections on decoded anchor & T-SCOPE-Full frames
        def _det_clip(recon_btchw: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
            flat = torch.clamp(recon_btchw[0], 0.0, 1.0)
            dets = []
            for st in range(0, flat.shape[0], 4):
                dets.extend({k: v.detach().cpu() for k, v in d.items()} for d in teacher.detector(list(flat[st : st + 4])))
            return dets

        anc_dets = _det_clip(anc_out["video_recon"])
        full_dets = _det_clip(full_out["video_recon"])

        # Collect qualitative row at t_show
        gt_cnt = int((sample["gt_boxes"][t_show][:, 0] >= 0.0).sum().item())
        anc_cnt = int(((anc_dets[t_show]["labels"] == 1) & (anc_dets[t_show]["scores"] >= args.score_thr)).sum().item())
        full_cnt = int(((full_dets[t_show]["labels"] == 1) & (full_dets[t_show]["scores"] >= args.score_thr)).sum().item())

        orig_arr = _to_uint8_hwc(video[0, t_show])
        edit_arr = _to_uint8_hwc(pred_full["i_edit"][0, t_show])
        attn_arr = _colorize_heatmap(pred_full["eff_attn_map"][0, t_show, 0].detach().cpu().numpy(), "inferno")
        dqp_arr = _overlay_dqp_grid(orig_arr, pred_full["ctu_qp_offsets"][0, t_show].detach().cpu().numpy())

        anc_mask = anc_dets[t_show]["labels"] == 1
        anc_det_im = np.asarray(
            _draw_boxes_pil(
                _to_uint8_hwc(anc_out["video_recon"][0, t_show]),
                anc_dets[t_show]["boxes"][anc_mask],
                anc_dets[t_show]["scores"][anc_mask],
                score_thr=args.score_thr,
                color=(255, 165, 0),
            )
        )
        full_mask = full_dets[t_show]["labels"] == 1
        full_det_im = np.asarray(
            _draw_boxes_pil(
                _to_uint8_hwc(full_out["video_recon"][0, t_show]),
                full_dets[t_show]["boxes"][full_mask],
                full_dets[t_show]["scores"][full_mask],
                score_thr=args.score_thr,
                color=(0, 230, 118),
            )
        )

        panel_rows.append(
            {
                "seq": seq_name,
                "t": t_show,
                "gt_cnt": gt_cnt,
                "anc_cnt": anc_cnt,
                "full_cnt": full_cnt,
                "anc_bpp": float(anc_out["total_bpp"]),
                "full_bpp": float(full_out["total_bpp"]),
                "iwa_bpp": float(iwa_out["total_bpp"]),
                "imgs": [orig_arr, edit_arr, attn_arr, dqp_arr, anc_det_im, full_det_im],
            }
        )

        # Build 2x2 side-by-side MP4 demo
        mp4_frames: List[np.ndarray] = []
        for t_i in range(t_len):
            tl = Image.fromarray(_to_uint8_hwc(video[0, t_i]))
            d_tl = ImageDraw.Draw(tl)
            d_tl.rectangle([0, 0, 210, 22], fill=(0, 0, 0))
            d_tl.text((6, 5), f"{seq_name} | Input t={t_i:02d}", fill=(255, 255, 255))

            am = anc_dets[t_i]["labels"] == 1
            tr = _draw_boxes_pil(
                _to_uint8_hwc(anc_out["video_recon"][0, t_i]),
                anc_dets[t_i]["boxes"][am],
                anc_dets[t_i]["scores"][am],
                score_thr=args.score_thr,
                color=(255, 165, 0),
            )
            ac = int((anc_dets[t_i]["scores"][am] >= args.score_thr).sum().item())
            d_tr = ImageDraw.Draw(tr)
            d_tr.rectangle([0, 0, 315, 22], fill=(0, 0, 0))
            d_tr.text((6, 5), f"Anchor x265 CRF{args.crf} | {anc_out['total_bpp']:.4f} bpp | det={ac}", fill=(255, 200, 100))

            bl_arr = _overlay_dqp_grid(
                _to_uint8_hwc(pred_full["i_edit"][0, t_i]),
                pred_full["ctu_qp_offsets"][0, t_i].detach().cpu().numpy(),
            )
            bl = Image.fromarray(bl_arr)
            d_bl = ImageDraw.Draw(bl)
            d_bl.rectangle([0, 0, 290, 22], fill=(0, 0, 0))
            d_bl.text((6, 5), "T-SCOPE Pre-Edit + 16x16 QG dQP", fill=(150, 230, 255))

            fm = full_dets[t_i]["labels"] == 1
            br = _draw_boxes_pil(
                _to_uint8_hwc(full_out["video_recon"][0, t_i]),
                full_dets[t_i]["boxes"][fm],
                full_dets[t_i]["scores"][fm],
                score_thr=args.score_thr,
                color=(0, 230, 118),
            )
            fc = int((full_dets[t_i]["scores"][fm] >= args.score_thr).sum().item())
            d_br = ImageDraw.Draw(br)
            d_br.rectangle([0, 0, 330, 22], fill=(0, 0, 0))
            d_br.text((6, 5), f"T-SCOPE Full CRF{args.crf} | {full_out['total_bpp']:.4f} bpp | det={fc}", fill=(120, 255, 160))

            top = np.concatenate([np.asarray(tl), np.asarray(tr)], axis=1)
            bot = np.concatenate([np.asarray(bl), np.asarray(br)], axis=1)
            mp4_frames.append(np.concatenate([top, bot], axis=0))

        safe_tag = seq_name.lower().replace("-", "_")
        mp4_path = out_dir / f"demo_{safe_tag}_crf{args.crf}.mp4"
        _write_mp4_from_frames(mp4_frames, mp4_path, fps=10)
        summary_stats[seq_name] = {
            "clip_frame_ids": sample["frame_ids"],
            "anchor_total_bpp": round(float(anc_out["total_bpp"]), 5),
            "tscope_full_total_bpp": round(float(full_out["total_bpp"]), 5),
            "tscope_iwa_total_bpp": round(float(iwa_out["total_bpp"]), 5),
            "mp4_file": mp4_path.name,
        }

    # Render clean Q1 qualitative grid PDF + PNG
    col_titles = [
        "Clean Input",
        "Pre-Edited $I_{\\mathrm{edit},t}$",
        "Attention $A_t^{\\mathrm{eff}}$",
        "16×16 QG $\\Delta\\mathrm{QP}$",
        "Anchor x265 + Det",
        "T-SCOPE Full + Det",
    ]
    n_rows = len(panel_rows)
    fig, axes = plt.subplots(n_rows, 6, figsize=(16.5, 3.0 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    for r_i, row in enumerate(panel_rows):
        for c_i in range(6):
            ax = axes[r_i, c_i]
            ax.imshow(row["imgs"][c_i])
            ax.set_xticks([])
            ax.set_yticks([])
            if r_i == 0:
                ax.set_title(col_titles[c_i], fontsize=10.5, fontweight="bold", pad=6)
            if c_i == 0:
                ax.set_ylabel(f"{row['seq']}\n(t={row['t']}, GT={row['gt_cnt']})", fontsize=9.5, fontweight="bold")
            if c_i == 4:
                ax.set_xlabel(f"{row['anc_bpp']:.4f} bpp | det={row['anc_cnt']}", fontsize=9)
            elif c_i == 5:
                save_pct = (1.0 - row["full_bpp"] / max(row["anc_bpp"], 1e-9)) * 100.0
                ax.set_xlabel(f"{row['full_bpp']:.4f} bpp (-{save_pct:.1f}%) | det={row['full_cnt']}", fontsize=9, fontweight="bold")

    fig.savefig(out_dir / "fig_qualitative_v3.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_qualitative_v3.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    (out_dir / "qualitative_demo_summary.json").write_text(json.dumps(summary_stats, indent=2), encoding="utf-8")
    print(f"Saved qualitative figure + MP4 demos -> {out_dir}")


if __name__ == "__main__":
    main()
