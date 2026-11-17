"""Measure prepared v3 soups (and constituent shards + Greedy Soup) on MOT17-09 under a uniform real-libx265 criterion."""
import argparse
import copy
import datetime
import inspect
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from losses.vcm_temporal_loss import VCMTemporalPreEditLoss
from models.frozen_detector_fpn import FrozenFPNDetectorTeacher
from models.temporal_scope import TemporalSCOPE
from scripts.experiment_protocol import (
    IMG_H,
    IMG_W,
    TRAIN_CLIP_LEN,
    VALIDATION_PROTOCOL,
    VARIANTS,
    read_manifest_files,
    sha256_file,
    validate_release_validation,
    validate_soup_provenance,
)
from scripts.kaggle_shard_runner import ABLATION_PRESETS, _call_codec_ste, _call_criterion, _safe_find_mounted_dataset
from scripts.merge_shards import average_compatible_states
from vcm_datasets.vcm_video_datasets import RealVCMVideoDataset
from video_codecs.hybrid_x265_ste import HybridX265STE


@torch.no_grad()
def extract_validation_features(teacher: Any, images: torch.Tensor, chunk_size: int = 1) -> Dict[str, torch.Tensor]:
    if type(chunk_size) is not int or chunk_size < 1 or len(images) == 0:
        raise ValueError("Validation feature chunks require a positive size and nonempty frames")
    chunks = [teacher.extract_fpn_features(chunk) for chunk in images.split(chunk_size)]
    keys = tuple(chunks[0])
    if any(tuple(chunk) != keys for chunk in chunks):
        raise ValueError("Validation feature levels differ between chunks")
    return {key: torch.cat([chunk[key] for chunk in chunks], dim=0) for key in keys}


def _build_val_dataset(data_root: str, img_h: int, img_w: int) -> RealVCMVideoDataset:
    sig = inspect.signature(RealVCMVideoDataset.__init__)
    size_arg: Any = (img_h, img_w) if (img_h != img_w or "img_h" in sig.parameters) else img_h
    kwargs: Dict[str, Any] = {
        "data_root": data_root,
        "shard_id": 0,
        "num_shards": 1,
        "clip_len": TRAIN_CLIP_LEN,
        "stride": TRAIN_CLIP_LEN,
        "img_size": size_arg,
        "is_train": False,
        "enforce_disjoint_split": True,
        "split": "val",
    }
    if "img_h" in sig.parameters and "img_w" in sig.parameters:
        kwargs["img_h"] = img_h
        kwargs["img_w"] = img_w
    return RealVCMVideoDataset(**kwargs)


@torch.no_grad()
def _score_state_dict_on_mot17_09(
    state_dict: Dict[str, torch.Tensor],
    variant: str,
    preset: Dict[str, Any],
    loader: DataLoader,
    teacher: Any,
    cached_teacher_batches: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]],
    uniform_codec: Any,
    uniform_criterion: Any,
    native_codec: Optional[Any],
    native_criterion: Optional[Any],
    device: torch.device,
    chunk_size: int,
) -> Dict[str, Any]:
    """Evaluate a model state_dict under the uniform real-libx265 criterion (+ optional native preset loss)."""
    model = TemporalSCOPE(
        base_ch=48,
        num_families=2,
        cond_dim=64,
        qg_size=16,
        max_delta_qp=preset["max_delta_qp"],
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    uniform_total = 0.0
    native_total = 0.0
    count = 0
    batch_sizes: List[int] = []

    for batch, teacher_features in cached_teacher_batches:
        video = batch["video"].to(device)
        attn = batch["m_attn"].to(device)
        mask = batch["m_det"].to(device)
        b, t, _, h, w = video.shape
        batch_sizes.append(b)
        fam = torch.zeros(b, dtype=torch.long, device=device)

        for severity_value in VALIDATION_PROTOCOL["severities"]:
            severity = torch.full((b,), severity_value, device=device, dtype=torch.float32)
            output = model(
                video,
                fam,
                severity,
                reuse_iframe_qp=preset["reuse_iframe_qp"],
                disable_temporal=preset["disable_temporal"],
            )
            qp_arg = output["ctu_qp_offsets"] if preset["max_delta_qp"] > 0 else None
            decoded = _call_codec_ste(uniform_codec, output["i_edit"], severity, qp_arg, family_id=fam)
            features = extract_validation_features(
                teacher, decoded["video_recon"].reshape(b * t, 3, h, w), chunk_size
            )
            u_loss = _call_criterion(
                uniform_criterion,
                i_edit=output["i_edit"],
                i_recon=decoded["video_recon"],
                attn_map=output["attn_map"],
                eff_attn_map=output.get("eff_attn_map"),
                delta_obj=output["delta_obj"],
                delta_bg=output["delta_bg"],
                rate_proxy_bpp=decoded["rate_proxy_bpp"],
                m_attn=attn,
                m_det=mask,
                teacher_fpn_feats=teacher_features,
                student_fpn_feats=features,
                i_frame_bpp_proxy=decoded.get("i_frame_bpp_proxy"),
                p_frame_bpp_proxy=decoded.get("p_frame_bpp_proxy"),
                ctu_qp_offsets=output.get("ctu_qp_offsets"),
                i_orig=video,
                aligned_prev_edit=output.get("aligned_prev_edit"),
                aligned_prev_orig=output.get("aligned_prev_orig"),
                warp_valid_mask=output.get("warp_valid_mask"),
            )
            u_val = float(u_loss["loss_total"].item())
            if not math.isfinite(u_val):
                raise ValueError("Soup uniform validation loss is nonfinite")
            uniform_total += u_val

            if native_codec is not None and native_criterion is not None:
                n_dec = _call_codec_ste(native_codec, output["i_edit"], severity, qp_arg, family_id=fam)
                n_feat = (
                    features
                    if native_codec is uniform_codec
                    else extract_validation_features(teacher, n_dec["video_recon"].reshape(b * t, 3, h, w), chunk_size)
                )
                n_loss = _call_criterion(
                    native_criterion,
                    i_edit=output["i_edit"],
                    i_recon=n_dec["video_recon"],
                    attn_map=output["attn_map"],
                    eff_attn_map=output.get("eff_attn_map"),
                    delta_obj=output["delta_obj"],
                    delta_bg=output["delta_bg"],
                    rate_proxy_bpp=n_dec["rate_proxy_bpp"],
                    m_attn=attn,
                    m_det=mask,
                    teacher_fpn_feats=teacher_features,
                    student_fpn_feats=n_feat,
                    i_frame_bpp_proxy=n_dec.get("i_frame_bpp_proxy"),
                    p_frame_bpp_proxy=n_dec.get("p_frame_bpp_proxy"),
                    ctu_qp_offsets=output.get("ctu_qp_offsets"),
                    i_orig=video,
                    aligned_prev_edit=output.get("aligned_prev_edit"),
                    aligned_prev_orig=output.get("aligned_prev_orig"),
                    warp_valid_mask=output.get("warp_valid_mask"),
                )
                n_val = float(n_loss["loss_total"].item())
                if not math.isfinite(n_val):
                    raise ValueError("Soup native validation loss is nonfinite")
                native_total += n_val
            else:
                native_total += u_val

            count += 1
            del features, u_loss, decoded, output

    del model
    mean_uniform = uniform_total / max(1, count)
    mean_native = native_total / max(1, count)
    if not math.isfinite(mean_uniform) or not math.isfinite(mean_native):
        raise ValueError("Soup validation loss is nonfinite")
    return {
        "uniform_loss_total": mean_uniform,
        "native_preset_loss_total": mean_native,
        "logical_batch_sizes": batch_sizes,
        "batch_severity_loss_count": count,
    }


def _discover_constituent_states(
    checkpoint: Dict[str, Any],
    variant: str,
    soup_dir: Path,
    constituent_dir: Optional[Path] = None,
    constituent_manifest: Optional[Path] = None,
) -> List[Tuple[int, Dict[str, torch.Tensor]]]:
    found: Dict[int, Dict[str, torch.Tensor]] = {}
    if constituent_manifest is not None and constituent_manifest.is_file():
        _, files = read_manifest_files(constituent_manifest, ".pt")
        for entry, path, _ in files:
            ck = torch.load(path, map_location="cpu", weights_only=True)
            if ck.get("variant") == variant and isinstance(ck.get("state_dict"), dict):
                found[int(entry["shard_id"])] = ck["state_dict"]
        if found:
            return [(sid, found[sid]) for sid in sorted(found)]

    search_roots = [r for r in (constituent_dir, soup_dir, soup_dir.parent) if r is not None and r.is_dir()]
    for member in checkpoint.get("constituents", []):
        if not isinstance(member, dict):
            continue
        sid = member.get("shard_id")
        src = member.get("source_path")
        if type(sid) is not int or not isinstance(src, str):
            continue
        for root in search_roots:
            cand = (root / src).resolve() if not Path(src).is_absolute() else Path(src)
            if cand.is_file():
                ck = torch.load(cand, map_location="cpu", weights_only=True)
                if isinstance(ck.get("state_dict"), dict):
                    found[sid] = ck["state_dict"]
                    break
            cand_flat = root / Path(src).name
            if cand_flat.is_file():
                ck = torch.load(cand_flat, map_location="cpu", weights_only=True)
                if isinstance(ck.get("state_dict"), dict):
                    found[sid] = ck["state_dict"]
                    break

    if constituent_dir is not None and constituent_dir.is_dir() and not found:
        for pt_file in sorted(constituent_dir.rglob(f"tscope_{variant}_shard*.pt")):
            ck = torch.load(pt_file, map_location="cpu", weights_only=True)
            if ck.get("variant") == variant and isinstance(ck.get("state_dict"), dict):
                found[int(ck["shard_id"])] = ck["state_dict"]

    return [(sid, found[sid]) for sid in sorted(found)]


@torch.no_grad()
def validate(args: argparse.Namespace) -> Dict[str, Path]:
    data_root = (getattr(args, "data_root", "") or "").strip()
    if data_root.lower() in ("", "auto", "none"):
        data_root = ""
        mounted = _safe_find_mounted_dataset("mot17")
        if mounted is not None:
            data_root = str(mounted)
    img_h = int(getattr(args, "img_h", VALIDATION_PROTOCOL["img_h"]))
    img_w = int(getattr(args, "img_w", VALIDATION_PROTOCOL["img_w"]))

    dataset = _build_val_dataset(data_root, img_h, img_w)
    if len(dataset) != VALIDATION_PROTOCOL["full_clips"] or any(
        meta["type"] != "mot_frames" or meta["seq_name"].split("-FRCNN")[0] != "MOT17-09"
        for meta in dataset.clips
    ):
        raise ValueError(
            "Release validation requires all 131 complete MOT17-09 clips (524 frames), not the four-frame local visualization subset"
        )
    frame_paths = [path for meta in dataset.clips for path in meta["frame_paths"]]
    frame_numbers = [int(path.stem) for path in frame_paths]
    if frame_numbers != list(range(1, 525)):
        raise ValueError("Validation frames must match the complete MOT17-09 evaluation manifest")
    gt_path = dataset.clips[0]["frame_paths"][0].parent.parent / "gt" / "gt.txt"
    fingerprints = {
        "gt_sha256": sha256_file(gt_path),
        "frames": [{"frame": int(path.stem), "sha256": sha256_file(path)} for path in frame_paths],
    }

    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(4)

    # Single-teacher Faster R-CNN v2 matching v3 training (F04 & F20 fix).
    teacher = FrozenFPNDetectorTeacher(pretrained=True, use_yolov8s=False).to(device)
    if not teacher.use_torchvision or teacher.detector is None:
        raise RuntimeError("Validation requires original pretrained Faster R-CNN v2 features")
    teacher.eval()

    selected_indices = list(range(0, len(dataset), VALIDATION_PROTOCOL["val_subsample"]))
    subset = Subset(dataset, selected_indices)
    loader = DataLoader(subset, batch_size=VALIDATION_PROTOCOL["batch_size"], shuffle=False)
    chunk_size = getattr(args, "feature_chunk_size", 1)
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Validation feature chunk size must be positive")

    # Uniform variant-invariant real-libx265 validation criterion (F20 fix).
    uniform_codec = HybridX265STE(use_real_x265_in_forward=True).to(device)
    if not getattr(args, "allow_proxy_in_unit_test", False) and not uniform_codec.use_real_x265:
        raise RuntimeError("F20 contract: uniform validation requires real libx265 (ffmpeg in PATH)")
    uniform_criterion = VCMTemporalPreEditLoss(lambda_temp_bg=1.0, lambda_temp_obj=0.5).to(device)

    # Pre-cache clean teacher features across the 66 subsampled MOT17-09 clips once for all variants & constituents!
    cached_teacher_batches: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = []
    for batch in loader:
        v_clean = batch["video"].to(device)
        b_n, t_n, _, h_n, w_n = v_clean.shape
        t_feats = extract_validation_features(teacher, v_clean.reshape(b_n * t_n, 3, h_n, w_n), chunk_size)
        cached_teacher_batches.append((batch, {k: v.detach() for k, v in t_feats.items()}))

    import torchvision

    ffmpeg_ver = None
    if uniform_codec.use_real_x265:
        try:
            ffmpeg_ver = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
        except Exception:
            ffmpeg_ver = "unavailable"

    provenance: Dict[str, Any] = {
        "measured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": str(device),
        "dtype": "float32",
        "feature_chunk_size": chunk_size,
        "feature_preprocessing": f"normalized_{img_h}x{img_w}_16_9_images_direct_backbone_no_detector_transform",
        "uniform_validation_criterion": {
            "codec": "libx265_real_ste_yuv420p",
            "lambda_temp_bg": 1.0,
            "lambda_temp_obj": 0.5,
            "teacher": "fasterrcnn_resnet50_fpn_v2_single_teacher",
        },
        "ffmpeg_version": ffmpeg_ver,
        "dependencies": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
        },
        "source_sha256": {
            relative: sha256_file(ROOT_DIR / relative)
            for relative in (
                "scripts/validate_soups.py",
                "scripts/experiment_protocol.py",
                "models/frozen_detector_fpn.py",
                "models/temporal_scope.py",
                "losses/vcm_temporal_loss.py",
                "vcm_datasets/vcm_video_datasets.py",
                "video_codecs/hybrid_x265_ste.py",
            )
        },
    }
    weight_path = Path(torch.hub.get_dir()) / "checkpoints" / "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
    provenance["detector"] = {
        "name": "fasterrcnn_resnet50_fpn_v2",
        "weights": "FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT",
        "cached_weights_sha256": sha256_file(weight_path) if weight_path.is_file() else None,
    }

    soup_dir = Path(args.soup_dir)
    constituent_dir = Path(args.constituent_dir) if getattr(args, "constituent_dir", "") else None
    constituent_manifest = Path(args.constituent_manifest) if getattr(args, "constituent_manifest", "") else None
    selection_mode = getattr(args, "selection_mode", "uniform")

    prepared: Dict[str, Dict[str, Any]] = {}
    connectivity_report: Dict[str, Any] = {
        "schema_version": 3,
        "split": "MOT17-09",
        "held_out_test_used": False,
        "protocol": copy.deepcopy(VALIDATION_PROTOCOL),
        "variants": {},
    }

    selected_frames = [int(path.stem) for index in selected_indices for path in dataset.clips[index]["frame_paths"]]
    measured_frame_ids = [f"MOT17-09/{number:06d}" for number in selected_frames]

    for variant in args.variants:
        source = soup_dir / f"tscope_soup_{variant}.pt"
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        validate_soup_provenance(checkpoint, variant)
        preset = ABLATION_PRESETS[variant]

        needs_separate_native = (
            not preset["use_real_x265_ste"]
            or preset["lambda_temp_bg"] != 1.0
            or preset["lambda_temp_obj"] != 0.5
        )
        native_codec = (
            (uniform_codec if preset["use_real_x265_ste"] else HybridX265STE(use_real_x265_in_forward=False).to(device))
            if needs_separate_native
            else None
        )
        native_criterion = (
            VCMTemporalPreEditLoss(
                lambda_temp_bg=preset["lambda_temp_bg"],
                lambda_temp_obj=preset["lambda_temp_obj"],
            ).to(device)
            if needs_separate_native
            else None
        )

        uniform_eval = _score_state_dict_on_mot17_09(
            checkpoint["state_dict"],
            variant,
            preset,
            loader,
            teacher,
            cached_teacher_batches,
            uniform_codec,
            uniform_criterion,
            native_codec,
            native_criterion,
            device,
            chunk_size,
        )

        # Evaluate constituent shards (mean +- std) and Greedy Soup when shard checkpoints are present (F15 fix)
        constituent_states = _discover_constituent_states(
            checkpoint, variant, soup_dir, constituent_dir=constituent_dir, constituent_manifest=constituent_manifest
        )
        constituent_uniform_losses: Dict[int, float] = {}
        greedy_soup_loss: Optional[float] = None
        greedy_shard_ids: List[int] = []
        greedy_state_dict: Optional[Dict[str, torch.Tensor]] = None

        if constituent_states:
            for sid, c_state in constituent_states:
                c_res = _score_state_dict_on_mot17_09(
                    c_state,
                    variant,
                    preset,
                    loader,
                    teacher,
                    cached_teacher_batches,
                    uniform_codec,
                    uniform_criterion,
                    None,
                    None,
                    device,
                    chunk_size,
                )
                constituent_uniform_losses[sid] = c_res["uniform_loss_total"]
                print(
                    f"  [{variant}] constituent shard {sid:02d} uniform_loss={c_res['uniform_loss_total']:.6f}",
                    flush=True,
                )

            # Wortsman et al. (ICML 2022) Greedy Soup on MOT17-09 validation split (ZERO test exposure)
            ranked = sorted(constituent_states, key=lambda pair: constituent_uniform_losses[pair[0]])
            kept_states = [ranked[0][1]]
            greedy_shard_ids = [ranked[0][0]]
            best_greedy_loss = constituent_uniform_losses[ranked[0][0]]
            greedy_state_dict = ranked[0][1]

            for sid, cand_state in ranked[1:]:
                trial_state = average_compatible_states(kept_states + [cand_state])
                trial_res = _score_state_dict_on_mot17_09(
                    trial_state,
                    variant,
                    preset,
                    loader,
                    teacher,
                    cached_teacher_batches,
                    uniform_codec,
                    uniform_criterion,
                    None,
                    None,
                    device,
                    chunk_size,
                )
                if trial_res["uniform_loss_total"] <= best_greedy_loss + 1e-6:
                    kept_states.append(cand_state)
                    greedy_shard_ids.append(sid)
                    best_greedy_loss = trial_res["uniform_loss_total"]
                    greedy_state_dict = trial_state
            greedy_soup_loss = best_greedy_loss

        c_vals = list(constituent_uniform_losses.values())
        c_mean = float(np.mean(c_vals)) if c_vals else None
        c_std = float(np.std(c_vals, ddof=1)) if len(c_vals) > 1 else (0.0 if len(c_vals) == 1 else None)
        u_loss = uniform_eval["uniform_loss_total"]

        use_greedy = (
            selection_mode == "greedy"
            and greedy_state_dict is not None
            and greedy_soup_loss is not None
        )
        if use_greedy:
            checkpoint["state_dict"] = greedy_state_dict
            chosen_loss = greedy_soup_loss
            selection_tag = "greedy_soup_val_mot17_09_no_test_selection"
        else:
            chosen_loss = u_loss
            selection_tag = "uniform_soup_no_test_selection"

        execution = copy.deepcopy(provenance)
        execution["codec_preset"] = copy.deepcopy(preset)
        execution.update(
            selected_clip_indices=selected_indices,
            measured_frame_ids=measured_frame_ids,
            logical_batch_sizes=uniform_eval["logical_batch_sizes"],
            batch_severity_loss_count=uniform_eval["batch_severity_loss_count"],
        )

        connectivity_summary = {
            "uniform_soup_loss_total": u_loss,
            "native_preset_loss_total": uniform_eval["native_preset_loss_total"],
            "constituent_Stopping_losses": [member.get("best_val_loss") for member in checkpoint["constituents"]],
            "constituent_uniform_losses": {str(k): v for k, v in constituent_uniform_losses.items()},
            "constituent_uniform_loss_mean": c_mean,
            "constituent_uniform_loss_std": c_std,
            "greedy_soup_loss_total": greedy_soup_loss,
            "greedy_selected_shard_ids": greedy_shard_ids,
            "linear_mode_connected": (u_loss <= c_mean) if c_mean is not None else None,
            "selected_mode": selection_tag,
        }
        connectivity_report["variants"][variant] = connectivity_summary

        checkpoint["validation"] = {
            "status": "validated",
            "split": "MOT17-09",
            "release_ready": True,
            "loss_total": chosen_loss,
            "uniform_loss_total": chosen_loss,
            "uniform_soup_loss_total": u_loss,
            "native_preset_loss_total": uniform_eval["native_preset_loss_total"],
            "soup_connectivity": connectivity_summary,
            "data_fingerprints": fingerprints,
            "source_soup_sha256": sha256_file(source),
            "protocol": copy.deepcopy(VALIDATION_PROTOCOL),
            "execution_provenance": execution,
            "constituent_best_losses": [member.get("best_val_loss") for member in checkpoint["constituents"]],
            "selection": selection_tag,
        }
        validate_release_validation(checkpoint["validation"])
        prepared[variant] = checkpoint
        print(
            f"{variant}: uniform_loss={u_loss:.6f} | native_loss={uniform_eval['native_preset_loss_total']:.6f}"
            + (
                f" | shards={c_mean:.6f}+-{c_std:.6f} | greedy={greedy_soup_loss:.6f}"
                if (c_mean is not None and c_std is not None and greedy_soup_loss is not None)
                else ""
            )
            + "; no test AP used",
            flush=True,
        )

    destination = Path(args.output_dir)
    paths = {variant: destination / f"tscope_soup_{variant}.pt" for variant in prepared}
    report_path = destination / "soup_connectivity_report.json"
    if any(path.exists() for path in paths.values()) or report_path.exists():
        raise FileExistsError("Validated soup output exists; use a new directory")
    destination.mkdir(parents=True, exist_ok=True)
    for variant, path in paths.items():
        with path.open("xb") as handle:
            torch.save(prepared[variant], handle)
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(connectivity_report, handle, indent=2, allow_nan=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="", help="MOT17 root directory (auto-detects Kaggle mount if omitted)")
    parser.add_argument("--soup_dir", required=True)
    parser.add_argument("--constituent_dir", default="", help="Optional folder containing individual shard .pt files")
    parser.add_argument("--constituent_manifest", default="", help="Optional collection_manifest.json for constituent shards")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--img_h", type=int, default=IMG_H)
    parser.add_argument("--img_w", type=int, default=IMG_W)
    parser.add_argument("--selection_mode", choices=["uniform", "greedy"], default="uniform")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feature_chunk_size", type=int, default=1)
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
