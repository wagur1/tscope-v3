#!/usr/bin/env python3
"""Plot schema-v3, final-ready merged benchmark + external SOTA RD curves (with P-frame bpp & Zero-Shot YOLOv8s)."""
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXPECTED_CRFS = [27, 32, 37, 42]
PRESERVED_CODEC_SETTINGS = {
    "pix_fmt": "yuv420p",
    "gop_len": 32,
    "qg_size": 16,
    "positive_qoffset_cap": 0.14,
    "negative_qoffset_cap": -0.18,
}
CURVE_FIELDS = {
    "total_bpp": "total_bpp",
    "p_bpp": "p_frame_bpp",
    "roi_psnr": "roi_psnr_db",
    "gt_ap50": "gt_ap50_pct",
    "gt_ap95": "gt_ap50_95_pct",
}
BDBR_FIELDS = {
    "bdbr_tot_gt50": "bdbr_tot_gt_ap50_pct",
    "bdbr_p_gt50": "bdbr_p_gt_ap50_pct",
    "bdbr_tot_gt95": "bdbr_tot_gt_ap50_95_pct",
    "bdbr_p_gt95": "bdbr_p_gt_ap50_95_pct",
}
METHOD_LABELS = {
    "anchor_x265": "Anchor libx265 (yuv420p, GOP=32)",
    "deterministic_blur_ema": "Deterministic blur + EMA (non-learned)",
    "spatial_scope_framewise": "Spatial-only SCOPE (real STE, no QG offsets)",
    "tscope_temporal_ctu_qp": "Full T-SCOPE v3 (MC + ConvGRU + QG carry)",
}
VARIANT_LABELS = {
    "full_tscope": "Full T-SCOPE (MC + ConvGRU + real STE + QG carry)",
    "wo_ctu_delta_qp": "w/o CTU ΔQP (MC + ConvGRU + real STE, ΔQP=0)",
    "spatial_only_scope": "Spatial-only SCOPE (framewise + real STE, ΔQP=0)",
    "iwa_gop_qp_reuse": "IWA-GOP QP reuse (keyframe U-Net + P-bypass)",
    "wo_x265_ste_proxy": "w/o real x265 STE (proxy-only STE, ΔQP=4)",
}
COLORS = ["#666666", "#E69F00", "#0072B2", "#D55E00", "#CC79A7"]
MARKERS = ["o", "v", "s", "D", "*"]


class PlotInputError(ValueError):
    """Evidence is not eligible for final v3 figures."""


def _object(value: Any, context: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise PlotInputError(f"{context}: expected an object")
    return value


def _finite(value: Any, context: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise PlotInputError(f"{context}: expected a finite numeric measurement, got {value!r}")
    return float(value)


def validate_crfs(crfs: Any) -> List[int]:
    if not isinstance(crfs, list) or len(crfs) != len(EXPECTED_CRFS) or any(type(q) is not int for q in crfs):
        raise PlotInputError(f"crfs: expected exact integer list {EXPECTED_CRFS}, got {crfs!r}")
    if crfs != EXPECTED_CRFS:
        raise PlotInputError(f"crfs: expected {EXPECTED_CRFS}, got {crfs!r}")
    return list(crfs)


def _extract_ci95(method: Dict[str, Any], field: str, method_key: str) -> Optional[Tuple[float, float]]:
    ci_key = f"{field}_ci95"
    ci_val = method.get(ci_key)
    if ci_val is None:
        return None
    if not isinstance(ci_val, list) or len(ci_val) != 2:
        raise PlotInputError(f"{method_key}.{ci_key}: expected [lo, hi] list, got {ci_val!r}")
    lo = _finite(ci_val[0], f"{method_key}.{ci_key}[0]")
    hi = _finite(ci_val[1], f"{method_key}.{ci_key}[1]")
    if lo > hi + 1e-9:
        raise PlotInputError(f"{method_key}.{ci_key}: lower bound exceeds upper bound ({lo} > {hi})")
    return (lo, hi)


def extract_curve(summary: Dict[str, Any], method_key: str, crfs: List[int]) -> Dict[str, Any]:
    if method_key not in summary:
        raise PlotInputError(f"summary missing method {method_key!r}")
    method = _object(summary[method_key], method_key)
    per_qp = _object(method.get("per_qp"), f"{method_key}.per_qp")
    expected_keys = {str(q) for q in crfs}
    if set(per_qp.keys()) != expected_keys:
        raise PlotInputError(f"{method_key}.per_qp: expected keys {sorted(expected_keys)}, got {sorted(per_qp.keys())}")
    curve: Dict[str, Any] = {target: [] for target in CURVE_FIELDS}
    curve["yolo_ap50"] = []
    for q in crfs:
        point = _object(per_qp[str(q)], f"{method_key}.per_qp.{q}")
        for target, source in CURVE_FIELDS.items():
            curve[target].append(_finite(point.get(source), f"{method_key}.per_qp.{q}.{source}"))
        if point.get("yolo_gt_ap50_pct") is not None:
            curve["yolo_ap50"].append(_finite(point.get("yolo_gt_ap50_pct"), f"{method_key}.per_qp.{q}.yolo_gt_ap50_pct"))
    reasons = method.get("bdbr_unavailable_reasons") or {}
    reasons = _object(reasons, f"{method_key}.bdbr_unavailable_reasons")
    curve["bdbr_unavailable_reasons"] = {}
    curve["bdbr_ci95"] = {}
    for target, field in BDBR_FIELDS.items():
        value = method.get(field)
        if value is None:
            reason = reasons.get(field)
            if not isinstance(reason, str) or not reason.strip():
                raise PlotInputError(f"{method_key}.{field} is null without bdbr_unavailable_reasons[{field!r}]")
            curve[target] = None
            curve["bdbr_unavailable_reasons"][target] = reason.strip()
            curve["bdbr_ci95"][target] = None
        else:
            curve[target] = _finite(value, f"{method_key}.{field}")
            curve["bdbr_ci95"][target] = _extract_ci95(method, field, method_key)
    if method.get("bdbr_tot_yolo_gt_ap50_pct") is not None:
        curve["bdbr_tot_yolo50"] = _finite(method["bdbr_tot_yolo_gt_ap50_pct"], f"{method_key}.bdbr_tot_yolo_gt_ap50_pct")
        curve["bdbr_ci95"]["bdbr_tot_yolo50"] = _extract_ci95(method, "bdbr_tot_yolo_gt_ap50_pct", method_key)
    else:
        curve["bdbr_tot_yolo50"] = None
    return curve


def validate_codec_settings(bundle: Dict[str, Any]) -> Dict[str, Any]:
    configuration = _object(bundle.get("configuration"), "configuration")
    codec = _object(configuration.get("codec"), "configuration.codec")
    for key, expected in PRESERVED_CODEC_SETTINGS.items():
        if isinstance(expected, str):
            actual = codec.get(key)
            if actual != expected:
                raise PlotInputError(f"configuration.codec.{key}: expected {expected!r}, got {actual!r}")
        else:
            actual = _finite(codec.get(key), f"configuration.codec.{key}")
            if actual != expected:
                raise PlotInputError(f"configuration.codec.{key}: expected preserved setting {expected}, got {actual}")
    grid = codec.get("actual_roi_grid")
    if grid is not None and (not isinstance(grid, list) or len(grid) != 2 or any(type(x) is not int or x <= 0 for x in grid)):
        raise PlotInputError("configuration.codec.actual_roi_grid: expected positive [H_qg, W_qg] integer grid")
    return codec


def codec_caption(bundle: Dict[str, Any]) -> str:
    codec = validate_codec_settings(bundle)
    grid = codec.get("actual_roi_grid", [24, 40])
    return (
        f"Format={codec['pix_fmt']}, GOP={int(codec['gop_len'])}, QG={int(codec['qg_size'])} px "
        f"({grid[1]}x{grid[0]} addroi grid); "
        f"qoffset caps [{codec['negative_qoffset_cap']:+.2f}, {codec['positive_qoffset_cap']:+.2f}]"
    )


def validate_bundle(bundle: Any) -> Dict[str, Any]:
    bundle = _object(bundle, "benchmark")
    if type(bundle.get("schema_version")) is not int or bundle["schema_version"] != 3:
        raise PlotInputError("benchmark: schema_version=3 is required; legacy v1/v2 summaries are not accepted")
    eligibility = _object(bundle.get("eligibility"), "eligibility")
    if eligibility.get("final_ready") is not True:
        raise PlotInputError("benchmark: eligibility.final_ready must be true; no partial/smoke-test figures")
    crfs = validate_crfs(bundle.get("crfs"))
    validate_codec_settings(bundle)
    for section, required in (
        ("overall_methods_summary", METHOD_LABELS),
        ("ablation_variants_summary", VARIANT_LABELS),
    ):
        summary = _object(bundle.get(section), section)
        for key in required:
            extract_curve(summary, key, crfs)
        for key in summary:
            extract_curve(summary, key, crfs)
    if "per_sequence_summary" in bundle:
        per_seq = _object(bundle["per_sequence_summary"], "per_sequence_summary")
        for seq_name, seq_obj in per_seq.items():
            seq_dict = _object(seq_obj, f"per_sequence_summary.{seq_name}")
            seq_vars = _object(seq_dict.get("ablation_variants_summary"), f"per_sequence_summary.{seq_name}.ablation_variants_summary")
            for key in VARIANT_LABELS:
                extract_curve(seq_vars, key, crfs)
    return bundle


def load_benchmark(path: Any) -> Dict[str, Any]:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            bundle = json.load(handle)
    except (OSError, ValueError) as exc:
        raise PlotInputError(f"Cannot read merged benchmark {path}: {exc}") from exc
    return validate_bundle(bundle)


def _discover_sota_bundle(benchmark_path: Path, explicit_sota: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    candidates: List[Path] = []
    if explicit_sota is not None:
        candidates.append(Path(explicit_sota))
    else:
        pool_dir = benchmark_path.resolve().parent
        candidates.extend(
            [
                pool_dir / "pulled_sota_00_kaggle" / "sota_and_tracking_report.json",
                pool_dir / "sota_and_tracking_benchmarks.json",
                pool_dir / "sota_and_tracking_report.json",
            ]
        )
        if pool_dir.is_dir():
            candidates.extend(sorted(pool_dir.rglob("sota_and_tracking*.json")))
    for cand in candidates:
        if cand.is_file():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict) and data.get("schema_version") == 3:
                methods = data.get("external_sota_baselines", {}).get("methods")
                if isinstance(methods, dict) and "official_miccunifi_scope" in methods:
                    return data
    return None


def _pchip_bdbr(r_anc: Sequence[float], a_anc: Sequence[float], r_tst: Sequence[float], a_tst: Sequence[float]) -> Optional[float]:
    import numpy as np
    from scipy.interpolate import PchipInterpolator

    r1 = np.asarray(r_anc, dtype=np.float64)
    a1 = np.asarray(a_anc, dtype=np.float64)
    r2 = np.asarray(r_tst, dtype=np.float64)
    a2 = np.asarray(a_tst, dtype=np.float64)
    if len(r1) < 4 or len(r2) < 4 or np.any(r1 <= 0) or np.any(r2 <= 0):
        return None
    if len(np.unique(a1)) < 4 or len(np.unique(a2)) < 4:
        return None
    if np.allclose(r1, r2) and np.allclose(a1, a2):
        return 0.0
    o1, o2 = np.argsort(a1), np.argsort(a2)
    a1, lr1 = a1[o1], np.log10(r1[o1])
    a2, lr2 = a2[o2], np.log10(r2[o2])
    lo, hi = float(max(a1[0], a2[0])), float(min(a1[-1], a2[-1]))
    if hi <= lo:
        return None
    i1 = float(PchipInterpolator(a1, lr1).integrate(lo, hi))
    i2 = float(PchipInterpolator(a2, lr2).integrate(lo, hi))
    return round(float((10.0 ** ((i2 - i1) / (hi - lo)) - 1.0) * 100.0), 2)


def _extract_sota_curve(methods: Dict[str, Any], key: str, crfs: List[int], anc_curve: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    m = methods[key]
    pqp = m["per_qp"]
    curve: Dict[str, Any] = {
        "total_bpp": [float(pqp[str(q)]["total_bpp"]) for q in crfs],
        "p_bpp": [float(pqp[str(q)]["p_frame_bpp"]) for q in crfs],
        "gt_ap50": [float(pqp[str(q)]["gt_ap50_pct"]) for q in crfs],
        "gt_ap95": [float(pqp[str(q)]["gt_ap50_95_pct"]) for q in crfs],
        "yolo_ap50": [float(pqp[str(q)]["yolo_gt_ap50_pct"]) for q in crfs] if all(pqp[str(q)].get("yolo_gt_ap50_pct") is not None for q in crfs) else [],
        "bdbr_tot_gt50": m.get("bdbr_tot_gt_ap50_pct"),
        "bdbr_p_gt50": m.get("bdbr_p_gt_ap50_pct"),
        "bdbr_tot_gt95": m.get("bdbr_tot_gt_ap50_95_pct"),
        "bdbr_p_gt95": m.get("bdbr_p_gt_ap50_95_pct"),
        "bdbr_tot_yolo50": m.get("bdbr_tot_yolo_gt_ap50_pct"),
        "bdbr_ci95": {
            "bdbr_tot_gt50": tuple(m["bdbr_tot_gt_ap50_pct_ci95"]) if isinstance(m.get("bdbr_tot_gt_ap50_pct_ci95"), list) else None,
            "bdbr_p_gt50": tuple(m["bdbr_p_gt_ap50_pct_ci95"]) if isinstance(m.get("bdbr_p_gt_ap50_pct_ci95"), list) else None,
            "bdbr_tot_gt95": tuple(m["bdbr_tot_gt_ap50_95_pct_ci95"]) if isinstance(m.get("bdbr_tot_gt_ap50_95_pct_ci95"), list) else None,
            "bdbr_tot_yolo50": tuple(m["bdbr_tot_yolo_gt_ap50_pct_ci95"]) if isinstance(m.get("bdbr_tot_yolo_gt_ap50_pct_ci95"), list) else None,
        },
        "bdbr_unavailable_reasons": {},
    }
    if curve["bdbr_tot_yolo50"] is None and curve["yolo_ap50"] and anc_curve is not None and anc_curve.get("yolo_ap50"):
        curve["bdbr_tot_yolo50"] = _pchip_bdbr(
            anc_curve["total_bpp"], anc_curve["yolo_ap50"], curve["total_bpp"], curve["yolo_ap50"]
        )
    return curve


def bdbr_text(curve: Dict[str, Any], field: str) -> str:
    value = curve.get(field)
    if value is None:
        reason = curve.get("bdbr_unavailable_reasons", {}).get(field, "N/A")
        return "BDBR unavailable: " + reason
    ci = curve.get("bdbr_ci95", {}).get(field)
    if ci is not None:
        return f"{value:+.2f}% [{ci[0]:+.1f}, {ci[1]:+.1f}]"
    return f"{value:+.2f}% BDBR"


def _style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9.2,
            "axes.titlesize": 10.0,
            "axes.titleweight": "bold",
            "legend.fontsize": 7.4,
            "legend.frameon": True,
            "legend.facecolor": "#FFFFFF",
            "legend.edgecolor": "#E2E8F0",
            "legend.framealpha": 0.92,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "lines.linewidth": 1.85,
        }
    )


def _save(fig: Any, output_dir: Path, stem: str, plt: Any) -> None:
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    plt.close(fig)


def render_figures(bundle: Dict[str, Any], output_dir: Any, sota_bundle: Optional[Dict[str, Any]] = None) -> None:
    validate_bundle(bundle)
    import matplotlib.pyplot as plt
    import numpy as np

    _style(plt)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    crfs = bundle["crfs"]

    sota_methods = (sota_bundle or {}).get("external_sota_baselines", {}).get("methods")
    has_external_sota = isinstance(sota_methods, dict) and all(
        k in sota_methods
        for k in ("anchor_x265", "official_miccunifi_scope", "mpeg_vcm_spatial_resampling", "detector_roi_codec_addroi")
    )

    if has_external_sota:
        anc_c = extract_curve(bundle["overall_methods_summary"], "anchor_x265", crfs)
        full_c = extract_curve(bundle["ablation_variants_summary"], "full_tscope", crfs)
        iwa_c = extract_curve(bundle["ablation_variants_summary"], "iwa_gop_qp_reuse", crfs)
        scope_c = _extract_sota_curve(sota_methods, "official_miccunifi_scope", crfs, anc_curve=anc_c)
        resamp_c = _extract_sota_curve(sota_methods, "mpeg_vcm_spatial_resampling", crfs, anc_curve=anc_c)
        detroi_c = _extract_sota_curve(sota_methods, "detector_roi_codec_addroi", crfs, anc_curve=anc_c)

        sota_specs = [
            ("anchor_x265", "Direct libx265 Anchor", anc_c, "#4B5563", "--", "o", 1.75, 5.2),
            ("official_scope", "Official SCOPE (Dani et al., 2026)", scope_c, "#8E44AD", "-.", "^", 1.85, 5.8),
            ("mpeg_vcm", "MPEG-VCM Spatial Resampling", resamp_c, "#D97706", ":", "v", 1.75, 5.5),
            ("detector_roi", "Detector-ROI addroi (enc. detector)", detroi_c, "#0072B2", "-", "s", 1.80, 5.2),
            ("tscope_iwa", "T-SCOPE IWA-Bypass (113.2 FPS, Ours)", iwa_c, "#009E73", "-", "D", 2.25, 5.8),
            ("tscope_full", "T-SCOPE Full (Ours)", full_c, "#D55E00", "-", "*", 2.45, 7.8),
        ]

        sota_panels = [
            ("(a) In-Domain Faster R-CNN AP@50 vs. Total Rate", "total_bpp", "gt_ap50", "bdbr_tot_gt50", "Total bitrate (bpp)", "Faster R-CNN AP@50 (%)"),
            ("(b) In-Domain Faster R-CNN AP@50 vs. Inter/P-Frame Rate", "p_bpp", "gt_ap50", "bdbr_p_gt50", "Inter / P-frame bitrate (bpp)", "Faster R-CNN AP@50 (%)"),
            ("(c) High-IoU Faster R-CNN AP@[.50:.95] vs. Total Rate", "total_bpp", "gt_ap95", "bdbr_tot_gt95", "Total bitrate (bpp)", "Faster R-CNN AP@[.50:.95] (%)"),
            ("(d) Zero-Shot Unseen YOLOv8s AP@50 vs. Total Rate", "total_bpp", "yolo_ap50", "bdbr_tot_yolo50", "Total bitrate (bpp)", "Zero-Shot YOLOv8s AP@50 (%)"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.4), constrained_layout=True)
        for ax, (ptitle, xf, yf, bf, xl, yl) in zip(axes.flat, sota_panels):
            main_y_vals: List[float] = []
            for key, label, curve, col, ls, mk, lw, ms in sota_specs:
                if not curve.get(yf):
                    continue
                is_ours = key in ("tscope_full", "tscope_iwa")
                ax.plot(
                    curve[xf],
                    curve[yf],
                    marker=mk,
                    markersize=ms,
                    color=col,
                    linestyle=ls,
                    linewidth=lw,
                    zorder=5 if is_ours else 3,
                    label=f"{label}: {bdbr_text(curve, bf)}",
                )
                if key != "mpeg_vcm":
                    main_y_vals.extend(curve[yf])
                if key == "tscope_full":
                    crf_offsets = {27: (-16, 6), 32: (-24, 6), 37: (-26, 5), 42: (6, -3)}
                    for q, x, y in zip(crfs, curve[xf], curve[yf]):
                        ax.annotate(
                            f"CRF {q}",
                            (x, y),
                            xytext=crf_offsets.get(q, (-18, 5)),
                            textcoords="offset points",
                            fontsize=6.8,
                            fontweight="bold",
                            color="#9A3412",
                            bbox=dict(boxstyle="round,pad=0.12", facecolor="#FFFFFF", edgecolor="none", alpha=0.82),
                            zorder=7,
                        )
            if main_y_vals:
                ylo = min(main_y_vals) - 2.2
                yhi = max(main_y_vals) + 1.4
                ax.set_ylim(ylo, yhi)
            ax.set_title(ptitle, pad=6)
            ax.set_xlabel(xl + " (lower is better)")
            ax.set_ylabel(yl + " (higher is better)")
            ax.legend(loc="lower right")
        fig.suptitle(
            "Held-out MOT17 32-frame GOP rate-accuracy comparison vs. external SOTA & CTC baselines (PCHIP BDBR + 10k bootstrap 95% CIs)\n"
            + codec_caption(bundle),
            fontsize=10.2,
            fontweight="bold",
        )
        _save(fig, output_dir, "fig_bdbr_curves", plt)
    else:
        curves = {k: extract_curve(bundle["overall_methods_summary"], k, crfs) for k in METHOD_LABELS}
        curve_plot_items = list(METHOD_LABELS.items())
        if "iwa_gop_qp_reuse" in bundle.get("ablation_variants_summary", {}):
            curves["iwa_gop_qp_reuse"] = extract_curve(
                bundle["ablation_variants_summary"], "iwa_gop_qp_reuse", crfs
            )
            curve_plot_items.insert(
                3,
                ("iwa_gop_qp_reuse", "IWA-GOP QP reuse (keyframe U-Net + P-bypass)"),
            )

        panels = [
            ("total_bpp", "gt_ap50", "bdbr_tot_gt50", "Total bitrate (bpp)", "GT AP@50 (%)"),
            ("total_bpp", "gt_ap95", "bdbr_tot_gt95", "Total bitrate (bpp)", "GT AP@[.50:.95] (%)"),
            ("p_bpp", "gt_ap50", "bdbr_p_gt50", "P-frame bitrate (bpp)", "GT AP@50 (%)"),
            ("p_bpp", "gt_ap95", "bdbr_p_gt95", "P-frame bitrate (bpp)", "GT AP@[.50:.95] (%)"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.2), constrained_layout=True)
        for ax, (xf, yf, bf, xl, yl) in zip(axes.flat, panels):
            for idx, (key, label) in enumerate(curve_plot_items):
                curve = curves[key]
                ax.plot(
                    curve[xf],
                    curve[yf],
                    marker=MARKERS[idx % len(MARKERS)],
                    color=COLORS[idx % len(COLORS)],
                    linestyle="--" if key == "anchor_x265" else "-",
                    label=f"{label}\n{bdbr_text(curve, bf)}",
                )
                if key == "tscope_temporal_ctu_qp":
                    for q, x, y in zip(crfs, curve[xf], curve[yf]):
                        ax.annotate(f"CRF {q}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=6.5)
            ax.set_xlabel(xl + " (lower is better)")
            ax.set_ylabel(yl + " (higher is better)")
            ax.legend(loc="best")
        fig.suptitle(
            "Held-out Total & P-Frame rate-accuracy curves (PCHIP BDBR + 10k clip-bootstrap 95% CIs)\n"
            + codec_caption(bundle),
            fontsize=10,
        )
        _save(fig, output_dir, "fig_bdbr_curves", plt)

    # 2x2 de-confounded ablation bars (Total BDBR & P-Frame BDBR + 95% CI error bars)
    ab_panels = [
        ("total_bpp", "gt_ap50", "bdbr_tot_gt50", "Total bitrate (bpp)", "GT AP@50 (%)"),
        ("total_bpp", "gt_ap95", "bdbr_tot_gt95", "Total bitrate (bpp)", "GT AP@[.50:.95] (%)"),
        ("p_bpp", "gt_ap50", "bdbr_p_gt50", "P-frame bitrate (bpp)", "GT AP@50 (%)"),
        ("p_bpp", "gt_ap95", "bdbr_p_gt95", "P-frame bitrate (bpp)", "GT AP@[.50:.95] (%)"),
    ]
    variants = {k: extract_curve(bundle["ablation_variants_summary"], k, crfs) for k in VARIANT_LABELS}
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.6), constrained_layout=True)
    for ax, (_, _, bf, xl, yl) in zip(axes.flat, ab_panels):
        unavailable = []
        for idx, (key, _) in enumerate(VARIANT_LABELS.items()):
            curve = variants[key]
            value = curve[bf]
            if value is None:
                ax.text(idx, 0.02, "N/A", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
                unavailable.append(f"{key}: {curve['bdbr_unavailable_reasons'][bf]}")
            else:
                ci = curve["bdbr_ci95"].get(bf)
                yerr = [[max(0.0, value - ci[0])], [max(0.0, ci[1] - value)]] if ci is not None else None
                ax.bar(idx, value, color=COLORS[idx], width=0.62, yerr=yerr, capsize=3 if yerr else 0)
                ax.annotate(
                    f"{value:+.1f}%",
                    (idx, value),
                    xytext=(0, 5 if value >= 0 else -5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=7.5,
                )
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xticks(np.arange(len(VARIANT_LABELS)))
        ax.set_xticklabels(list(VARIANT_LABELS.values()), rotation=18, ha="right", fontsize=6.8)
        ax.set_ylabel(f"{xl.split(' (')[0]} BDBR at {yl}")
        ax.margins(y=0.25)
        if unavailable:
            ax.set_title("Unavailable BDBR\n" + "\n".join(unavailable), fontsize=7)
    fig.suptitle(
        "De-confounded 5-variant ablations (balanced shards, PCHIP BDBR + 95% bootstrap CIs)\n"
        + codec_caption(bundle),
        fontsize=10,
    )
    _save(fig, output_dir, "fig_ablation_bdbr", plt)

    # Optional deterministic per-sequence plot (F09 fix)
    per_seq = bundle.get("per_sequence_summary")
    if isinstance(per_seq, dict) and per_seq:
        seq_keys = sorted(per_seq)
        fig, axes = plt.subplots(1, len(seq_keys), figsize=(6.2 * len(seq_keys), 4.8), constrained_layout=True)
        ax_list = [axes] if len(seq_keys) == 1 else list(axes.flat)
        for ax, seq_name in zip(ax_list, seq_keys):
            seq_vars = {
                k: extract_curve(per_seq[seq_name]["ablation_variants_summary"], k, crfs)
                for k in VARIANT_LABELS
            }
            for idx, (key, _) in enumerate(VARIANT_LABELS.items()):
                val = seq_vars[key]["bdbr_tot_gt50"]
                if val is not None:
                    ax.bar(idx, val, color=COLORS[idx], width=0.6)
                    ax.annotate(
                        f"{val:+.1f}%",
                        (idx, val),
                        xytext=(0, 3 if val >= 0 else -3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom" if val >= 0 else "top",
                        fontsize=7.5,
                    )
            ax.axhline(0, color="#333333", linewidth=0.8)
            ax.set_xticks(np.arange(len(VARIANT_LABELS)))
            ax.set_xticklabels(list(VARIANT_LABELS.keys()), rotation=20, ha="right", fontsize=7.5)
            ax.set_title(f"{seq_name} ({per_seq[seq_name].get('frame_count', '?')} frames)")
            ax.set_ylabel("Total BDBR at GT AP@50 (%)")
            ax.margins(y=0.25)
        fig.suptitle("Verified per-sequence BDBR breakdown (single source of truth)\n" + codec_caption(bundle), fontsize=10)
        _save(fig, output_dir, "fig_per_sequence_bdbr", plt)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True, help="Schema-v3 final-ready merged JSON")
    parser.add_argument("--sota-report", type=Path, default=None, help="Optional external SOTA report JSON")
    parser.add_argument("--output-dir", type=Path, help="Explicit local destination; no manuscript copying")
    parser.add_argument("--validate-only", action="store_true", help="Check input without rendering or writing")
    args = parser.parse_args(argv)
    try:
        bundle = load_benchmark(args.benchmark)
        if args.validate_only:
            print(f"Validated schema-v3 final-ready benchmark: {args.benchmark}; CRFs={bundle['crfs']}")
            return
        if args.output_dir is None:
            parser.error("--output-dir is required unless --validate-only")
        sota_bundle = _discover_sota_bundle(args.benchmark, args.sota_report)
        render_figures(bundle, args.output_dir, sota_bundle=sota_bundle)
        print(f"Saved v3 curves and de-confounded ablation plots to {args.output_dir}")
    except PlotInputError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
