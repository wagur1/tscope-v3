#!/usr/bin/env python3
"""Plot only a schema-v3, final-ready merged benchmark; restores P-frame bpp & P-BDBR + 95% bootstrap CIs."""
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    if not isinstance(crfs, list) or any(type(q) is not int for q in crfs) or crfs != EXPECTED_CRFS:
        raise PlotInputError(f"crfs: expected the v3 protocol {EXPECTED_CRFS}, got {crfs!r}")
    return list(crfs)


def _extract_ci95(method: Dict[str, Any], field: str, context: str) -> Optional[Tuple[float, float]]:
    ci_val = method.get(f"{field}_ci95")
    if ci_val is None:
        nested = method.get("bootstrap_ci95")
        if isinstance(nested, dict) and isinstance(nested.get(field), dict):
            ci_val = nested[field].get("ci95")
    if ci_val is None:
        return None
    if not isinstance(ci_val, list) or len(ci_val) != 2:
        raise PlotInputError(f"{context}.{field}_ci95: expected a 2-element [low, high] list")
    low = _finite(ci_val[0], f"{context}.{field}_ci95[0]")
    high = _finite(ci_val[1], f"{context}.{field}_ci95[1]")
    if low > high + 1e-9:
        raise PlotInputError(f"{context}.{field}_ci95: lower bound exceeds upper bound ({low} > {high})")
    return (low, high)


def extract_curve(summary_dict: Any, method_key: str, qps: Any) -> Dict[str, Any]:
    """Validate complete real curves, P-frame bpp, nullable BDBR, and optional bootstrap 95% CIs."""
    qps = validate_crfs(qps)
    summary_dict = _object(summary_dict, "summary")
    if method_key not in summary_dict:
        raise PlotInputError(f"summary: missing method {method_key!r}")
    method = _object(summary_dict[method_key], method_key)
    pqp = _object(method.get("per_qp"), f"{method_key}.per_qp")
    if set(pqp) != {str(q) for q in qps}:
        raise PlotInputError(f"{method_key}.per_qp: expected exactly string CRF keys {qps}")
    curve: Dict[str, Any] = {key: [] for key in CURVE_FIELDS}
    for q in qps:
        row = _object(pqp[str(q)], f"{method_key}.per_qp[{q}]")
        for target, field in CURVE_FIELDS.items():
            value = _finite(row.get(field), f"{method_key}.CRF{q}.{field}")
            if field in ("total_bpp", "p_frame_bpp") and value <= 0:
                raise PlotInputError(f"{method_key}.CRF{q}.{field}: bitrate must be positive")
            if field in ("gt_ap50_pct", "gt_ap50_95_pct") and not 0 <= value <= 100:
                raise PlotInputError(f"{method_key}.CRF{q}.{field}: AP must be in [0, 100]")
            curve[target].append(value)
    reasons = _object(method.get("bdbr_unavailable_reasons", {}), f"{method_key}.bdbr_unavailable_reasons")
    curve["bdbr_unavailable_reasons"] = {}
    curve["bdbr_ci95"] = {}
    for target, field in BDBR_FIELDS.items():
        if field not in method:
            raise PlotInputError(f"{method_key}: missing {field}")
        value = method[field]
        if value is None:
            reason = reasons.get(field)
            if not isinstance(reason, str) or not reason.strip():
                raise PlotInputError(f"{method_key}.{field}: null BDBR requires an unavailable reason")
            curve[target] = None
            curve["bdbr_unavailable_reasons"][target] = reason.strip()
            curve["bdbr_ci95"][target] = None
        else:
            curve[target] = _finite(value, f"{method_key}.{field}")
            curve["bdbr_ci95"][target] = _extract_ci95(method, field, method_key)
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


def bdbr_text(curve: Dict[str, Any], field: str) -> str:
    value = curve[field]
    if value is None:
        return "BDBR unavailable: " + curve["bdbr_unavailable_reasons"][field]
    ci = curve.get("bdbr_ci95", {}).get(field)
    if ci is not None:
        return f"{value:+.2f}% [{ci[0]:+.1f}, {ci[1]:+.1f}] BDBR"
    return f"{value:+.2f}% BDBR"


def _style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "legend.fontsize": 6.8,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
            "lines.linewidth": 1.8,
        }
    )


def _save(fig: Any, output_dir: Path, stem: str, plt: Any) -> None:
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    plt.close(fig)


def render_figures(bundle: Dict[str, Any], output_dir: Any) -> None:
    validate_bundle(bundle)
    import matplotlib.pyplot as plt
    import numpy as np

    _style(plt)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    crfs = bundle["crfs"]
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

    # 2x2 figure restoring P-frame bpp & P-BPP BDBR alongside Total bpp (F21 fix)
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
                marker=MARKERS[idx],
                color=COLORS[idx],
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

    # 2x2 de-confounded ablation bars (Total BDBR & P-Frame BDBR + 95% CI error bars, F07/F15/F21 fix)
    variants = {k: extract_curve(bundle["ablation_variants_summary"], k, crfs) for k in VARIANT_LABELS}
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.6), constrained_layout=True)
    for ax, (_, _, bf, xl, yl) in zip(axes.flat, panels):
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
        render_figures(bundle, args.output_dir)
        print(f"Saved v3 curves and de-confounded ablation plots to {args.output_dir}")
    except PlotInputError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
