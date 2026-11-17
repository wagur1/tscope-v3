#!/usr/bin/env python3
"""Deterministic Single-Source-of-Truth Table & Macro Synchronizer for T-SCOPE v3 (`F09` Elimination).

Ingests checked `schema_version = 3` evaluation JSON bundles:
  1. `outputs_kaggle_pool/merged_benchmarks.json`
  2. `outputs_kaggle_pool/cross_backbone_bdbr.json`
  3. `outputs_kaggle_pool/sota_and_tracking_benchmarks.json` (or `sota_and_tracking_report.json`)
  4. Optional `outputs_kaggle_pool/soup_connectivity_report.json`

And deterministically generates with zero human copy-paste:
  - `paper/metrics_macros.tex`
  - `paper/tables_generated.tex`
  - `docs/V3_Q1_BENCHMARK_REPORT.md`
  - `README.md` benchmark section (`<!-- BEGIN_V3_BENCHMARKS -->...<!-- END_V3_BENCHMARKS -->` or `## 5.` to EOF)

Also supports `--init-placeholders` to emit clean LaTeX-compilable `"TBD"` macros and tables while
distributed GPU training/evaluation is in progress.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.interpolate import PchipInterpolator


SCHEMA_VERSION_EXPECTED = 3
CRFS_DEFAULT = [27, 32, 37, 42]


def _pchip_bdbr_from_per_qp(
    anc_pqp: Optional[Dict[str, Any]],
    tst_pqp: Optional[Dict[str, Any]],
    rate_key: str,
    acc_key: str,
    crfs: Sequence[int],
) -> Optional[float]:
    if not isinstance(anc_pqp, dict) or not isinstance(tst_pqp, dict) or len(crfs) < 4:
        return None
    try:
        r1 = np.asarray([float(anc_pqp[str(q)][rate_key]) for q in crfs], dtype=np.float64)
        a1 = np.asarray([float(anc_pqp[str(q)][acc_key]) for q in crfs], dtype=np.float64)
        r2 = np.asarray([float(tst_pqp[str(q)][rate_key]) for q in crfs], dtype=np.float64)
        a2 = np.asarray([float(tst_pqp[str(q)][acc_key]) for q in crfs], dtype=np.float64)
        if not (np.isfinite(r1).all() and np.isfinite(a1).all() and np.isfinite(r2).all() and np.isfinite(a2).all()):
            return None
        if np.any(r1 <= 0) or np.any(r2 <= 0) or len(np.unique(a1)) < 4 or len(np.unique(a2)) < 4:
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
    except (KeyError, TypeError, ValueError):
        return None

BEGIN_README_MARKER = "<!-- BEGIN_V3_BENCHMARKS -->"
END_README_MARKER = "<!-- END_V3_BENCHMARKS -->"


def _sha256_short(path: Optional[Path]) -> str:
    if path is None or not path.is_file():
        return "pending"
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _sha256_full(path: Optional[Path]) -> str:
    if path is None or not path.is_file():
        return "pending"
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def _fmt_pct(val: Any, signed: bool = True, digits: int = 2, tex: bool = True) -> str:
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        return r"\texttt{TBD}" if tex else "`TBD`"
    num = float(val)
    fmt = f"{num:+0.{digits}f}" if signed else f"{num:0.{digits}f}"
    return f"{fmt}\\%" if tex else f"{fmt}%"


def _fmt_num(val: Any, digits: int = 3, tex: bool = True) -> str:
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        return r"\texttt{TBD}" if tex else "`TBD`"
    return f"{float(val):0.{digits}f}"


def _fmt_ci(ci: Any, digits: int = 2, tex: bool = True) -> str:
    if (
        not isinstance(ci, (list, tuple))
        or len(ci) != 2
        or any(x is None or not math.isfinite(float(x)) for x in ci)
    ):
        return r"\texttt{TBD}" if tex else "`TBD`"
    lo, hi = float(ci[0]), float(ci[1])
    if tex:
        return f"$[{lo:+0.{digits}f},\\,{hi:+0.{digits}f}]\\%$"
    return f"`[{lo:+0.{digits}f}%, {hi:+0.{digits}f}%]`"


def _fmt_pct_with_ci(val: Any, ci: Any, tex: bool = True, bold: bool = False) -> str:
    p_str = _fmt_pct(val, signed=True, digits=2, tex=tex)
    if p_str in (r"\texttt{TBD}", "`TBD`"):
        return p_str
    if bold:
        p_str = f"\\textbf{{{p_str}}}" if tex else f"**{p_str}**"
    if (
        isinstance(ci, (list, tuple))
        and len(ci) == 2
        and all(x is not None and math.isfinite(float(x)) for x in ci)
    ):
        lo, hi = float(ci[0]), float(ci[1])
        if tex:
            return f"{p_str} {{\\scriptsize $[{lo:+.1f},{hi:+.1f}]$}}"
        return f"{p_str} (`[{lo:+.2f}, {hi:+.2f}]`)"
    return p_str


def _fmt_curve(per_qp: Optional[Dict[str, Any]], key: str, crfs: Sequence[int], digits: int = 3, tex: bool = True) -> str:
    if not isinstance(per_qp, dict):
        return r"\texttt{TBD}" if tex else "`TBD`"
    vals: List[str] = []
    for q in crfs:
        row = per_qp.get(str(q), {})
        v = row.get(key) if isinstance(row, dict) else None
        if v is None or not math.isfinite(float(v)):
            return r"\texttt{TBD}" if tex else "`TBD`"
        vals.append(f"{float(v):0.{digits}f}")
    joined = " / ".join(vals)
    return joined if tex else f"`{joined}`"


def _resolve_optional_json(explicit: Optional[str], candidates: Sequence[Path]) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _load_checked_json(path: Optional[Path], require_final_ready: bool = False) -> Optional[Dict[str, Any]]:
    if path is None or not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    schema_ver = data.get("schema_version")
    if schema_ver != SCHEMA_VERSION_EXPECTED:
        raise ValueError(
            f"Schema mismatch in {path}: expected schema_version={SCHEMA_VERSION_EXPECTED}, got {schema_ver!r}"
        )
    if require_final_ready:
        elig = data.get("eligibility", {})
        if not isinstance(elig, dict) or elig.get("final_ready") is not True:
            raise ValueError(f"Benchmark {path} is not marked eligibility.final_ready=True")
    return data


def _discover_sota_json_paths(
    explicit: Optional[Union[str, Sequence[str]]],
    pool_dir: Path,
) -> List[Path]:
    """Resolves one or more `sota_and_tracking*.json` files (supports task-split multi-kernel runs)."""
    if explicit is not None:
        raw_items = [explicit] if isinstance(explicit, str) else [x for x in explicit if x]
        if raw_items:
            resolved: List[Path] = []
            seen: set[Path] = set()
            for item in raw_items:
                p = Path(item)
                if not p.is_file():
                    raise FileNotFoundError(f"Explicit --sota-tracking-json file not found: {p}")
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    resolved.append(p)
            return resolved

    candidates: List[Path] = [
        pool_dir / "sota_and_tracking_benchmarks.json",
        pool_dir / "sota_and_tracking_report.json",
        pool_dir / "eval_sota_and_tracking" / "sota_and_tracking_report.json",
    ]
    if pool_dir.is_dir():
        candidates.extend(sorted(pool_dir.rglob("sota_and_tracking*.json")))

    found: List[Path] = []
    seen_resolved: set[Path] = set()
    for cand in candidates:
        if cand.is_file():
            rc = cand.resolve()
            if rc not in seen_resolved:
                seen_resolved.add(rc)
                found.append(cand)
    return found


def _load_and_merge_sota_jsons(paths: Sequence[Path]) -> Optional[Dict[str, Any]]:
    """Deep-merges top-level sections across one or more `sota_and_tracking*.json` bundles."""
    bundles = [b for p in paths if (b := _load_checked_json(p, require_final_ready=False)) is not None]
    if not bundles:
        return None
    if len(bundles) == 1:
        return bundles[0]

    merged_out: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_EXPECTED,
        "provenance": {
            "tasks_executed": [],
            "merged_source_files": [str(p.name) for p in paths],
        },
    }
    tasks_union: set[str] = set()
    ext_repos: Dict[str, Any] = {}
    ext_methods: Dict[str, Any] = {}
    trk_methods: Dict[str, Any] = {}
    sec_real: Dict[str, Any] = {}
    sec_skipped_fallback: Optional[Dict[str, Any]] = None
    comp_merged: Dict[str, Any] = {}
    comp_np: Dict[str, Any] = {}
    comp_cw: Dict[str, Any] = {}

    for b in bundles:
        prov = b.get("provenance", {})
        if isinstance(prov, dict):
            for k, v in prov.items():
                if k == "tasks_executed" and isinstance(v, list):
                    tasks_union.update(str(t) for t in v)
                elif k not in merged_out["provenance"]:
                    merged_out["provenance"][k] = v

        ext = b.get("external_sota_baselines")
        if isinstance(ext, dict):
            for rk, rv in (ext.get("external_repositories_status") or {}).items():
                prev_r = ext_repos.get(rk)
                if prev_r is None or (isinstance(rv, dict) and rv.get("executed") and not prev_r.get("executed")):
                    ext_repos[rk] = rv
            for mk, mv in (ext.get("methods") or {}).items():
                if isinstance(mv, dict):
                    ext_methods[mk] = {**ext_methods.get(mk, {}), **mv}

        trk = b.get("mot17_tracking_metrics")
        if isinstance(trk, dict):
            for mk, mv in trk.items():
                if isinstance(mv, dict):
                    trk_methods[mk] = {**trk_methods.get(mk, {}), **mv}

        sec = b.get("second_dataset_vcm_summary")
        if isinstance(sec, dict):
            if sec.get("status") == "skipped":
                sec_skipped_fallback = sec
            else:
                for dk, dv in sec.items():
                    if dk not in ("status", "reason") and isinstance(dv, dict):
                        sec_real[dk] = dv

        comp = b.get("complexity_runtime_table")
        if isinstance(comp, dict):
            for ck, cv in comp.items():
                if ck == "neural_pre_editors" and isinstance(cv, dict):
                    comp_np.update(cv)
                elif ck == "codec_wall_clock" and isinstance(cv, dict):
                    comp_cw.update(cv)
                elif cv is not None:
                    comp_merged[ck] = cv

    merged_out["provenance"]["tasks_executed"] = sorted(tasks_union)
    if ext_repos or ext_methods:
        merged_out["external_sota_baselines"] = {
            "external_repositories_status": ext_repos,
            "methods": ext_methods,
        }
    if trk_methods:
        merged_out["mot17_tracking_metrics"] = trk_methods
    if sec_real:
        merged_out["second_dataset_vcm_summary"] = sec_real
    elif sec_skipped_fallback is not None:
        merged_out["second_dataset_vcm_summary"] = sec_skipped_fallback
    if comp_merged or comp_np or comp_cw:
        if comp_np:
            comp_merged["neural_pre_editors"] = comp_np
        if comp_cw:
            comp_merged["codec_wall_clock"] = comp_cw
        merged_out["complexity_runtime_table"] = comp_merged

    return merged_out


def build_macros_dict(
    merged: Optional[Dict[str, Any]],
    cross: Optional[Dict[str, Any]],
    sota_trk: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    """Builds the complete LaTeX macro dictionary (deterministic constants + empirical metrics)."""
    tbd = r"\texttt{TBD}"
    macros: Dict[str, str] = {
        # Exact architectural & protocol constants from tscope-v3 codebase
        "tscopeTotalParams": "6,108,814",
        "tscopeEvalFrames": "1,632",
        "tscopeEvalClips": "51",
        "tscopeGopLen": "32",
        "tscopeFullGmacsPerFrame": "137.78",
        "tscopeIwaGmacsPerFrame": "4.52",
        "tscopeSpatialGmacsPerFrame": "107.11",
        "tscopeIwaAnalyticalGmacSaving": r"96.72\%",
        # Empirical macros default to TBD until JSONs land
        "tscopeManifestShaShort": r"\texttt{pending}",
        "tscopeFullTotBdbrFrcnnApFifty": tbd,
        "tscopeFullTotBdbrFrcnnApFiftyCi": tbd,
        "tscopeFullPBdbrFrcnnApFifty": tbd,
        "tscopeFullPBdbrFrcnnApFiftyCi": tbd,
        "tscopeFullTotBdbrFrcnnApNinetyFive": tbd,
        "tscopeFullPBdbrFrcnnApNinetyFive": tbd,
        "tscopeFullTotBdbrYoloApFifty": tbd,
        "tscopeFullTotBdbrYoloApFiftyCi": tbd,
        "tscopeFullPBdbrYoloApFifty": tbd,
        "tscopeFullTotBdbrYoloApNinetyFive": tbd,
        "tscopeIwaTotBdbrFrcnnApFifty": tbd,
        "tscopeIwaPBdbrFrcnnApFifty": tbd,
        "tscopeWoDqpTotBdbrFrcnnApFifty": tbd,
        "tscopeWoDqpPBdbrFrcnnApFifty": tbd,
        "tscopeSpatialTotBdbrFrcnnApFifty": tbd,
        "tscopeSpatialPBdbrFrcnnApFifty": tbd,
        "tscopeWoSteTotBdbrFrcnnApFifty": tbd,
        "tscopeWoStePBdbrFrcnnApFifty": tbd,
        "tscopeBlurEmaTotBdbrFrcnnApFifty": tbd,
        "tscopeOracleTotBdbrFrcnnApFifty": tbd,
        "tscopeMotElevenFullBdbrFrcnnApFifty": tbd,
        "tscopeMotThirteenFullBdbrFrcnnApFifty": tbd,
        "tscopeClaimOneStatus": tbd,
        "tscopeClaimTwoDelta": tbd,
        "tscopeClaimTwoStatus": tbd,
        "tscopeClaimThreeDelta": tbd,
        "tscopeClaimFourDelta": tbd,
        "tscopeClaimFiveStatus": tbd,
        "tscopeOffScopeTotBdbrFrcnnApFifty": tbd,
        "tscopeResampleTotBdbrFrcnnApFifty": tbd,
        "tscopeDetRoiTotBdbrFrcnnApFifty": tbd,
        "tscopeFtDetTotBdbrFrcnnApFifty": tbd,
        "tscopeZeroShotRetinaBdbr": tbd,
        "tscopeZeroShotFcosBdbr": tbd,
        "tscopeZeroShotSsdBdbr": tbd,
        "tscopeZeroShotSwinSnrBdbr": tbd,
        "tscopeAvcFullBdbrFrcnnApFifty": tbd,
        "tscopeRaFullBdbrFrcnnApFifty": tbd,
        "tscopeMotMotaBdbr": tbd,
        "tscopeMotIdfOneBdbr": tbd,
        "tscopeMotHotaBdbr": tbd,
        "tscopeMotTwentyMapFiftyBdbr": tbd,
        "tscopeKittiMapFiftyBdbr": tbd,
        "tscopeFullMeanMs": tbd,
        "tscopeFullFps": tbd,
        "tscopeIwaMeanMs": tbd,
        "tscopeIwaFps": tbd,
        "tscopeIrlsMeanMs": tbd,
        "tscopePeakVramMb": tbd,
        "tscopeAnchorEncMs": tbd,
        "tscopeFullDynEncMs": tbd,
        "tscopeIwaStatEncMs": tbd,
    }

    if merged is not None:
        elig = merged.get("eligibility", {})
        if "frame_count" in elig:
            macros["tscopeEvalFrames"] = f"{int(elig['frame_count']):,}"
        if "clip_count" in elig:
            macros["tscopeEvalClips"] = str(int(elig["clip_count"]))
        m_sha = str(merged.get("source_manifest_sha256", ""))[:12]
        if m_sha:
            macros["tscopeManifestShaShort"] = f"\\texttt{{{_tex_escape(m_sha)}}}"

        ov = merged.get("overall_methods_summary", {})
        ab = merged.get("ablation_variants_summary", {})
        full = ab.get("full_tscope", ov.get("tscope_temporal_ctu_qp", {}))
        iwa = ab.get("iwa_gop_qp_reuse", {})
        wodqp = ab.get("wo_ctu_delta_qp", {})
        spat = ab.get("spatial_only_scope", ov.get("spatial_scope_framewise", {}))
        woste = ab.get("wo_x265_ste_proxy", {})
        blur = ov.get("deterministic_blur_ema", {})
        orac = ov.get("oracle_gt_mask_upper_bound", {})

        macros["tscopeFullTotBdbrFrcnnApFifty"] = _fmt_pct(full.get("bdbr_tot_gt_ap50_pct"))
        macros["tscopeFullTotBdbrFrcnnApFiftyCi"] = _fmt_ci(full.get("bdbr_tot_gt_ap50_pct_ci95"))
        macros["tscopeFullPBdbrFrcnnApFifty"] = _fmt_pct(full.get("bdbr_p_gt_ap50_pct"))
        macros["tscopeFullPBdbrFrcnnApFiftyCi"] = _fmt_ci(full.get("bdbr_p_gt_ap50_pct_ci95"))
        macros["tscopeFullTotBdbrFrcnnApNinetyFive"] = _fmt_pct(full.get("bdbr_tot_gt_ap50_95_pct"))
        macros["tscopeFullPBdbrFrcnnApNinetyFive"] = _fmt_pct(full.get("bdbr_p_gt_ap50_95_pct"))
        macros["tscopeFullTotBdbrYoloApFifty"] = _fmt_pct(full.get("bdbr_tot_yolo_gt_ap50_pct"))
        macros["tscopeFullTotBdbrYoloApFiftyCi"] = _fmt_ci(full.get("bdbr_tot_yolo_gt_ap50_pct_ci95"))
        macros["tscopeFullPBdbrYoloApFifty"] = _fmt_pct(full.get("bdbr_p_yolo_gt_ap50_pct"))
        macros["tscopeFullTotBdbrYoloApNinetyFive"] = _fmt_pct(full.get("bdbr_tot_yolo_gt_ap50_95_pct"))

        macros["tscopeIwaTotBdbrFrcnnApFifty"] = _fmt_pct(iwa.get("bdbr_tot_gt_ap50_pct"))
        macros["tscopeIwaPBdbrFrcnnApFifty"] = _fmt_pct(iwa.get("bdbr_p_gt_ap50_pct"))
        macros["tscopeWoDqpTotBdbrFrcnnApFifty"] = _fmt_pct(wodqp.get("bdbr_tot_gt_ap50_pct"))
        macros["tscopeWoDqpPBdbrFrcnnApFifty"] = _fmt_pct(wodqp.get("bdbr_p_gt_ap50_pct"))
        macros["tscopeSpatialTotBdbrFrcnnApFifty"] = _fmt_pct(spat.get("bdbr_tot_gt_ap50_pct"))
        macros["tscopeSpatialPBdbrFrcnnApFifty"] = _fmt_pct(spat.get("bdbr_p_gt_ap50_pct"))
        macros["tscopeWoSteTotBdbrFrcnnApFifty"] = _fmt_pct(woste.get("bdbr_tot_gt_ap50_pct"))
        macros["tscopeWoStePBdbrFrcnnApFifty"] = _fmt_pct(woste.get("bdbr_p_gt_ap50_pct"))
        macros["tscopeBlurEmaTotBdbrFrcnnApFifty"] = _fmt_pct(blur.get("bdbr_tot_gt_ap50_pct"))
        macros["tscopeOracleTotBdbrFrcnnApFifty"] = _fmt_pct(orac.get("bdbr_tot_gt_ap50_pct"))

        pseq = merged.get("per_sequence_summary", {})
        s11 = pseq.get("MOT17-11", {}).get("ablation_variants_summary", {}).get("full_tscope", {})
        s13 = pseq.get("MOT17-13", {}).get("ablation_variants_summary", {}).get("full_tscope", {})
        macros["tscopeMotElevenFullBdbrFrcnnApFifty"] = _fmt_pct(s11.get("bdbr_tot_gt_ap50_pct"))
        macros["tscopeMotThirteenFullBdbrFrcnnApFifty"] = _fmt_pct(s13.get("bdbr_tot_gt_ap50_pct"))

        claims = merged.get("claim_falsification_audit", {}).get("claims", {})
        c1 = claims.get("C01_total_bdbr_vs_anchor", {})
        c2 = claims.get("C02_long_gop_p_frame_rate_reduction", {})
        c3 = claims.get("C03_ctu_delta_qp_gain", {})
        c4 = claims.get("C04_real_codec_ste_gain", {})
        c5 = claims.get("C05_zero_shot_unseen_yolov8s_transfer", {})
        if c1.get("status"):
            macros["tscopeClaimOneStatus"] = f"\\texttt{{{_tex_escape(c1['status'])}}}"
        macros["tscopeClaimTwoDelta"] = _fmt_pct(c2.get("delta_bdbr_pct"))
        if c2.get("status"):
            macros["tscopeClaimTwoStatus"] = f"\\texttt{{{_tex_escape(c2['status'])}}}"
        macros["tscopeClaimThreeDelta"] = _fmt_pct(c3.get("delta_bdbr_pct"))
        macros["tscopeClaimFourDelta"] = _fmt_pct(c4.get("delta_bdbr_pct"))
        if c5.get("status"):
            macros["tscopeClaimFiveStatus"] = f"\\texttt{{{_tex_escape(c5['status'])}}}"

    if cross is not None:
        regimes = cross.get("regimes", {})
        hevc_ldp = regimes.get("hevc_x265_low_delay_p", {}).get("backbones", cross.get("backbones", {}))
        macros["tscopeZeroShotRetinaBdbr"] = _fmt_pct(
            hevc_ldp.get("retinanet_r50_fpn_v2", {}).get("tscope_total_bdbr_gt_ap50_pct")
        )
        macros["tscopeZeroShotFcosBdbr"] = _fmt_pct(
            hevc_ldp.get("fcos_r50_fpn", {}).get("tscope_total_bdbr_gt_ap50_pct")
        )
        macros["tscopeZeroShotSsdBdbr"] = _fmt_pct(
            hevc_ldp.get("ssdlite_mobilenet_v3", {}).get("tscope_total_bdbr_gt_ap50_pct")
        )
        macros["tscopeZeroShotSwinSnrBdbr"] = _fmt_pct(
            hevc_ldp.get("swin_t_transformer", {}).get("tscope_total_bdbr_feat_snr_pct")
        )
        avc_ldp = regimes.get("avc_x264_low_delay_p", {}).get("backbones", {})
        macros["tscopeAvcFullBdbrFrcnnApFifty"] = _fmt_pct(
            avc_ldp.get("fasterrcnn_r50_fpn_v2", {}).get("tscope_total_bdbr_gt_ap50_pct")
        )
        hevc_ra = regimes.get("hevc_x265_random_access_ibbp", {}).get("backbones", {})
        macros["tscopeRaFullBdbrFrcnnApFifty"] = _fmt_pct(
            hevc_ra.get("fasterrcnn_r50_fpn_v2", {}).get("tscope_total_bdbr_gt_ap50_pct")
        )

    if sota_trk is not None:
        sota_m = sota_trk.get("external_sota_baselines", {}).get("methods", {})
        macros["tscopeOffScopeTotBdbrFrcnnApFifty"] = _fmt_pct(
            sota_m.get("official_miccunifi_scope", {}).get("bdbr_tot_gt_ap50_pct")
        )
        macros["tscopeResampleTotBdbrFrcnnApFifty"] = _fmt_pct(
            sota_m.get("mpeg_vcm_spatial_resampling", {}).get("bdbr_tot_gt_ap50_pct")
        )
        macros["tscopeDetRoiTotBdbrFrcnnApFifty"] = _fmt_pct(
            sota_m.get("detector_roi_codec_addroi", {}).get("bdbr_tot_gt_ap50_pct")
        )
        macros["tscopeFtDetTotBdbrFrcnnApFifty"] = _fmt_pct(
            sota_m.get("compressed_video_finetuned_frcnn", {}).get("bdbr_tot_gt_ap50_pct")
        )

        trk_full = sota_trk.get("mot17_tracking_metrics", {}).get("tscope_full", {})
        macros["tscopeMotMotaBdbr"] = _fmt_pct(trk_full.get("bdbr_tot_mota_pct"))
        macros["tscopeMotIdfOneBdbr"] = _fmt_pct(trk_full.get("bdbr_tot_idf1_pct"))
        macros["tscopeMotHotaBdbr"] = _fmt_pct(trk_full.get("bdbr_tot_hota_pct"))

        sec_ds = sota_trk.get("second_dataset_vcm_summary", {})
        if isinstance(sec_ds, dict):
            mot20_full = sec_ds.get("mot20", {}).get("methods", {}).get("tscope_full", {})
            kitti_full = sec_ds.get("kitti", {}).get("methods", {}).get("tscope_full", {})
            macros["tscopeMotTwentyMapFiftyBdbr"] = _fmt_pct(mot20_full.get("bdbr_tot_map50_pct"))
            macros["tscopeKittiMapFiftyBdbr"] = _fmt_pct(kitti_full.get("bdbr_tot_map50_pct"))

        comp = sota_trk.get("complexity_runtime_table", {})
        np_p = comp.get("neural_pre_editors", {})
        cw_p = comp.get("codec_wall_clock", {})
        if "full_tscope" in np_p:
            macros["tscopeFullMeanMs"] = _fmt_num(np_p["full_tscope"].get("mean_ms_per_frame"), digits=2)
            macros["tscopeFullFps"] = _fmt_num(np_p["full_tscope"].get("fps"), digits=1)
            macros["tscopePeakVramMb"] = _fmt_num(np_p["full_tscope"].get("peak_vram_mb"), digits=1)
        if "iwa_gop_qp_reuse" in np_p:
            macros["tscopeIwaMeanMs"] = _fmt_num(np_p["iwa_gop_qp_reuse"].get("mean_ms_per_frame"), digits=2)
            macros["tscopeIwaFps"] = _fmt_num(np_p["iwa_gop_qp_reuse"].get("fps"), digits=1)
        if "cauchy_irls_6dof_motion_estimator_only" in np_p:
            macros["tscopeIrlsMeanMs"] = _fmt_num(
                np_p["cauchy_irls_6dof_motion_estimator_only"].get("mean_ms_per_frame"), digits=2
            )
        if "anchor_libx265_unedited" in cw_p:
            macros["tscopeAnchorEncMs"] = _fmt_num(
                cw_p["anchor_libx265_unedited"].get("mean_encode_decode_ms_per_frame"), digits=2
            )
        if "tscope_full_libx265_dynamic_addroi" in cw_p:
            macros["tscopeFullDynEncMs"] = _fmt_num(
                cw_p["tscope_full_libx265_dynamic_addroi"].get("mean_encode_decode_ms_per_frame"), digits=2
            )
        if "tscope_iwa_libx265_static_addroi" in cw_p:
            macros["tscopeIwaStatEncMs"] = _fmt_num(
                cw_p["tscope_iwa_libx265_static_addroi"].get("mean_encode_decode_ms_per_frame"), digits=2
            )

    return macros


def render_metrics_macros_tex(macros: Dict[str, str], provenance_comment: str) -> str:
    lines = [
        "% AUTO-GENERATED BY scripts/sync_paper_tables.py — DO NOT EDIT MANUALLY.",
        f"% Provenance: {provenance_comment}",
    ]
    for name in sorted(macros):
        lines.append(f"\\newcommand{{\\{name}}}{{{macros[name]}}}")
    return "\n".join(lines) + "\n"


def render_tables_generated_tex(
    merged: Optional[Dict[str, Any]],
    cross: Optional[Dict[str, Any]],
    sota_trk: Optional[Dict[str, Any]],
    soup_conn: Optional[Dict[str, Any]] = None,
) -> str:
    """Renders Tables II..VI in IEEEtran two-column/one-column booktabs LaTeX."""
    crfs = merged.get("crfs", CRFS_DEFAULT) if merged else CRFS_DEFAULT
    ov = merged.get("overall_methods_summary", {}) if merged else {}
    ab = merged.get("ablation_variants_summary", {}) if merged else {}
    pseq = merged.get("per_sequence_summary", {}) if merged else {}
    claims = merged.get("claim_falsification_audit", {}).get("claims", {}) if merged else {}
    sota_m = sota_trk.get("external_sota_baselines", {}).get("methods", {}) if sota_trk else {}

    anc_pqp = (ov.get("anchor_x265") or sota_m.get("anchor_x265") or {}).get("per_qp")

    def _resolve_yolo_bdbr(src: Dict[str, Any], acc_key: str, direct_key: str, tex: bool) -> str:
        if src.get(direct_key) is not None:
            return _fmt_pct(src.get(direct_key), tex=tex)
        pqp = src.get("per_qp")
        if isinstance(pqp, dict) and pqp:
            if all(isinstance(pqp.get(str(q)), dict) and acc_key not in pqp[str(q)] for q in crfs):
                return "--" if tex else "`--`"
            derived = _pchip_bdbr_from_per_qp(anc_pqp, pqp, "total_bpp", acc_key, crfs)
            if derived is not None:
                return _fmt_pct(derived, tex=tex)
        return r"\texttt{TBD}" if tex else "`TBD`"

    def _sota_or_ov_row(
        label: str,
        category: str,
        primary: Optional[Dict[str, Any]],
        fallback: Optional[Dict[str, Any]] = None,
        bold: bool = False,
    ) -> str:
        src = primary if (primary and primary.get("per_qp")) else (fallback or {})
        tot_fr50 = _fmt_pct_with_ci(
            src.get("bdbr_tot_gt_ap50_pct"),
            src.get("bdbr_tot_gt_ap50_pct_ci95", src.get("bdbr_tot_gt_ap50_ci95")),
            tex=True,
            bold=bold,
        )
        p_fr50 = _fmt_pct(src.get("bdbr_p_gt_ap50_pct"), tex=True)
        tot_fr95 = _fmt_pct(src.get("bdbr_tot_gt_ap50_95_pct"), tex=True)
        p_fr95 = _fmt_pct(src.get("bdbr_p_gt_ap50_95_pct"), tex=True)
        tot_yo50 = _resolve_yolo_bdbr(src, "yolo_gt_ap50_pct", "bdbr_tot_yolo_gt_ap50_pct", tex=True)
        tot_yo95 = _resolve_yolo_bdbr(src, "yolo_gt_ap50_95_pct", "bdbr_tot_yolo_gt_ap50_95_pct", tex=True)
        if bold and p_fr50 != r"\texttt{TBD}":
            p_fr50 = f"\\textbf{{{p_fr50}}}"
        lbl = f"\\textbf{{{label}}}" if bold else label
        return f"{lbl} & {category} & {tot_fr50} & {p_fr50} & {tot_fr95} & {p_fr95} & {tot_yo50} & {tot_yo95} \\\\"

    # 1. Table II: Main SOTA & Baseline Comparison
    t2_rows = [
        _sota_or_ov_row(r"Direct \texttt{libx265} Anchor", "Standard HEVC 4:2:0", ov.get("anchor_x265"), sota_m.get("anchor_x265")),
        _sota_or_ov_row("Deterministic Blur + EMA", "Non-Learned Filter", ov.get("deterministic_blur_ema")),
        _sota_or_ov_row("MPEG-VCM Spatial Resampling", "Down/Upscale CTC", sota_m.get("mpeg_vcm_spatial_resampling")),
        _sota_or_ov_row(r"Detector-ROI \texttt{addroi}", "Codec QG Only", sota_m.get("detector_roi_codec_addroi")),
        _sota_or_ov_row("Compressed-Domain FT FRCNN", "Adapted Detector", sota_m.get("compressed_video_finetuned_frcnn")),
        _sota_or_ov_row(r"Official SCOPE~\cite{dani2026scope}", "External Image SOTA", sota_m.get("official_miccunifi_scope")),
        _sota_or_ov_row("T-SCOPE Spatial-Only", "Framewise Pre-Edit", ab.get("spatial_only_scope"), sota_m.get("tscope_spatial_only")),
        _sota_or_ov_row(r"T-SCOPE IWA-Bypass ($K{=}32$)", "Keyframe + P-Bypass", ab.get("iwa_gop_qp_reuse"), sota_m.get("tscope_iwa_gop_qp_reuse")),
        _sota_or_ov_row("T-SCOPE Full (Ours)", "MC + QG P-Carry", ab.get("full_tscope"), sota_m.get("tscope_full"), bold=True),
        r"\midrule",
        _sota_or_ov_row(r"Oracle GT-Mask Bound$^\dagger$", "Non-Deployable GT", ov.get("oracle_gt_mask_upper_bound")),
    ]

    # 2. Table III: De-Confounded 5-Variant Ablation + Per-Sequence + Pre-Registered Claims
    ab_specs = [
        ("full_tscope", r"\textbf{Full T-SCOPE}", r"\checkmark", r"\checkmark", r"\checkmark", "Dynamic", True),
        ("iwa_gop_qp_reuse", r"\texttt{iwa\_gop\_qp\_reuse}", r"\checkmark", r"P-Bypass", r"\checkmark", "Keyframe", False),
        ("wo_ctu_delta_qp", r"\texttt{wo\_ctu\_delta\_qp}", r"\checkmark", r"\checkmark", "--", "--", False),
        ("spatial_only_scope", r"\texttt{spatial\_only\_scope}", "--", r"\checkmark", "--", "--", False),
        ("wo_x265_ste_proxy", r"\texttt{wo\_x265\_ste\_proxy}", r"\checkmark", "Proxy Only", r"\checkmark", "Dynamic", False),
    ]
    t3_rows: List[str] = []
    conn_variants = soup_conn.get("variants", {}) if isinstance(soup_conn, dict) else {}
    for v_key, v_lbl, mc_f, ste_f, dqp_f, carry_f, is_bold in ab_specs:
        ve = ab.get(v_key, {})
        s11 = pseq.get("MOT17-11", {}).get("ablation_variants_summary", {}).get(v_key, {})
        s13 = pseq.get("MOT17-13", {}).get("ablation_variants_summary", {}).get(v_key, {})
        v_conn = conn_variants.get(v_key, {})
        val_u = v_conn.get("uniform_soup_loss_total", v_conn.get("uniform_loss_total"))
        val_str = _fmt_num(val_u, digits=3, tex=True) if val_u is not None else r"\texttt{TBD}"
        tot_50 = _fmt_pct_with_ci(
            ve.get("bdbr_tot_gt_ap50_pct"),
            ve.get("bdbr_tot_gt_ap50_pct_ci95"),
            tex=True,
            bold=is_bold,
        )
        p_50 = _fmt_pct(ve.get("bdbr_p_gt_ap50_pct"), tex=True)
        tot_95 = _fmt_pct(ve.get("bdbr_tot_gt_ap50_95_pct"), tex=True)
        yo_50 = _fmt_pct(ve.get("bdbr_tot_yolo_gt_ap50_pct"), tex=True)
        m11_50 = _fmt_pct(s11.get("bdbr_tot_gt_ap50_pct"), tex=True)
        m13_50 = _fmt_pct(s13.get("bdbr_tot_gt_ap50_pct"), tex=True)
        t3_rows.append(
            f"{v_lbl} & {mc_f} & {ste_f} & {dqp_f} & {carry_f} & {val_str} & "
            f"{tot_50} & {p_50} & {tot_95} & {yo_50} & {m11_50} & {m13_50} \\\\"
        )

    claim_rows: List[str] = []
    claim_labels = [
        ("C01_total_bdbr_vs_anchor", "C01: Full vs. Anchor (FRCNN Tot AP@50)"),
        ("C02_long_gop_p_frame_rate_reduction", "C02: Temporal MC vs. Spatial (Inter/P-BPP AP@50)"),
        ("C03_ctu_delta_qp_gain", "C03: Halo-Guarded QG dQP vs. Pixel-Only"),
        ("C04_real_codec_ste_gain", "C04: Hybrid Real-Codec STE vs. Proxy-Only"),
        ("C05_zero_shot_unseen_yolov8s_transfer", "C05: Single-Teacher -> Unseen YOLOv8s Tot AP@50"),
    ]
    for cid, clbl in claim_labels:
        ce = claims.get(cid, {})
        d_str = _fmt_pct(ce.get("delta_bdbr_pct"), tex=True)
        ci_str = _fmt_ci(ce.get("ci95_pct"), tex=True)
        st_raw = ce.get("status")
        st_str = f"\\texttt{{{_tex_escape(st_raw)}}}" if st_raw else r"\texttt{TBD}"
        claim_rows.append(f"\\multicolumn{{4}}{{l}}{{{clbl}}} & \\multicolumn{{3}}{{c}}{{$\\Delta = {d_str}$}} & \\multicolumn{{3}}{{c}}{{{ci_str}}} & \\multicolumn{{2}}{{c}}{{{st_str}}} \\\\")

    # 3. Table IV: Zero-Shot Cross-Detector, Cross-Codec (libx264), and Random Access (IBBP)
    regimes = cross.get("regimes", {}) if cross else {}
    hevc_ldp = regimes.get("hevc_x265_low_delay_p", {}).get("backbones", cross.get("backbones", {}) if cross else {})
    avc_ldp = regimes.get("avc_x264_low_delay_p", {}).get("backbones", {})
    hevc_ra = regimes.get("hevc_x265_random_access_ibbp", {}).get("backbones", {})

    t4_rows: List[str] = []
    backbone_order = [
        ("fasterrcnn_r50_fpn_v2", "Faster R-CNN R50-FPN v2", "In-Domain Teacher"),
        ("yolov8s", "YOLOv8s (CSPDarknet)", "Zero-Shot Unseen"),
        ("retinanet_r50_fpn_v2", "RetinaNet R50-FPN v2", "Zero-Shot Unseen"),
        ("fcos_r50_fpn", "FCOS R50-FPN", "Zero-Shot Unseen"),
        ("ssdlite_mobilenet_v3", "SSDLite MobileNetV3", "Zero-Shot Unseen"),
        ("swin_t_transformer", "Swin-T ViT (Feat SNR)", "Zero-Shot Feature"),
    ]
    for b_key, b_name, b_role in backbone_order:
        be = hevc_ldp.get(b_key, {})
        snr_b = _fmt_pct(be.get("tscope_total_bdbr_feat_snr_pct"), tex=True)
        if b_key == "swin_t_transformer":
            sp_50 = "--"
            ts_50 = "--"
            ts_p50 = "--"
            ts_95 = "--"
        else:
            sp_50 = _fmt_pct(be.get("spatial_total_bdbr_gt_ap50_pct"), tex=True)
            ts_50 = _fmt_pct_with_ci(
                be.get("tscope_total_bdbr_gt_ap50_pct"),
                be.get("tscope_total_bdbr_gt_ap50_ci95"),
                tex=True,
                bold=(b_key != "fasterrcnn_r50_fpn_v2"),
            )
            ts_p50 = _fmt_pct(be.get("tscope_p_bdbr_gt_ap50_pct"), tex=True)
            ts_95 = _fmt_pct(be.get("tscope_total_bdbr_gt_ap50_95_pct"), tex=True)
        t4_rows.append(
            f"HEVC LDP (\\texttt{{x265}}, $B{{=}}0$) & {b_name} & {b_role} & {sp_50} & {ts_50} & {ts_p50} & {ts_95} & {snr_b} \\\\"
        )
    t4_rows.append(r"\midrule")
    for reg_lbl, reg_dict in [
        ("AVC LDP (\\texttt{x264}, $f{=}1$)", avc_ldp),
        ("HEVC RA (\\texttt{x265}, $B{=}4$)", hevc_ra),
    ]:
        for b_key, b_name in [
            ("fasterrcnn_r50_fpn_v2", "Faster R-CNN R50-FPN v2"),
            ("yolov8s", "YOLOv8s (Zero-Shot)"),
            ("retinanet_r50_fpn_v2", "RetinaNet (Zero-Shot)"),
        ]:
            be = reg_dict.get(b_key, {})
            sp_50 = _fmt_pct(be.get("spatial_total_bdbr_gt_ap50_pct"), tex=True)
            ts_50 = _fmt_pct_with_ci(
                be.get("tscope_total_bdbr_gt_ap50_pct"),
                be.get("tscope_total_bdbr_gt_ap50_ci95"),
                tex=True,
            )
            ts_p50 = _fmt_pct(be.get("tscope_p_bdbr_gt_ap50_pct"), tex=True)
            ts_95 = _fmt_pct(be.get("tscope_total_bdbr_gt_ap50_95_pct"), tex=True)
            snr_b = _fmt_pct(be.get("tscope_total_bdbr_feat_snr_pct"), tex=True)
            role = "In-Domain" if b_key == "fasterrcnn_r50_fpn_v2" else "Zero-Shot"
            t4_rows.append(
                f"{reg_lbl} & {b_name} & {role} & {sp_50} & {ts_50} & {ts_p50} & {ts_95} & {snr_b} \\\\"
            )

    # 4. Table V: ByteTrack MOTA/IDF1/HOTA Tracking & Secondary VCM Datasets (MOT20, KITTI)
    trk = sota_trk.get("mot17_tracking_metrics", {}) if sota_trk else {}
    sec = sota_trk.get("second_dataset_vcm_summary", {}) if sota_trk else {}
    if not isinstance(sec, dict):
        sec = {}

    t5_rows: List[str] = []
    trk_methods = [
        ("anchor_x265", "Direct \\texttt{libx265} Anchor", False),
        ("mpeg_vcm_spatial_resampling", "MPEG-VCM Spatial Resampling", False),
        ("detector_roi_codec_addroi", "Detector-ROI \\texttt{addroi}", False),
        ("official_miccunifi_scope", "Official SCOPE~\\cite{dani2026scope}", False),
        ("tscope_spatial_only", "T-SCOPE Spatial-Only", False),
        ("tscope_iwa_gop_qp_reuse", "T-SCOPE IWA-Bypass ($K{=}32$)", False),
        ("tscope_full", "\\textbf{T-SCOPE Full (Ours)}", True),
    ]
    for m_key, m_lbl, is_b in trk_methods:
        te = trk.get(m_key, {})
        q27 = te.get("per_qp", {}).get("27", {})
        mota_27 = _fmt_num(q27.get("mota_pct"), digits=2, tex=True)
        hota_27 = _fmt_num(q27.get("hota_pct"), digits=2, tex=True)
        idsw_27 = str(int(q27["idsw"])) if isinstance(q27.get("idsw"), int) else r"\texttt{TBD}"
        b_mota = _fmt_pct(te.get("bdbr_tot_mota_pct"), tex=True)
        b_idf1 = _fmt_pct(te.get("bdbr_tot_idf1_pct"), tex=True)
        b_hota = _fmt_pct(te.get("bdbr_tot_hota_pct"), tex=True)
        m20_e = sec.get("mot20", {}).get("methods", {}).get(m_key, {})
        kit_e = sec.get("kitti", {}).get("methods", {}).get(m_key, {})
        b_m20 = _fmt_pct(m20_e.get("bdbr_tot_map50_pct"), tex=True) if m_key in ("anchor_x265", "tscope_spatial_only", "tscope_iwa_gop_qp_reuse", "tscope_full") else "--"
        b_kit = _fmt_pct(kit_e.get("bdbr_tot_map50_pct"), tex=True) if m_key in ("anchor_x265", "tscope_spatial_only", "tscope_iwa_gop_qp_reuse", "tscope_full") else "--"
        if is_b and b_hota != r"\texttt{TBD}":
            b_hota = f"\\textbf{{{b_hota}}}"
        t5_rows.append(
            f"{m_lbl} & {mota_27} & {hota_27} & {idsw_27} & {b_mota} & {b_idf1} & {b_hota} & {b_m20} & {b_kit} \\\\"
        )

    # 5. Table VI: Complexity, Latency, VRAM & CPU Codec Wall-Clock
    comp = sota_trk.get("complexity_runtime_table", {}) if sota_trk else {}
    np_p = comp.get("neural_pre_editors", {})
    cw_p = comp.get("codec_wall_clock", {})

    def _np_row(key: str, label: str, params_fallback: str, gmacs_fallback: str) -> str:
        e = np_p.get(key, {})
        p_str = _fmt_num(e.get("params_millions"), digits=3, tex=True) if e.get("params_millions") is not None else params_fallback
        g_str = _fmt_num(e.get("gmacs_per_frame"), digits=2, tex=True) if e.get("gmacs_per_frame") is not None else gmacs_fallback
        m_ms = _fmt_num(e.get("mean_ms_per_frame"), digits=2, tex=True)
        p95_ms = _fmt_num(e.get("p95_ms_per_frame"), digits=2, tex=True)
        fps = _fmt_num(e.get("fps"), digits=1, tex=True)
        vram = _fmt_num(e.get("peak_vram_mb"), digits=1, tex=True)
        return f"{label} & {p_str} & {g_str} & {m_ms} & {p95_ms} & {fps} & {vram} \\\\"

    def _cw_row(key: str, label: str) -> str:
        e = cw_p.get(key, {})
        enc = _tex_escape(e.get("encoder", "libx265" if "265" in key else "libx264"))
        ms = _fmt_num(e.get("mean_encode_decode_ms_per_frame"), digits=2, tex=True)
        fps = _fmt_num(e.get("encode_decode_fps"), digits=1, tex=True)
        bpp = _fmt_num(e.get("crf32_total_bpp"), digits=4, tex=True)
        return f"{label} & \\texttt{{{enc}}} & -- & {ms} & -- & {fps} & {bpp} (bpp) \\\\"

    t6_rows = [
        _np_row("cauchy_irls_6dof_motion_estimator_only", "6-DOF Cauchy-IRLS Estimator Only", "0.000", "0.10"),
        _np_row("spatial_only_scope", "T-SCOPE Spatial-Only ($B{=}0$ MC)", "6.109", "107.11"),
        _np_row("iwa_gop_qp_reuse", "\\textbf{T-SCOPE IWA-Bypass ($K{=}32$, $-96.7\\%$ GMACs)}", "6.109", "\\textbf{4.52}"),
        _np_row("full_tscope", "T-SCOPE Full (Per-Frame U-Net + MC)", "6.109", "137.78"),
        _np_row("official_miccunifi_scope", "Official SCOPE~\\cite{dani2026scope} (Framewise)", r"\texttt{TBD}", r"\texttt{TBD}"),
        r"\midrule",
        _cw_row("anchor_libx265_unedited", "CPU Encode+Decode: Unedited \\texttt{libx265} Anchor"),
        _cw_row("tscope_iwa_libx265_static_addroi", "CPU Encode+Decode: T-SCOPE IWA Static \\texttt{addroi}"),
        _cw_row("tscope_full_libx265_dynamic_addroi", "CPU Encode+Decode: T-SCOPE Full Dynamic \\texttt{addroi}"),
        _cw_row("anchor_libx264_unedited", "CPU Encode+Decode: Unedited \\texttt{libx264} Anchor"),
        _cw_row("tscope_full_libx264_dynamic_addroi", "CPU Encode+Decode: T-SCOPE Full \\texttt{libx264} \\texttt{addroi}"),
    ]

    tex_blocks = f"""% AUTO-GENERATED BY scripts/sync_paper_tables.py — DO NOT EDIT MANUALLY.
% Regenerate via: python scripts/sync_paper_tables.py

\\begin{{table*}}[t]
\\centering
\\caption{{Main Rate--Accuracy Comparison on Held-Out \\texttt{{MOT17}} ($640 \\times 384$ 16:9, \\texttt{{yuv420p}}, $\\text{{GOP}}=32$ Low-Delay P, $\\text{{CRF}} \\in \\{{{', '.join(str(q) for q in crfs)}\\}}$). Monotonic PCHIP BDBR (\\%) vs.\\ Direct \\texttt{{libx265}} Anchor (Negative = Bitrate Savings; Bracketed = 10,000-Resample Paired Clip-Block Bootstrap 95\\% CI). \\texttt{{YOLOv8s}} is Strictly Unseen During Single-Teacher Training. $^\\dagger$Non-Deployable Ground-Truth Upper Bound.}}
\\label{{tab:main_mot17_sota}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\footnotesize
\\begin{{tabular}}{{llcccccc}}
\\toprule
\\textbf{{Method}} & \\textbf{{Paradigm}} & \\textbf{{Tot BDBR}} & \\textbf{{Inter/P BDBR}} & \\textbf{{Tot BDBR}} & \\textbf{{Inter/P BDBR}} & \\textbf{{Tot BDBR}} & \\textbf{{Tot BDBR}} \\\\
 & & \\textbf{{FRCNN AP@50}} & \\textbf{{FRCNN AP@50}} & \\textbf{{FRCNN AP@50:95}} & \\textbf{{FRCNN AP@50:95}} & \\textbf{{YOLO AP@50 (ZS)}} & \\textbf{{YOLO AP@50:95 (ZS)}} \\\\
\\midrule
{chr(10).join(t2_rows)}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table*}}

\\begin{{table*}}[t]
\\centering
\\caption{{De-Confounded 5-Variant Ablation Study (Balanced Shards, Shared Seed 2026), Per-Sequence Disaggregation (\\texttt{{MOT17-11}} Moving Pedestrian Street vs.\\ \\texttt{{MOT17-13}} Fast Bus Camera), and Pre-Registered Claim Falsification Audit (\\texttt{{C01}}--\\texttt{{C05}}).}}
\\label{{tab:deconfounded_ablations}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\footnotesize
\\begin{{tabular}}{{lccccc cccc cc}}
\\toprule
\\textbf{{Ablation Variant}} & \\textbf{{MC+GRU}} & \\textbf{{Codec STE}} & \\textbf{{QG dQP}} & \\textbf{{P-Carry}} & \\textbf{{Val Loss}} & \\textbf{{Tot BDBR}} & \\textbf{{P-Frame BDBR}} & \\textbf{{Tot BDBR}} & \\textbf{{YOLO (ZS)}} & \\textbf{{MOT17-11}} & \\textbf{{MOT17-13}} \\\\
 & & & ($16{{\\times}}16$) & & (Unif.) & \\textbf{{FR50}} & \\textbf{{FR50}} & \\textbf{{FR50:95}} & \\textbf{{Tot AP50}} & \\textbf{{Tot FR50}} & \\textbf{{Tot FR50}} \\\\
\\midrule
{chr(10).join(t3_rows)}
\\midrule
\\multicolumn{{12}}{{l}}{{\\textit{{Pre-Registered Quantitative Claim Falsification Audit (Requires Point $\\Delta < 0\\%$ and Upper 95\\% Bootstrap CI $< 0\\%$):}}}} \\\\
{chr(10).join(claim_rows)}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table*}}

\\begin{{table*}}[t]
\\centering
\\caption{{Zero-Shot Cross-Detector Transfer (Single-Teacher Faster R-CNN Checkpoint Evaluated on 4 Unseen Detectors + \\texttt{{Swin-T}} ViT), Cross-Codec \\texttt{{libx264}} (AVC, \\texttt{{family\\_id=1}}), and HEVC Random Access (\\texttt{{IBBP}}, \\texttt{{bframes=4}}) Generalization.}}
\\label{{tab:cross_backbone_codec}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\footnotesize
\\begin{{tabular}}{{lllccccc}}
\\toprule
\\textbf{{Codec \\& GOP Regime}} & \\textbf{{Downstream Backbone}} & \\textbf{{Supervision Role}} & \\textbf{{Spatial-Only}} & \\textbf{{T-SCOPE Full}} & \\textbf{{T-SCOPE}} & \\textbf{{T-SCOPE Full}} & \\textbf{{Feat-SNR}} \\\\
 & & & \\textbf{{Tot AP@50}} & \\textbf{{Tot AP@50 (95\\% CI)}} & \\textbf{{P-BDBR AP@50}} & \\textbf{{Tot AP@50:95}} & \\textbf{{Tot BDBR}} \\\\
\\midrule
{chr(10).join(t4_rows)}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table*}}

\\begin{{table*}}[t]
\\centering
\\caption{{Multi-Object Tracking Performance (\\texttt{{ByteTrack}} Kalman--IoU Association on Contiguous 32-Frame GOPs: \\texttt{{MOTA}}, \\texttt{{IDF1}}, 19-Threshold \\texttt{{HOTA}}, \\texttt{{IDSW}}, and Tracking BDBR on Held-Out \\texttt{{MOT17}}) and Zero-Shot Secondary VCM Dataset Generalization (\\texttt{{MOT20}} \\& Multi-Class \\texttt{{KITTI}}).}}
\\label{{tab:tracking_and_second_dataset}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\footnotesize
\\begin{{tabular}}{{lccc ccc cc}}
\\toprule
\\textbf{{Method}} & \\textbf{{CRF 27}} & \\textbf{{CRF 27}} & \\textbf{{CRF 27}} & \\textbf{{Tot BDBR}} & \\textbf{{Tot BDBR}} & \\textbf{{Tot BDBR}} & \\textbf{{MOT20 (ZS)}} & \\textbf{{KITTI Multi-Cls}} \\\\
 & \\textbf{{MOTA (\\%)}} & \\textbf{{HOTA (\\%)}} & \\textbf{{IDSW $\\downarrow$}} & \\textbf{{MOTA}} & \\textbf{{IDF1}} & \\textbf{{HOTA}} & \\textbf{{Tot mAP@50}} & \\textbf{{Tot mAP@50}} \\\\
\\midrule
{chr(10).join(t5_rows)}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table*}}

\\begin{{table*}}[t]
\\centering
\\caption{{Computational Complexity, Neural Pre-Processing Latency (Mean / P95 ms per Frame, FPS, Peak GPU VRAM, Analytical GMACs at $640 \\times 384$, $T=32$), and Measured CPU \\texttt{{libx265}}/\\texttt{{libx264}} Wall-Clock Encoding+Decoding Time at CRF 32.}}
\\label{{tab:complexity_runtime}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\footnotesize
\\begin{{tabular}}{{lcccccc}}
\\toprule
\\textbf{{Component / Operating Mode}} & \\textbf{{Params (M)}} & \\textbf{{GMACs / Frame}} & \\textbf{{Mean (ms/f)}} & \\textbf{{P95 (ms/f)}} & \\textbf{{Throughput (FPS)}} & \\textbf{{Peak VRAM (MB) / Rate}} \\\\
\\midrule
{chr(10).join(t6_rows)}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table*}}
"""
    return tex_blocks


def render_markdown_benchmark_report(
    merged: Optional[Dict[str, Any]],
    cross: Optional[Dict[str, Any]],
    sota_trk: Optional[Dict[str, Any]],
    merged_path: Optional[Path],
    cross_path: Optional[Path],
    sota_paths: Sequence[Path],
    soup_conn: Optional[Dict[str, Any]] = None,
    soup_path: Optional[Path] = None,
) -> str:
    """Renders docs/V3_Q1_BENCHMARK_REPORT.md deterministically from the v3 JSON bundles."""
    crfs = merged.get("crfs", CRFS_DEFAULT) if merged else CRFS_DEFAULT
    elig = merged.get("eligibility", {}) if merged else {}
    ov = merged.get("overall_methods_summary", {}) if merged else {}
    ab = merged.get("ablation_variants_summary", {}) if merged else {}
    pseq = merged.get("per_sequence_summary", {}) if merged else {}
    audit = merged.get("claim_falsification_audit", {}) if merged else {}
    sota_m = sota_trk.get("external_sota_baselines", {}).get("methods", {}) if sota_trk else {}
    sota_sha_str = (
        ", ".join(f"{p.name}:{_sha256_full(p)}" for p in sota_paths)
        if sota_paths
        else "pending"
    )

    md: List[str] = [
        "# T-SCOPE v3 Q1 Benchmark Report (Single-Source-of-Truth Auto-Generated)",
        "",
        "> **Zero-Drift Provenance Notice (`F09` Fix):** This file is deterministically generated by `scripts/sync_paper_tables.py` directly from checked `schema_version = 3` JSON artifacts. Never edit numerical cells by hand.",
        "",
        "## 1. Cryptographic Artifact Provenance & Protocol Summary",
        "",
        f"- **`merged_benchmarks.json` SHA-256:** `{_sha256_full(merged_path)}`",
        f"- **`source_manifest_sha256`:** `{merged.get('source_manifest_sha256', 'pending') if merged else 'pending'}`",
        f"- **`cross_backbone_bdbr.json` SHA-256:** `{_sha256_full(cross_path)}`",
        f"- **`sota_and_tracking` JSON SHA-256(s):** `{sota_sha_str}`",
        f"- **`soup_connectivity_report.json` SHA-256:** `{_sha256_full(soup_path)}`",
        f"- **Eligibility:** `final_ready = {elig.get('final_ready', False)}`, `expected_shards = {elig.get('expected_shards', 'TBD')}`, `complete_shard_variants = {elig.get('complete_shard_variants', 'TBD')}`, `frame_count = {elig.get('frame_count', 1632)}`, `clip_count = {elig.get('clip_count', 51)}`",
        f"- **Codec & CTC Protocol:** `640x384` (16:9 aspect-preserving), `-pix_fmt yuv420p` (Main 4:2:0), `gop_len = 32` (`bframes = 0` LDP & `bframes = 4` RA), `CRFS = {crfs}`, native `16x16` px QGs (`40x24` grid), `PchipInterpolator` BDBR + `10,000`-resample paired clip-block bootstrap 95% CIs.",
        "- **Single-Teacher Training:** Frozen `Faster R-CNN ResNet50-FPN v2` only (`use_yolov8s = False`); `YOLOv8s`, `RetinaNet`, `FCOS`, `SSDLite`, and `Swin-T` are strictly unseen zero-shot evaluators.",
        "",
        "## 2. Held-Out MOT17 Overall & External SOTA Comparison (`1,632` Frames, `GOP=32`)",
        "",
        "| Method | Tot BDBR (`FRCNN AP@50` + 95% CI) | Inter/P BDBR (`FRCNN AP@50`) | Tot BDBR (`FRCNN AP@50:95`) | Tot BDBR (`YOLO AP@50` Zero-Shot) | Total BPP (`CRF 27 / 32 / 37 / 42`) | Inter/P BPP (`CRF 27 / 32 / 37 / 42`) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    rows_spec = [
        ("`anchor_x265`", ov.get("anchor_x265"), sota_m.get("anchor_x265"), False),
        ("`deterministic_blur_ema`", ov.get("deterministic_blur_ema"), None, False),
        ("`mpeg_vcm_spatial_resampling`", sota_m.get("mpeg_vcm_spatial_resampling"), None, False),
        ("`detector_roi_codec_addroi`", sota_m.get("detector_roi_codec_addroi"), None, False),
        ("`compressed_video_finetuned_frcnn`", sota_m.get("compressed_video_finetuned_frcnn"), None, False),
        ("`official_miccunifi_scope`", sota_m.get("official_miccunifi_scope"), None, False),
        ("`spatial_only_scope`", ab.get("spatial_only_scope"), sota_m.get("tscope_spatial_only"), False),
        ("`iwa_gop_qp_reuse`", ab.get("iwa_gop_qp_reuse"), sota_m.get("tscope_iwa_gop_qp_reuse"), False),
        ("**`full_tscope` (Ours)**", ab.get("full_tscope"), sota_m.get("tscope_full"), True),
        ("`oracle_gt_mask_upper_bound` (GT)", ov.get("oracle_gt_mask_upper_bound"), None, False),
    ]
    anc_md_pqp = (ov.get("anchor_x265") or sota_m.get("anchor_x265") or {}).get("per_qp")
    for lbl, pri, fb, bold in rows_spec:
        src = pri if (pri and pri.get("per_qp")) else (fb or {})
        pqp = src.get("per_qp")
        if src.get("bdbr_tot_yolo_gt_ap50_pct") is not None:
            yo_md = _fmt_pct(src.get("bdbr_tot_yolo_gt_ap50_pct"), tex=False)
        elif isinstance(pqp, dict) and pqp and all("yolo_gt_ap50_pct" not in (pqp.get(str(q)) or {}) for q in crfs):
            yo_md = "`--`"
        else:
            d_yo = _pchip_bdbr_from_per_qp(anc_md_pqp, pqp, "total_bpp", "yolo_gt_ap50_pct", crfs)
            yo_md = _fmt_pct(d_yo, tex=False)
        md.append(
            f"| {lbl} | "
            f"{_fmt_pct_with_ci(src.get('bdbr_tot_gt_ap50_pct'), src.get('bdbr_tot_gt_ap50_pct_ci95', src.get('bdbr_tot_gt_ap50_ci95')), tex=False, bold=bold)} | "
            f"{_fmt_pct(src.get('bdbr_p_gt_ap50_pct'), tex=False)} | "
            f"{_fmt_pct(src.get('bdbr_tot_gt_ap50_95_pct'), tex=False)} | "
            f"{yo_md} | "
            f"{_fmt_curve(pqp, 'total_bpp', crfs, digits=4, tex=False)} | "
            f"{_fmt_curve(pqp, 'p_frame_bpp', crfs, digits=4, tex=False)} |"
        )

    md.extend(
        [
            "",
            "## 3. De-Confounded 5-Variant Ablation & Per-Sequence Disaggregation (`F01`, `F07`, `F09`, `F21`)",
            "",
            "| Variant | Tot BDBR (`FR50`) | Inter/P BDBR (`FR50`) | Tot BDBR (`FR50:95`) | Inter/P BDBR (`FR50:95`) | Zero-Shot `YOLO AP50` | `MOT17-11` (`896`f) Tot `FR50` | `MOT17-13` (`736`f) Tot `FR50` |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for v_key in ("full_tscope", "iwa_gop_qp_reuse", "wo_ctu_delta_qp", "spatial_only_scope", "wo_x265_ste_proxy"):
        ve = ab.get(v_key, {})
        s11 = pseq.get("MOT17-11", {}).get("ablation_variants_summary", {}).get(v_key, {})
        s13 = pseq.get("MOT17-13", {}).get("ablation_variants_summary", {}).get(v_key, {})
        md.append(
            f"| `{v_key}` | "
            f"{_fmt_pct_with_ci(ve.get('bdbr_tot_gt_ap50_pct'), ve.get('bdbr_tot_gt_ap50_pct_ci95'), tex=False, bold=(v_key == 'full_tscope'))} | "
            f"{_fmt_pct(ve.get('bdbr_p_gt_ap50_pct'), tex=False)} | "
            f"{_fmt_pct(ve.get('bdbr_tot_gt_ap50_95_pct'), tex=False)} | "
            f"{_fmt_pct(ve.get('bdbr_p_gt_ap50_95_pct'), tex=False)} | "
            f"{_fmt_pct(ve.get('bdbr_tot_yolo_gt_ap50_pct'), tex=False)} | "
            f"{_fmt_pct(s11.get('bdbr_tot_gt_ap50_pct'), tex=False)} | "
            f"{_fmt_pct(s13.get('bdbr_tot_gt_ap50_pct'), tex=False)} |"
        )

    conn_vars = soup_conn.get("variants", {}) if isinstance(soup_conn, dict) else {}
    if conn_vars:
        md.extend(
            [
                "",
                "### 3.1 MOT17-09 Model Soup Connectivity & Uniform Real-`libx265` Validation Audit (`F15`, `F20`)",
                "",
                "| Variant | Uniform Soup Loss | Constituent Shards (`Mean ± Std`) | Greedy Soup Loss | Linear Mode Connected | Selected Mode | Native Preset Loss |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for v_key in ("full_tscope", "iwa_gop_qp_reuse", "wo_ctu_delta_qp", "spatial_only_scope", "wo_x265_ste_proxy"):
            vc = conn_vars.get(v_key, {})
            u_l = vc.get("uniform_soup_loss_total", vc.get("uniform_loss_total"))
            c_m = vc.get("constituent_uniform_loss_mean")
            c_s = vc.get("constituent_uniform_loss_std")
            g_l = vc.get("greedy_soup_loss_total")
            lmc = vc.get("linear_mode_connected")
            sel = vc.get("selected_mode", "--")
            nat = vc.get("native_preset_loss_total")
            c_str = (
                f"`{_fmt_num(c_m, digits=4, tex=False)} ± {_fmt_num(c_s, digits=4, tex=False)}`"
                if (c_m is not None and c_s is not None)
                else "`--`"
            )
            u_str = f"`{_fmt_num(u_l, digits=4, tex=False)}`" if u_l is not None else "`--`"
            g_str = f"`{_fmt_num(g_l, digits=4, tex=False)}`" if g_l is not None else "`--`"
            n_str = f"`{_fmt_num(nat, digits=4, tex=False)}`" if nat is not None else "`--`"
            lmc_str = f"`{lmc}`" if lmc is not None else "`--`"
            md.append(f"| `{v_key}` | {u_str} | {c_str} | {g_str} | {lmc_str} | `{sel}` | {n_str} |")

    md.extend(
        [
            "",
            "## 4. Pre-Registered Claim Falsification Audit (`C01..C05`, `F18`)",
            "",
            "| Claim ID | Variant vs. Comparator | Metric | Point Delta | 95% Bootstrap CI | Status |",
            "| :--- | :--- | :--- | :---: | :---: | :---: |",
        ]
    )
    for cid, ce in (audit.get("claims", {}) or {}).items():
        md.append(
            f"| `{cid}` | `{ce.get('variant')}` vs. `{ce.get('comparator')}` | `{ce.get('metric')}` | "
            f"{_fmt_pct(ce.get('delta_bdbr_pct'), tex=False)} | {_fmt_ci(ce.get('ci95_pct'), tex=False)} | `{ce.get('status', 'TBD')}` |"
        )

    md.extend(
        [
            "",
            "## 5. Zero-Shot Cross-Detector, Cross-Codec (`libx264`) & Random-Access (`IBBP`) Generalization",
            "",
            "| Regime | Backbone | Zero-Shot | Spatial-Only Tot `AP@50` | T-SCOPE Full Tot `AP@50` (95% CI) | T-SCOPE Inter/P `AP@50` | Feat-SNR Tot BDBR |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    regimes = cross.get("regimes", {}) if cross else {}
    for reg_key, reg_obj in regimes.items():
        for b_key, be in (reg_obj.get("backbones", {}) or {}).items():
            is_det = bool(be.get("is_object_detector", b_key != "swin_t_transformer"))
            sp_c = _fmt_pct(be.get("spatial_total_bdbr_gt_ap50_pct"), tex=False) if is_det else "`--`"
            ts_c = (
                _fmt_pct_with_ci(be.get("tscope_total_bdbr_gt_ap50_pct"), be.get("tscope_total_bdbr_gt_ap50_ci95"), tex=False)
                if is_det
                else "`--`"
            )
            tp_c = _fmt_pct(be.get("tscope_p_bdbr_gt_ap50_pct"), tex=False) if is_det else "`--`"
            md.append(
                f"| `{reg_key}` | {be.get('label', b_key)} | `{be.get('zero_shot_unseen_in_training')}` | "
                f"{sp_c} | {ts_c} | {tp_c} | "
                f"{_fmt_pct(be.get('tscope_total_bdbr_feat_snr_pct'), tex=False)} |"
            )

    md.extend(
        [
            "",
            "## 6. ByteTrack Multi-Object Tracking & Complexity Summary (`F11`, `F16`, `F17`, `F24`)",
            "",
            "| Method | Tracking `MOTA` BDBR | Tracking `IDF1` BDBR | Tracking `HOTA` BDBR |",
            "| :--- | :---: | :---: | :---: |",
        ]
    )
    for m_key, te in ((sota_trk.get("mot17_tracking_metrics", {}) if sota_trk else {}) or {}).items():
        md.append(
            f"| `{m_key}` | {_fmt_pct(te.get('bdbr_tot_mota_pct'), tex=False)} | "
            f"{_fmt_pct(te.get('bdbr_tot_idf1_pct'), tex=False)} | {_fmt_pct(te.get('bdbr_tot_hota_pct'), tex=False)} |"
        )

    comp = sota_trk.get("complexity_runtime_table", {}) if sota_trk else {}
    if comp:
        md.extend(
            [
                "",
                f"### Complexity & Codec Wall-Clock (`{comp.get('gpu_name', 'GPU')}`, `{comp.get('resolution', '640x384')}`, `GOP={comp.get('gop_len', 32)}`)",
                "",
                "| Mode | Params (M) | GMACs/frame | Mean ms/frame | P95 ms/frame | FPS | Peak VRAM (MB) |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for k, e in (comp.get("neural_pre_editors", {}) or {}).items():
            md.append(
                f"| `{k}` | `{e.get('params_millions', '--')}` | `{e.get('gmacs_per_frame', '--')}` | "
                f"`{e.get('mean_ms_per_frame', '--')}` | `{e.get('p95_ms_per_frame', '--')}` | "
                f"`{e.get('fps', '--')}` | `{e.get('peak_vram_mb', '--')}` |"
            )
    return "\n".join(md) + "\n"


def update_readme_benchmark_section(readme_path: Path, report_snippet: str) -> None:
    """Atomically updates the benchmark section of README.md via markers or `## 5.` heading."""
    block = f"{BEGIN_README_MARKER}\n{report_snippet.strip()}\n{END_README_MARKER}\n"
    if not readme_path.is_file():
        readme_path.write_text(block, encoding="utf-8")
        return
    content = readme_path.read_text(encoding="utf-8")
    if BEGIN_README_MARKER in content and END_README_MARKER in content:
        pattern = re.compile(
            re.escape(BEGIN_README_MARKER) + r".*?" + re.escape(END_README_MARKER) + r"\n?",
            flags=re.DOTALL,
        )
        updated = pattern.sub(lambda _: block, content, count=1)
    else:
        m = re.search(r"^##\s+5\.", content, flags=re.MULTILINE)
        if m:
            updated = content[: m.start()].rstrip() + "\n\n" + block
        else:
            updated = content.rstrip() + "\n\n---\n\n" + block
    readme_path.write_text(updated, encoding="utf-8")


def build_readme_benchmark_snippet(
    merged: Optional[Dict[str, Any]],
    cross: Optional[Dict[str, Any]],
    sota_trk: Optional[Dict[str, Any]],
) -> str:
    crfs = merged.get("crfs", CRFS_DEFAULT) if merged else CRFS_DEFAULT
    ab = merged.get("ablation_variants_summary", {}) if merged else {}
    ov = merged.get("overall_methods_summary", {}) if merged else {}
    m_sha = merged.get("source_manifest_sha256", "pending") if merged else "pending"
    lines = [
        "## 5. Verified T-SCOPE v3 Q1 Benchmark Summary (Auto-Synced via `scripts/sync_paper_tables.py`)",
        "",
        f"Source manifest SHA-256: `{m_sha}` (`schema_version = 3`, `640x384` 16:9, `yuv420p`, `GOP = 32`, `CRFS = {crfs}`, `PchipInterpolator` + `10,000` paired clip-bootstrap 95% CIs). Full disaggregation in [`docs/V3_Q1_BENCHMARK_REPORT.md`](./docs/V3_Q1_BENCHMARK_REPORT.md).",
        "",
        "| Variant / Method | Total BDBR (`FRCNN AP@50` + 95% CI) | Inter/P-Frame BDBR (`FRCNN AP@50`) | Total BDBR (`FRCNN AP@50:95`) | Zero-Shot `YOLOv8s AP@50` BDBR |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    rows = [
        ("`anchor_x265`", ov.get("anchor_x265", {}), False),
        ("`deterministic_blur_ema`", ov.get("deterministic_blur_ema", {}), False),
        ("`spatial_only_scope`", ab.get("spatial_only_scope", {}), False),
        ("`wo_x265_ste_proxy`", ab.get("wo_x265_ste_proxy", {}), False),
        ("`wo_ctu_delta_qp`", ab.get("wo_ctu_delta_qp", {}), False),
        ("`iwa_gop_qp_reuse` (`-96.7%` GMACs)", ab.get("iwa_gop_qp_reuse", {}), False),
        ("**`full_tscope` (Ours)**", ab.get("full_tscope", {}), True),
        ("`oracle_gt_mask_upper_bound` (GT)", ov.get("oracle_gt_mask_upper_bound", {}), False),
    ]
    for lbl, src, bold in rows:
        lines.append(
            f"| {lbl} | "
            f"{_fmt_pct_with_ci(src.get('bdbr_tot_gt_ap50_pct'), src.get('bdbr_tot_gt_ap50_pct_ci95'), tex=False, bold=bold)} | "
            f"{_fmt_pct(src.get('bdbr_p_gt_ap50_pct'), tex=False)} | "
            f"{_fmt_pct(src.get('bdbr_tot_gt_ap50_95_pct'), tex=False)} | "
            f"{_fmt_pct(src.get('bdbr_tot_yolo_gt_ap50_pct'), tex=False)} |"
        )
    if cross:
        avc_b = (
            cross.get("regimes", {})
            .get("avc_x264_low_delay_p", {})
            .get("backbones", {})
            .get("fasterrcnn_r50_fpn_v2", {})
            .get("tscope_total_bdbr_gt_ap50_pct")
        )
        ra_b = (
            cross.get("regimes", {})
            .get("hevc_x265_random_access_ibbp", {})
            .get("backbones", {})
            .get("fasterrcnn_r50_fpn_v2", {})
            .get("tscope_total_bdbr_gt_ap50_pct")
        )
        lines.append("")
        lines.append(
            f"- **Cross-Codec & Random Access:** `libx264` (`family_id=1`) FRCNN AP@50 BDBR = {_fmt_pct(avc_b, tex=False)} | HEVC Random Access (`IBBP`, `bframes=4`) FRCNN AP@50 BDBR = {_fmt_pct(ra_b, tex=False)}."
        )
    if sota_trk:
        trk_f = sota_trk.get("mot17_tracking_metrics", {}).get("tscope_full", {})
        lines.append(
            f"- **ByteTrack Multi-Object Tracking BDBR:** `MOTA` = {_fmt_pct(trk_f.get('bdbr_tot_mota_pct'), tex=False)}, `IDF1` = {_fmt_pct(trk_f.get('bdbr_tot_idf1_pct'), tex=False)}, `HOTA` = {_fmt_pct(trk_f.get('bdbr_tot_hota_pct'), tex=False)}."
        )
    return "\n".join(lines) + "\n"


def sync_all(
    pool_dir: Path,
    paper_dir: Path,
    docs_report_path: Optional[Path],
    readme_path: Optional[Path],
    merged_json: Optional[str] = None,
    cross_json: Optional[str] = None,
    sota_json: Optional[Union[str, Sequence[str]]] = None,
    soup_conn_json: Optional[str] = None,
    init_placeholders: bool = False,
) -> Dict[str, str]:
    if init_placeholders:
        merged_p = cross_p = soup_p = None
        sota_paths: List[Path] = []
        merged = cross = sota_trk = soup_conn = None
        prov = "placeholder initialization (--init-placeholders; awaiting Kaggle v3 evaluation JSONs)"
    else:
        merged_p = _resolve_optional_json(
            merged_json,
            [pool_dir / "merged_benchmarks.json", pool_dir / "eval_v3_final" / "merged_benchmarks.json"],
        )
        if merged_p is None:
            raise FileNotFoundError(
                f"Missing required v3 merged_benchmarks.json under {pool_dir} (use --init-placeholders before eval lands)"
            )
        cross_p = _resolve_optional_json(
            cross_json,
            [pool_dir / "cross_backbone_bdbr.json", pool_dir / "eval_cross_backbone" / "cross_backbone_bdbr.json"],
        )
        sota_paths = _discover_sota_json_paths(sota_json, pool_dir)
        soup_p = _resolve_optional_json(
            soup_conn_json,
            [pool_dir / "soup_connectivity_report.json", pool_dir / "validate_soups" / "soup_connectivity_report.json"],
        )
        merged = _load_checked_json(merged_p, require_final_ready=True)
        cross = _load_checked_json(cross_p, require_final_ready=False)
        sota_trk = _load_and_merge_sota_jsons(sota_paths)
        soup_conn = json.loads(soup_p.read_text(encoding="utf-8")) if (soup_p and soup_p.is_file()) else None
        sota_prov = "+".join(_sha256_short(p) for p in sota_paths) if sota_paths else "pending"
        prov = (
            f"merged={_sha256_short(merged_p)}, "
            f"cross={_sha256_short(cross_p)}, "
            f"sota_trk={sota_prov}, "
            f"soup={_sha256_short(soup_p)}"
        )

    macros = build_macros_dict(merged, cross, sota_trk)
    macros_tex = render_metrics_macros_tex(macros, prov)
    tables_tex = render_tables_generated_tex(merged, cross, sota_trk, soup_conn=soup_conn)

    paper_dir.mkdir(parents=True, exist_ok=True)
    macros_out = paper_dir / "metrics_macros.tex"
    tables_out = paper_dir / "tables_generated.tex"
    macros_out.write_text(macros_tex, encoding="utf-8")
    tables_out.write_text(tables_tex, encoding="utf-8")
    written = {"metrics_macros_tex": str(macros_out), "tables_generated_tex": str(tables_out)}

    if not init_placeholders:
        if docs_report_path is not None:
            docs_report_path.parent.mkdir(parents=True, exist_ok=True)
            report_md = render_markdown_benchmark_report(
                merged,
                cross,
                sota_trk,
                merged_p,
                cross_p,
                sota_paths,
                soup_conn=soup_conn,
                soup_path=soup_p,
            )
            docs_report_path.write_text(report_md, encoding="utf-8")
            written["docs_report_md"] = str(docs_report_path)
        if readme_path is not None:
            snippet = build_readme_benchmark_snippet(merged, cross, sota_trk)
            update_readme_benchmark_section(readme_path, snippet)
            written["readme_md"] = str(readme_path)

    return written


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Synchronize LaTeX macros/tables, docs/V3_Q1_BENCHMARK_REPORT.md, and README.md from v3 JSONs."
    )
    parser.add_argument("--pool-dir", "--pool_dir", dest="pool_dir", type=str, default=str(repo_root / "outputs_kaggle_pool"))
    parser.add_argument("--merged-json", "--merged_json", dest="merged_json", type=str, default=None)
    parser.add_argument("--cross-backbone-json", "--cross_backbone_json", dest="cross_json", type=str, default=None)
    parser.add_argument(
        "--sota-tracking-json",
        "--sota_tracking_json",
        dest="sota_json",
        type=str,
        nargs="*",
        default=None,
        help="One or more sota_and_tracking*.json paths to deep-merge (auto-discovers across --pool-dir if omitted).",
    )
    parser.add_argument("--soup-connectivity-json", "--soup_connectivity_json", dest="soup_conn_json", type=str, default=None)
    parser.add_argument("--paper-dir", "--paper_dir", dest="paper_dir", type=str, default=str(repo_root / "paper"))
    parser.add_argument(
        "--docs-report",
        "--docs_report",
        dest="docs_report",
        type=str,
        default=str(repo_root / "docs" / "V3_Q1_BENCHMARK_REPORT.md"),
    )
    parser.add_argument("--readme", type=str, default=str(repo_root / "README.md"))
    parser.add_argument("--skip-docs", action="store_true", default=False)
    parser.add_argument("--skip-readme", action="store_true", default=False)
    parser.add_argument(
        "--init-placeholders",
        "--init_placeholders",
        dest="init_placeholders",
        action="store_true",
        default=False,
        help="Write clean TBD placeholder macros and tables into --paper-dir without touching README/docs.",
    )
    args = parser.parse_args()

    written = sync_all(
        pool_dir=Path(args.pool_dir),
        paper_dir=Path(args.paper_dir),
        docs_report_path=None if args.skip_docs else Path(args.docs_report),
        readme_path=None if args.skip_readme else Path(args.readme),
        merged_json=args.merged_json,
        cross_json=args.cross_json,
        sota_json=args.sota_json,
        soup_conn_json=args.soup_conn_json,
        init_placeholders=args.init_placeholders,
    )
    for key, path_str in written.items():
        print(f"[sync_paper_tables] Wrote {key}: {path_str}")


if __name__ == "__main__":
    main()
