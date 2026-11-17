"""Input and rendering contract tests for T-SCOPE v3 plots (schema_version=3, P-frame BPP, 95% CIs)."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import gen_fig_bdbr_curves as curves


def method_fixture():
    method = {
        "per_qp": {
            str(q): {
                "total_bpp": 0.04 + (42 - q) * 0.012,
                "p_frame_bpp": 0.02 + (42 - q) * 0.008,
                "roi_psnr_db": 26.0 + (42 - q) * 0.35,
                "gt_ap50_pct": 42.0 + (42 - q) * 0.9,
                "gt_ap50_95_pct": 22.0 + (42 - q) * 0.5,
            }
            for q in curves.EXPECTED_CRFS
        },
        "bdbr_unavailable_reasons": {},
    }
    for key in curves.BDBR_FIELDS.values():
        method[key] = -14.5
        method[f"{key}_ci95"] = [-18.2, -10.8]
    return method


def bundle_fixture():
    codec_cfg = dict(curves.PRESERVED_CODEC_SETTINGS, actual_roi_grid=[24, 40])
    return {
        "schema_version": 3,
        "eligibility": {"final_ready": True},
        "crfs": list(curves.EXPECTED_CRFS),
        "configuration": {"codec": codec_cfg},
        "overall_methods_summary": {key: method_fixture() for key in curves.METHOD_LABELS},
        "ablation_variants_summary": {key: method_fixture() for key in curves.VARIANT_LABELS},
        "per_sequence_summary": {
            "MOT17-11": {
                "frame_count": 896,
                "ablation_variants_summary": {key: method_fixture() for key in curves.VARIANT_LABELS},
            },
            "MOT17-13": {
                "frame_count": 736,
                "ablation_variants_summary": {key: method_fixture() for key in curves.VARIANT_LABELS},
            },
        },
    }


class BenchmarkPlotV3Tests(unittest.TestCase):
    def test_schema_v3_complete_protocol_is_valid(self):
        bundle = bundle_fixture()
        self.assertIs(curves.validate_bundle(bundle), bundle)
        extracted = curves.extract_curve(bundle["overall_methods_summary"], "anchor_x265", bundle["crfs"])
        self.assertEqual(len(extracted["total_bpp"]), 4)
        self.assertEqual(len(extracted["p_bpp"]), 4)
        self.assertEqual(extracted["bdbr_ci95"]["bdbr_tot_gt50"], (-18.2, -10.8))
        self.assertIn("[-18.2, -10.8]", curves.bdbr_text(extracted, "bdbr_tot_gt50"))

    def test_old_or_unspecified_schema_fails(self):
        for version in (None, 1, 2, "3", True, 3.0):
            with self.subTest(version=version):
                bundle = bundle_fixture()
                bundle["schema_version"] = version
                with self.assertRaisesRegex(curves.PlotInputError, "schema_version"):
                    curves.validate_bundle(bundle)

    def test_actual_v3_crfs_enforced(self):
        for bad_crfs in (None, [22, 27, 32, 37], [27, 32, 37, 42, 47], [42, 37, 32, 27], ["27", "32", "37", "42"]):
            with self.subTest(crfs=bad_crfs):
                bundle = bundle_fixture()
                bundle["crfs"] = bad_crfs
                with self.assertRaisesRegex(curves.PlotInputError, "crfs"):
                    curves.validate_bundle(bundle)

    def test_missing_deconfounded_variant_fails(self):
        bundle = bundle_fixture()
        del bundle["ablation_variants_summary"]["wo_x265_ste_proxy"]
        with self.assertRaisesRegex(curves.PlotInputError, "missing method.*wo_x265_ste_proxy"):
            curves.validate_bundle(bundle)

    def test_invalid_ci95_bounds_rejected(self):
        bundle = bundle_fixture()
        bundle["ablation_variants_summary"]["full_tscope"]["bdbr_tot_gt_ap50_pct_ci95"] = [-5.0, -12.0]
        with self.assertRaisesRegex(curves.PlotInputError, "lower bound exceeds upper bound"):
            curves.validate_bundle(bundle)

    def test_null_bdbr_is_annotated_not_fabricated(self):
        bundle = bundle_fixture()
        method = bundle["ablation_variants_summary"]["full_tscope"]
        for field in curves.BDBR_FIELDS.values():
            method[field] = None
            method["bdbr_unavailable_reasons"][field] = "no common AP interval"
        curves.validate_bundle(bundle)
        extracted = curves.extract_curve(bundle["ablation_variants_summary"], "full_tscope", bundle["crfs"])
        for field in curves.BDBR_FIELDS:
            self.assertIsNone(extracted[field])
            self.assertEqual(curves.bdbr_text(extracted, field), "BDBR unavailable: no common AP interval")

    def test_codec_caption_reflects_yuv420p_gop32_and_40x24_grid(self):
        caption = curves.codec_caption(bundle_fixture())
        self.assertIn("yuv420p", caption)
        self.assertIn("GOP=32", caption)
        self.assertIn("40x24 addroi grid", caption)

    def test_validate_only_loads_json_without_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            output = Path(directory) / "never-render"
            path.write_text(json.dumps(bundle_fixture(), allow_nan=False), encoding="utf-8")
            with patch.object(curves, "render_figures") as render, patch("builtins.print"):
                curves.main(["--benchmark", str(path), "--output-dir", str(output), "--validate-only"])
                render.assert_not_called()
            self.assertFalse(output.exists())

    def test_render_figures_emits_total_pframe_and_per_seq_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "figs"
            curves.render_figures(bundle_fixture(), out)
            self.assertTrue((out / "fig_bdbr_curves.png").is_file())
            self.assertTrue((out / "fig_ablation_bdbr.png").is_file())
            self.assertTrue((out / "fig_per_sequence_bdbr.png").is_file())


if __name__ == "__main__":
    unittest.main()
