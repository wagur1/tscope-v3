import argparse
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.experiment_protocol import (
    CRFS,
    EVAL_GOP_LEN,
    EXCLUDED_TEST_TAIL_FRAME_IDS,
    EXPECTED_TEST_FRAME_IDS,
    IMG_H,
    IMG_W,
    SCHEMA_VERSION,
    VARIANT_CONFIGS,
    VARIANTS,
)
from scripts.kaggle_shard_runner import run_poc_wishlist2_and_eval_shard, run_train_shard
from scripts.validate_soups import validate
from video_codecs.hybrid_x265_ste import HybridX265STE


class EvaluationContractTests(unittest.TestCase):
    def test_schema_v3_and_long_gop_16_9_protocol(self):
        self.assertEqual(SCHEMA_VERSION, 3)
        self.assertEqual(CRFS, [27, 32, 37, 42])
        self.assertEqual((IMG_H, IMG_W), (384, 640))
        self.assertEqual(EVAL_GOP_LEN, 32)
        # 28*32 (896) from MOT17-11 + 23*32 (736) from MOT17-13 = 1,632 complete-GOP frames
        self.assertEqual(len(EXPECTED_TEST_FRAME_IDS), 1632)
        self.assertEqual(len(set(EXPECTED_TEST_FRAME_IDS)), 1632)
        self.assertNotIn("MOT17-11/000897", EXPECTED_TEST_FRAME_IDS)
        self.assertNotIn("MOT17-13/000737", EXPECTED_TEST_FRAME_IDS)
        self.assertIn("MOT17-11/000897", EXCLUDED_TEST_TAIL_FRAME_IDS)
        self.assertIn("MOT17-13/000750", EXCLUDED_TEST_TAIL_FRAME_IDS)

    def test_deconfounded_5_variants_contract(self):
        self.assertEqual(
            VARIANTS,
            [
                "full_tscope",
                "wo_ctu_delta_qp",
                "spatial_only_scope",
                "iwa_gop_qp_reuse",
                "wo_x265_ste_proxy",
            ],
        )
        # F07 fix: spatial_only_scope uses fair real-x265 STE
        self.assertTrue(VARIANT_CONFIGS["spatial_only_scope"]["use_real_x265_ste"])
        self.assertTrue(VARIANT_CONFIGS["spatial_only_scope"]["disable_temporal"])
        # F07 fix: wo_x265_ste_proxy matches full_tscope identically except use_real_x265_ste=False
        for key in ("lambda_temp_bg", "lambda_temp_obj", "max_delta_qp", "reuse_iframe_qp", "disable_temporal"):
            self.assertEqual(VARIANT_CONFIGS["wo_x265_ste_proxy"][key], VARIANT_CONFIGS["full_tscope"][key])
        self.assertFalse(VARIANT_CONFIGS["wo_x265_ste_proxy"]["use_real_x265_ste"])

    def test_train_rejects_yolov8s_teacher_leakage(self):
        args = argparse.Namespace(use_yolov8s=True, shard_id=0, num_shards=5, variant="full_tscope")
        with self.assertRaisesRegex(ValueError, "F04"):
            run_train_shard(args)

    def test_final_evaluation_rejects_synthetic(self):
        args = argparse.Namespace(use_pretrained_fpn=True, disjoint_split=True, gop_len=32, clip_len=32)
        with patch("scripts.kaggle_shard_runner.build_sharded_dataset", return_value=object()):
            with self.assertRaisesRegex(RuntimeError, "real held-out"):
                run_poc_wishlist2_and_eval_shard(args)

    def test_release_validation_rejects_missing_data_before_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                data_root=directory,
                soup_dir=directory,
                output_dir=str(Path(directory) / "out"),
                variants=["spatial_only_scope"],
                device="cpu",
                img_h=384,
                img_w=640,
            )
            with self.assertRaisesRegex(ValueError, "131"):
                validate(args)
            self.assertFalse((Path(directory) / "out").exists())

    def test_dynamic_roi_graph_preserves_full_qg16_grid_on_640x384(self):
        # 384/16 = 24 rows, 640/16 = 40 cols -> row 23 is y=368, col 39 is x=624 (no 16x16 cap!)
        offsets = torch.zeros(3, 24, 40)
        offsets[0, 23, 39] = 4.0
        offsets[1, 0, 0] = 4.0
        offsets[2, 12, 20] = 3.5
        args = HybridX265STE._build_addroi_filter_from_offsets(offsets, 384, 640)
        graph = args[1]
        self.assertIn("split=3", graph)
        self.assertIn("x=624:y=368:w=16:h=16", graph)
        for index in range(3):
            self.assertIn(f"select='eq(n,{index})'", graph)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Real codec tools unavailable")
    def test_real_x265_and_x264_yuv420p_long_gop_and_ra(self):
        generator = torch.Generator().manual_seed(11)
        video = torch.rand(1, 8, 3, 64, 64, generator=generator)
        offsets = torch.zeros(1, 8, 4, 4)
        offsets[0, :, 0, 0] = 3.5
        codec = HybridX265STE(use_real_x265_in_forward=True)

        # HEVC Low-Delay P (family_id=0, bframes=0)
        out_hevc_ldp = codec.evaluate_real_x265(video, 32, offsets, family_id=0, gop_len=32, bframes=0)
        self.assertEqual(out_hevc_ldp["pix_fmt"], "yuv420p")
        self.assertEqual(out_hevc_ldp["frame_types"], ["I"] + ["P"] * 7)
        self.assertEqual(len(out_hevc_ldp["packet_sizes"]), 8)
        self.assertTrue(all(type(sz) is int and sz > 0 for sz in out_hevc_ldp["packet_sizes"]))

        # AVC libx264 Low-Delay P (family_id=1, bframes=0)
        out_avc_ldp = codec.evaluate_real_x265(video, 32, offsets, family_id=1, gop_len=32, bframes=0)
        self.assertEqual(out_avc_ldp["pix_fmt"], "yuv420p")
        self.assertEqual(out_avc_ldp["encoder"], "libx264")
        self.assertEqual(len(out_avc_ldp["packet_sizes"]), 8)

        # HEVC Random Access IBBP (family_id=0, bframes=4)
        out_hevc_ra = codec.evaluate_real_x265(video, 32, offsets, family_id=0, gop_len=32, bframes=4)
        self.assertEqual(out_hevc_ra["pix_fmt"], "yuv420p")
        self.assertEqual(out_hevc_ra["frame_types"][0], "I")
        self.assertIn("B", out_hevc_ra["frame_types"])
        self.assertEqual(len(out_hevc_ra["packet_sizes"]), 8)


if __name__ == "__main__":
    unittest.main()
