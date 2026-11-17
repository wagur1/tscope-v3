"""Tests for T-SCOPE v3 uniform real-libx265 MOT17-09 soup validation & mode connectivity (F15 & F20)."""
import copy
import json
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from losses.vcm_temporal_loss import VCMTemporalPreEditLoss
from scripts.experiment_protocol import VALIDATION_PROTOCOL, validate_release_validation
from scripts.validate_soups import extract_validation_features, validate


class FakeFeatureTeacher:
    use_torchvision = True
    detector = object()

    def __init__(self):
        self.call_sizes = []
        self.eval_called = False

    def to(self, device):
        return self

    def eval(self):
        self.eval_called = True
        return self

    def extract_fpn_features(self, images):
        self.call_sizes.append(images.shape[0])
        return {
            f"p{level}": F.avg_pool2d(images, kernel_size=stride, stride=stride)
            for level, stride in zip(range(2, 6), (4, 8, 16, 32))
        }


def validation_evidence():
    return {
        "status": "validated",
        "release_ready": True,
        "split": "MOT17-09",
        "loss_total": 1.0,
        "uniform_loss_total": 1.0,
        "protocol": copy.deepcopy(VALIDATION_PROTOCOL),
        "data_fingerprints": {
            "gt_sha256": "a" * 64,
            "frames": [{"frame": number, "sha256": "b" * 64} for number in range(1, 525)],
        },
        "source_soup_sha256": "c" * 64,
        "selection": "uniform_soup_no_test_selection",
    }


class FakeCompleteValidationDataset:
    def __init__(self, root):
        sequence = Path(root) / "MOT17-09-FRCNN" / "img1"
        self.clips = [
            {
                "type": "mot_frames",
                "seq_name": "MOT17-09-FRCNN",
                "frame_paths": [sequence / f"{number:06d}.jpg" for number in range(index * 4 + 1, index * 4 + 5)],
            }
            for index in range(131)
        ]

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        return {
            "video": torch.full((4, 3, 32, 64), float(index) * 0.001),
            "m_attn": torch.ones((4, 1, 32, 64)),
            "m_det": torch.ones((4, 1, 32, 64)),
        }


class FakeValidationModel:
    def __init__(self):
        self.loaded_bias = 0.0

    def to(self, device):
        return self

    def load_state_dict(self, state, strict):
        if "bias" in state:
            self.loaded_bias = float(state["bias"].item())

    def eval(self):
        return self

    def __call__(self, video, family, severity, reuse_iframe_qp, disable_temporal):
        b, t, _, h, w = video.shape
        return {
            "i_edit": video + self.loaded_bias,
            "attn_map": torch.ones((b, t, 1, h, w)),
            "eff_attn_map": torch.ones((b, t, 1, h, w)),
            "delta_obj": torch.zeros_like(video),
            "delta_bg": torch.zeros_like(video),
            "ctu_qp_offsets": torch.zeros((b, t, h // 16, w // 16)),
        }


class FakeUniformCodec:
    use_real_x265 = True

    def to(self, device):
        return self

    def forward(self, video_edit, severity, ctu_qp_offsets=None, family_id=None):
        b = video_edit.shape[0]
        return {
            "video_recon": video_edit,
            "rate_proxy_bpp": torch.full((b,), 0.1 + float(video_edit.mean().item()) * 0.0),
            "i_frame_bpp_proxy": torch.full((b,), 0.2),
            "p_frame_bpp_proxy": torch.full((b,), 0.05),
        }

    __call__ = forward


class FakeLossFromEdit:
    def to(self, device):
        return self

    def forward(self, **inputs):
        # Quadratic convex loss centered at i_orig + 0.2 so Uniform Soup (average of 0.0 and 0.4 -> 0.2) beats constituents!
        diff = inputs["i_edit"] - (inputs["i_orig"] + 0.2)
        return {"loss_total": (diff * diff).mean() + 1.0}

    __call__ = forward


class SoupValidationV3Tests(unittest.TestCase):
    def test_v3_validation_protocol_contract(self):
        self.assertEqual(VALIDATION_PROTOCOL["img_h"], 384)
        self.assertEqual(VALIDATION_PROTOCOL["img_w"], 640)
        self.assertEqual(VALIDATION_PROTOCOL["severities"], [0.2, 0.6])
        self.assertEqual(VALIDATION_PROTOCOL["uniform_codec"], "libx265_real_ste_yuv420p")
        ev = validation_evidence()
        self.assertIs(validate_release_validation(ev), ev)
        ev_greedy = copy.deepcopy(ev)
        ev_greedy["selection"] = "greedy_soup_val_mot17_09_no_test_selection"
        self.assertIs(validate_release_validation(ev_greedy), ev_greedy)

    def test_chunked_feature_extraction_exactness(self):
        images = torch.arange(5, dtype=torch.float32).view(5, 1, 1, 1).expand(5, 3, 64, 64)
        expected = FakeFeatureTeacher().extract_fpn_features(images)
        teacher = FakeFeatureTeacher()
        actual = extract_validation_features(teacher, images, chunk_size=2)
        self.assertEqual(teacher.call_sizes, [2, 2, 1])
        for level in expected:
            torch.testing.assert_close(actual[level], expected[level], rtol=0, atol=0)

    def test_uniform_criterion_constituent_variance_and_greedy_soup_connectivity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c_dir = root / "constituents"
            c_dir.mkdir()
            s_dir = root / "soups"
            s_dir.mkdir()
            out_dir = root / "validated"

            # Save 2 constituent shards with bias 0.0 and 0.4 -> Uniform soup has bias 0.2 (optimal!)
            torch.save({"variant": "wo_x265_ste_proxy", "shard_id": 0, "state_dict": {"bias": torch.tensor(0.0)}}, c_dir / "tscope_wo_x265_ste_proxy_shard00of05.pt")
            torch.save({"variant": "wo_x265_ste_proxy", "shard_id": 1, "state_dict": {"bias": torch.tensor(0.4)}}, c_dir / "tscope_wo_x265_ste_proxy_shard01of05.pt")

            soup_ckpt = {
                "schema_version": 3,
                "variant": "wo_x265_ste_proxy",
                "merged_shards": 2,
                "state_dict": {"bias": torch.tensor(0.2)},
                "constituents": [
                    {"shard_id": 0, "best_val_loss": 1.04, "source_path": "tscope_wo_x265_ste_proxy_shard00of05.pt"},
                    {"shard_id": 1, "best_val_loss": 1.04, "source_path": "tscope_wo_x265_ste_proxy_shard01of05.pt"},
                ],
            }
            torch.save(soup_ckpt, s_dir / "tscope_soup_wo_x265_ste_proxy.pt")

            args = SimpleNamespace(
                data_root=str(root),
                soup_dir=str(s_dir),
                constituent_dir=str(c_dir),
                constituent_manifest="",
                output_dir=str(out_dir),
                variants=["wo_x265_ste_proxy"],
                img_h=384,
                img_w=640,
                selection_mode="uniform",
                device="cpu",
                feature_chunk_size=4,
            )

            with ExitStack() as stack:
                stack.enter_context(patch("scripts.validate_soups._build_val_dataset", return_value=FakeCompleteValidationDataset(root)))
                stack.enter_context(patch("scripts.validate_soups.FrozenFPNDetectorTeacher", return_value=FakeFeatureTeacher()))
                stack.enter_context(patch("scripts.validate_soups.TemporalSCOPE", side_effect=lambda **kw: FakeValidationModel()))
                stack.enter_context(patch("scripts.validate_soups.HybridX265STE", return_value=FakeUniformCodec()))
                stack.enter_context(patch("scripts.validate_soups.VCMTemporalPreEditLoss", return_value=FakeLossFromEdit()))
                stack.enter_context(patch("scripts.validate_soups.sha256_file", return_value="a" * 64))
                stack.enter_context(patch("scripts.validate_soups.validate_soup_provenance"))
                stack.enter_context(patch("scripts.validate_soups.torch.set_num_threads"))
                stack.enter_context(patch("scripts.validate_soups.subprocess.check_output", return_value="ffmpeg version test"))
                stack.enter_context(patch("builtins.print"))
                paths = validate(args)

            saved_soup = torch.load(paths["wo_x265_ste_proxy"], map_location="cpu", weights_only=True)
            val_meta = saved_soup["validation"]
            conn = val_meta["soup_connectivity"]
            self.assertEqual(val_meta["status"], "validated")
            self.assertAlmostEqual(conn["uniform_soup_loss_total"], 1.0, places=5)
            self.assertAlmostEqual(conn["constituent_uniform_loss_mean"], 1.04, places=5)
            self.assertTrue(conn["linear_mode_connected"])
            self.assertAlmostEqual(conn["greedy_soup_loss_total"], 1.0, places=5)
            self.assertTrue((out_dir / "soup_connectivity_report.json").is_file())
            rep = json.loads((out_dir / "soup_connectivity_report.json").read_text(encoding="utf-8"))
            self.assertIn("wo_x265_ste_proxy", rep["variants"])


if __name__ == "__main__":
    unittest.main()
