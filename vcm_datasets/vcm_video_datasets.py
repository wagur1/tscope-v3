import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image


# Official MPEG-VCM / HEVC CTC 18 sequences in SFU-HW-Objects-v1 (Choi & Bajic)
SFU_HW_SEQUENCES: Dict[str, Tuple[int, int, int, str]] = {
    # SeqName: (Width, Height, FPS, Class)
    "Traffic": (2560, 1600, 30, "A"),
    "PeopleOnStreet": (2560, 1600, 30, "A"),
    "Kimono": (1920, 1080, 24, "B"),
    "ParkScene": (1920, 1080, 24, "B"),
    "Cactus": (1920, 1080, 50, "B"),
    "BasketballDrive": (1920, 1080, 50, "B"),
    "BQTerrace": (1920, 1080, 60, "B"),
    "BasketballDrill": (832, 480, 50, "C"),
    "BQMall": (832, 480, 60, "C"),
    "PartyScene": (832, 480, 50, "C"),
    "RaceHorsesC": (832, 480, 30, "C"),
    "BasketballPass": (416, 240, 50, "D"),
    "BQSquare": (416, 240, 60, "D"),
    "BlowingBubbles": (416, 240, 50, "D"),
    "RaceHorses": (416, 240, 30, "D"),
    "FourPeople": (1280, 720, 60, "E"),
    "Johnny": (1280, 720, 60, "E"),
    "KristenAndSara": (1280, 720, 60, "E"),
}

# Standard 80-class YOLO 0-based index -> Torchvision 91-class COCO 1-based category ID
YOLO80_TO_COCO91: List[int] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]

# KITTI Multi-Object Tracking string labels -> Torchvision COCO 1-based category IDs
KITTI_TYPE_TO_COCO91: Dict[str, int] = {
    "Pedestrian": 1,
    "Person_sitting": 1,
    "Person": 1,
    "Cyclist": 2,
    "Car": 3,
    "Van": 3,
    "Truck": 8,
    "Tram": 7,
}


def get_shard_slice(items: List[Any], shard_id: int, num_shards: int) -> List[Any]:
    """Deterministically partitions items across `num_shards` workers."""
    if not (0 <= shard_id < num_shards):
        raise ValueError(f"shard_id={shard_id} must be in [0, {num_shards - 1}]")
    return [item for idx, item in enumerate(items) if idx % num_shards == shard_id]


def parse_mot_gt_with_tracks(
    gt_path: Path,
) -> Tuple[
    Dict[int, List[Tuple[float, float, float, float, int, int]]],
    Dict[int, List[Tuple[float, float, float, float]]],
]:
    """
    Parses MOT17 / MOT20 ground-truth file `gt/gt.txt` following official MOTChallenge protocol,
    preserving per-instance `track_id` for MOTA / IDF1 / HOTA tracking evaluation (`F11` fix).
    Format per line: `frame_id, track_id, x, y, w, h, mark, class_id, visibility`
      - Active pedestrian targets: `mark == 1 and class_id == 1` -> `(x1, y1, x2, y2, track_id, 1)`.
      - Ignore / distractor regions: `mark == 0 or class_id in (2, 7, 8, 12)` -> `(x1, y1, x2, y2)`.
    """
    active_records: Dict[int, List[Tuple[float, float, float, float, int, int]]] = {}
    ignore_boxes: Dict[int, List[Tuple[float, float, float, float]]] = {}
    if not gt_path.exists():
        return active_records, ignore_boxes

    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame_id = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            if w <= 1.0 or h <= 1.0:
                continue
            if len(parts) >= 8:
                mark = int(float(parts[6]))
                cls_id = int(float(parts[7]))
                if mark == 1 and cls_id == 1:
                    active_records.setdefault(frame_id, []).append(
                        (x, y, x + w, y + h, track_id, 1)
                    )
                elif mark == 0 or cls_id in (2, 7, 8, 12):
                    ignore_boxes.setdefault(frame_id, []).append((x, y, x + w, y + h))
            else:
                active_records.setdefault(frame_id, []).append(
                    (x, y, x + w, y + h, track_id, 1)
                )
    return active_records, ignore_boxes


def parse_mot_gt_file(
    gt_path: Path,
) -> Tuple[
    Dict[int, List[Tuple[float, float, float, float]]],
    Dict[int, List[Tuple[float, float, float, float]]],
]:
    """Backward-compatible wrapper returning `(active_boxes_by_frame, ignore_boxes_by_frame)`."""
    active_records, ignore_boxes = parse_mot_gt_with_tracks(gt_path)
    active_boxes = {
        fid: [(r[0], r[1], r[2], r[3]) for r in rows]
        for fid, rows in active_records.items()
    }
    return active_boxes, ignore_boxes


def parse_kitti_tracking_label_file(
    label_path: Path,
) -> Tuple[
    Dict[int, List[Tuple[float, float, float, float, int, int]]],
    Dict[int, List[Tuple[float, float, float, float]]],
]:
    """
    Parses KITTI Multi-Object Tracking `label_02/<seq>.txt` (`leducnhuan/kitti-tracking`):
      `frame track_id type truncated occluded alpha x1 y1 x2 y2 ...` (0-based `frame`).
    """
    active_records: Dict[int, List[Tuple[float, float, float, float, int, int]]] = {}
    ignore_boxes: Dict[int, List[Tuple[float, float, float, float]]] = {}
    if not label_path.exists():
        return active_records, ignore_boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            frame_idx = int(parts[0])
            track_id = int(parts[1])
            obj_type = parts[2]
            x1, y1, x2, y2 = float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9])
            if (x2 - x1) <= 1.0 or (y2 - y1) <= 1.0:
                continue
            if obj_type in KITTI_TYPE_TO_COCO91:
                coco_cls = KITTI_TYPE_TO_COCO91[obj_type]
                active_records.setdefault(frame_idx, []).append(
                    (x1, y1, x2, y2, track_id, coco_cls)
                )
            elif obj_type in ("DontCare", "Misc"):
                ignore_boxes.setdefault(frame_idx, []).append((x1, y1, x2, y2))
    return active_records, ignore_boxes


def parse_yolo_label_with_classes(
    txt_path: Path, width: int, height: int
) -> List[Tuple[float, float, float, float, int, int]]:
    """
    Parses SFU-HW-Objects-v1 YOLO format label file (`class_id cx cy w h` normalized in [0,1]),
    mapping 0-indexed YOLO COCO class IDs to 1-based Torchvision COCO category IDs.
    """
    records: List[Tuple[float, float, float, float, int, int]] = []
    if not txt_path.exists():
        return records
    with open(txt_path, "r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            yolo_cls = int(float(parts[0]))
            coco_cls = (
                YOLO80_TO_COCO91[yolo_cls]
                if 0 <= yolo_cls < len(YOLO80_TO_COCO91)
                else max(1, yolo_cls + 1)
            )
            cx, cy, bw, bh = (
                float(parts[1]) * width,
                float(parts[2]) * height,
                float(parts[3]) * width,
                float(parts[4]) * height,
            )
            x1, y1 = max(0.0, cx - 0.5 * bw), max(0.0, cy - 0.5 * bh)
            x2, y2 = min(float(width), cx + 0.5 * bw), min(float(height), cy + 0.5 * bh)
            if x2 > x1 and y2 > y1:
                records.append((x1, y1, x2, y2, row_idx + 1, coco_cls))
    return records


def parse_yolo_label_file(
    txt_path: Path, width: int, height: int
) -> List[Tuple[float, float, float, float]]:
    """Backward-compatible box-only wrapper for `parse_yolo_label_with_classes`."""
    return [
        (r[0], r[1], r[2], r[3])
        for r in parse_yolo_label_with_classes(txt_path, width, height)
    ]


def boxes_to_masks(
    boxes: Sequence[Sequence[float]],
    height: int,
    width: int,
    context_pad: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs spatial masks `M_attn` and `M_det` from bounding boxes:
      - `M_attn`: Tight object mask `[1, H, W]`
      - `M_det` : Expanded detector context mask `[1, H, W]` (`M_bg = 1 - M_det`)
    """
    m_attn = torch.zeros((1, height, width), dtype=torch.float32)
    m_det = torch.zeros((1, height, width), dtype=torch.float32)

    for box in boxes:
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        ix1 = max(0, min(width - 1, int(round(x1))))
        iy1 = max(0, min(height - 1, int(round(y1))))
        ix2 = max(ix1 + 1, min(width, int(round(x2))))
        iy2 = max(iy1 + 1, min(height, int(round(y2))))
        m_attn[0, iy1:iy2, ix1:ix2] = 1.0

        dx1 = max(0, ix1 - context_pad)
        dy1 = max(0, iy1 - context_pad)
        dx2 = min(width, ix2 + context_pad)
        dy2 = min(height, iy2 + context_pad)
        m_det[0, dy1:dy2, dx1:dx2] = 1.0

    return m_attn, m_det


def read_raw_yuv420_frame(
    yuv_path: Path, width: int, height: int, frame_idx: int
) -> torch.Tensor:
    """
    Reads a single frame from an uncompressed 8-bit YUV 4:2:0 planar (`.yuv`) file
    (standard MPEG-VCM / SFU-HW-Objects-v1 format) and converts to RGB tensor `[3, H, W]` in `[0, 1]`.
    """
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    frame_bytes = y_size + 2 * uv_size

    with open(yuv_path, "rb") as f:
        f.seek(frame_idx * frame_bytes)
        raw = f.read(frame_bytes)
        if len(raw) < frame_bytes:
            raise EOFError(f"Frame {frame_idx} out of range in {yuv_path}")

    buf = np.frombuffer(raw, dtype=np.uint8)
    y = buf[:y_size].reshape((height, width)).astype(np.float32)
    u = (
        buf[y_size : y_size + uv_size]
        .reshape((height // 2, width // 2))
        .repeat(2, axis=0)
        .repeat(2, axis=1)
        .astype(np.float32)
    )
    v = (
        buf[y_size + uv_size :]
        .reshape((height // 2, width // 2))
        .repeat(2, axis=0)
        .repeat(2, axis=1)
        .astype(np.float32)
    )

    y_val = y - 16.0
    u_val = u - 128.0
    v_val = v - 128.0

    r = 1.164 * y_val + 1.596 * v_val
    g = 1.164 * y_val - 0.392 * u_val - 0.813 * v_val
    b = 1.164 * y_val + 2.017 * u_val

    rgb = np.stack([r, g, b], axis=0) / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    return torch.from_numpy(rgb.astype(np.float32))


# Disjoint Train / Validation / Held-Out Test Split on MOT17
MOT17_TRAIN_PREFIXES = ("MOT17-02", "MOT17-04", "MOT17-05", "MOT17-10")
MOT17_VAL_PREFIXES = ("MOT17-09",)
MOT17_TEST_PREFIXES = ("MOT17-11", "MOT17-13")


def normalize_img_size(
    img_size: Union[int, Tuple[int, int], List[int]] = (384, 640),
    img_h: Optional[int] = None,
    img_w: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Resolves `(target_h, target_w)` divisible by 16 (and 32 for `(384, 640)`):
      - `img_h` & `img_w` take precedence if provided.
      - `img_size=(384, 640)` preserves native 16:9 widescreen aspect ratio (`F11` fix).
      - `img_size=512` (scalar `int`) remains 100% backward-compatible as `(512, 512)`.
    """
    if img_h is not None and img_w is not None:
        raw_h, raw_w = int(img_h), int(img_w)
    elif isinstance(img_size, (tuple, list)) and len(img_size) == 2:
        raw_h, raw_w = int(img_size[0]), int(img_size[1])
    elif isinstance(img_size, int):
        raw_h, raw_w = int(img_size), int(img_size)
    else:
        raise ValueError(f"Unsupported img_size={img_size!r}; expected int or (H, W) tuple.")
    th = max(16, (raw_h // 16) * 16)
    tw = max(16, (raw_w // 16) * 16)
    return th, tw


class RealVCMVideoDataset(Dataset):
    """
    Unified Sharded Video Dataset (`tscope-v3`, `F05`, `F11`, `F25` fixes):
      - Supports **MOT17**, **MOT20** (`ismailelbouknify/mot-20`), **KITTI Multi-Object Tracking**
        (`leducnhuan/kitti-tracking`), and **SFU-HW-Objects-v1 / HEVC Class B** (`minhngt02/hevc-b`).
      - Supports aspect-preserving 16:9 widescreen resize `img_size=(384, 640)` (`--img_h 384 --img_w 640`)
        with independent `(sx, sy)` box scaling and backward-compatible scalar `int` support.
      - Supports `clip_len=32, stride=32` contiguous GOP=32 evaluation clips (`1,632` frames = `51` clips
        across `MOT17-11` and `MOT17-13`).
      - Returns aligned `gt_boxes` `[T, max_boxes, 4]`, `gt_track_ids` `[T, max_boxes]`, and
        `gt_classes` `[T, max_boxes]` (`-1` padded) for both AP and ByteTrack MOTA/IDF1/HOTA evaluation.
    """

    MAX_BOXES: int = 512

    def __init__(
        self,
        data_root: str,
        shard_id: int = 0,
        num_shards: int = 20,
        clip_len: int = 4,
        stride: int = 4,
        img_size: Union[int, Tuple[int, int], List[int]] = (384, 640),
        is_train: bool = True,
        enforce_disjoint_split: bool = False,
        split: Optional[str] = None,
        img_h: Optional[int] = None,
        img_w: Optional[int] = None,
        sfu_label_root: Optional[str] = None,
        label_root: Optional[str] = None,
        dataset_filter: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root) if data_root else None
        self.clip_len = int(clip_len)
        self.stride = int(stride)
        self.target_h, self.target_w = normalize_img_size(img_size, img_h=img_h, img_w=img_w)
        self.img_size = (
            self.target_h
            if self.target_h == self.target_w and isinstance(img_size, int) and img_h is None and img_w is None
            else (self.target_h, self.target_w)
        )
        self.is_train = is_train
        self.enforce_disjoint_split = enforce_disjoint_split
        self.split = split if split is not None else ("train" if is_train else "test")
        if self.split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {self.split}")
        eff_lbl = sfu_label_root or label_root
        self.sfu_label_root = Path(eff_lbl) if eff_lbl else None
        self.dataset_filter = dataset_filter.lower().strip() if dataset_filter else None

        all_clips = self._discover_clips()
        self.clips = get_shard_slice(all_clips, shard_id, num_shards)

    def _resolve_sfu_label_file(self, yuv_path: Path, seq_stem: str, frame_idx: int) -> Optional[Path]:
        prefix = seq_stem.split("_")[0]
        candidate_dirs: List[Path] = []
        if self.sfu_label_root is not None:
            candidate_dirs.extend(
                [
                    self.sfu_label_root / seq_stem,
                    self.sfu_label_root / prefix,
                    self.sfu_label_root / "labels" / seq_stem,
                    self.sfu_label_root / "labels" / prefix,
                    self.sfu_label_root,
                ]
            )
        candidate_dirs.extend(
            [
                yuv_path.parent / "labels" / seq_stem,
                yuv_path.parent / "labels" / prefix,
                yuv_path.parent / "SFU-HW-Objects-v1" / seq_stem,
                yuv_path.parent / "SFU-HW-Objects-v1" / prefix,
            ]
        )
        name_patterns = (
            f"{seq_stem}_{frame_idx:03d}.txt",
            f"{seq_stem}_{frame_idx:04d}.txt",
            f"{prefix}_{frame_idx:03d}.txt",
            f"{prefix}_{frame_idx:04d}.txt",
            f"{frame_idx:06d}.txt",
            f"{frame_idx:04d}.txt",
        )
        for cdir in candidate_dirs:
            if not cdir.is_dir():
                continue
            for pat in name_patterns:
                cand = cdir / pat
                if cand.is_file():
                    return cand
        return None

    def _discover_clips(self) -> List[Dict[str, Any]]:
        clips: List[Dict[str, Any]] = []
        if not self.data_root or not self.data_root.exists():
            return clips

        allow_mot = self.dataset_filter in (None, "mot", "mot17", "mot20")
        allow_kitti = self.dataset_filter in (None, "kitti")
        allow_sfu = self.dataset_filter in (None, "sfu", "hevc", "yuv")

        # 1. Discover MOT17 / MOT20 sequences (`img1` + `gt/gt.txt`)
        if allow_mot:
            for img1_dir in sorted(self.data_root.rglob("img1")):
                if not img1_dir.is_dir():
                    continue
                seq_dir = img1_dir.parent
                if any(tag in seq_dir.name for tag in ("-SDP", "-DPM")):
                    continue
                if self.dataset_filter == "mot17" and not seq_dir.name.startswith("MOT17-"):
                    continue
                if self.dataset_filter == "mot20" and not seq_dir.name.startswith("MOT20-"):
                    continue
                if self.enforce_disjoint_split and seq_dir.name.startswith("MOT17-"):
                    allowed = {
                        "train": MOT17_TRAIN_PREFIXES,
                        "val": MOT17_VAL_PREFIXES,
                        "test": MOT17_TEST_PREFIXES,
                    }[self.split]
                    if not any(seq_dir.name.startswith(p) for p in allowed):
                        continue
                gt_path = seq_dir / "gt" / "gt.txt"
                if not gt_path.exists():
                    continue
                gt_records, ignore_map = parse_mot_gt_with_tracks(gt_path)
                frames = sorted(img1_dir.glob("*.jpg")) + sorted(img1_dir.glob("*.png"))
                for start in range(0, len(frames) - self.clip_len + 1, self.stride):
                    clips.append(
                        {
                            "type": "mot_frames",
                            "seq_name": seq_dir.name,
                            "frame_paths": frames[start : start + self.clip_len],
                            "gt_records": gt_records,
                            "ignore_map": ignore_map,
                        }
                    )

        # 2. Discover KITTI Multi-Object Tracking sequences (`image_02/<seq>/*.png` + `label_02/<seq>.txt`)
        if allow_kitti:
            for img02_root in sorted(self.data_root.rglob("image_02")):
                if not img02_root.is_dir():
                    continue
                lbl02_root = img02_root.parent / "label_02"
                if not lbl02_root.is_dir():
                    continue
                for seq_dir in sorted(p for p in img02_root.iterdir() if p.is_dir()):
                    lbl_file = lbl02_root / f"{seq_dir.name}.txt"
                    if not lbl_file.is_file():
                        continue
                    gt_records, ignore_map = parse_kitti_tracking_label_file(lbl_file)
                    frames = sorted(seq_dir.glob("*.png")) + sorted(seq_dir.glob("*.jpg"))
                    for start in range(0, len(frames) - self.clip_len + 1, self.stride):
                        clips.append(
                            {
                                "type": "kitti_frames",
                                "seq_name": f"KITTI-{seq_dir.name}",
                                "frame_paths": frames[start : start + self.clip_len],
                                "gt_records": gt_records,
                                "ignore_map": ignore_map,
                            }
                        )

        # 3. Discover SFU-HW-Objects-v1 / HEVC CTC raw YUV 4:2:0 sequences (`*.yuv`)
        if allow_sfu:
            yuv_files = (
                [self.data_root]
                if self.data_root.is_file() and self.data_root.suffix.lower() == ".yuv"
                else sorted(self.data_root.rglob("*.yuv"))
            )
            for yuv_path in yuv_files:
                m = re.search(r"_(\d+)x(\d+)_", yuv_path.name)
                if m:
                    w, h = int(m.group(1)), int(m.group(2))
                else:
                    prefix = yuv_path.stem.split("_")[0]
                    w, h = SFU_HW_SEQUENCES.get(prefix, (416, 240, 30, "D"))[:2]

                frame_bytes = int(w * h * 1.5)
                num_frames = yuv_path.stat().st_size // max(1, frame_bytes)
                for start in range(0, num_frames - self.clip_len + 1, self.stride):
                    clips.append(
                        {
                            "type": "sfu_yuv",
                            "seq_name": yuv_path.stem,
                            "yuv_path": yuv_path,
                            "width": w,
                            "height": h,
                            "start_frame": start,
                        }
                    )

        return clips

    def physical_frame_manifest(self) -> List[str]:
        frames: List[str] = []
        for meta in self._discover_clips():
            if meta["type"] != "mot_frames":
                raise ValueError("Final MOT17 evaluation does not accept non-MOT image sequences")
            sequence = meta["seq_name"].split("-FRCNN")[0]
            if sequence not in ("MOT17-11", "MOT17-13"):
                raise ValueError("Final frame manifest contains a non-held-out sequence")
            frames.extend(
                f"{sequence}/{int(re.sub(r'\\D', '', path.stem) or '1'):06d}"
                for path in meta["frame_paths"]
            )
        if len(frames) != len(set(frames)):
            raise ValueError("Final frame manifest contains overlapping or duplicate frames")
        return frames

    def sequence_frame_manifest(self) -> List[str]:
        """General non-overlapping frame manifest across any discovered dataset (MOT20, KITTI, SFU, MOT17)."""
        frames: List[str] = []
        for meta in self._discover_clips():
            if meta["type"] in ("mot_frames", "kitti_frames"):
                seq = meta["seq_name"].split("-FRCNN")[0]
                frames.extend(
                    f"{seq}/{int(re.sub(r'\\D', '', path.stem) or '0'):06d}"
                    for path in meta["frame_paths"]
                )
            else:
                frames.extend(
                    f"{meta['seq_name']}/{meta['start_frame'] + off:06d}"
                    for off in range(self.clip_len)
                )
        return frames

    def __len__(self) -> int:
        return len(self.clips)

    @classmethod
    def _pack_boxes_tensor(
        cls, boxes_list: List[torch.Tensor], max_boxes: Optional[int] = None
    ) -> torch.Tensor:
        cap = cls.MAX_BOXES if max_boxes is None else max_boxes
        t = len(boxes_list)
        out = torch.full((t, cap, 4), -1.0, dtype=torch.float32)
        for f_idx, b_t in enumerate(boxes_list):
            if b_t.numel() == 0:
                continue
            if b_t.shape[0] > cap:
                raise RuntimeError(f"Frame has {b_t.shape[0]} boxes exceeding max_boxes={cap}")
            n = b_t.shape[0]
            out[f_idx, :n] = b_t[:n]
        return out

    @classmethod
    def _pack_int_meta_tensor(
        cls, ids_list: List[torch.Tensor], max_boxes: Optional[int] = None
    ) -> torch.Tensor:
        cap = cls.MAX_BOXES if max_boxes is None else max_boxes
        t = len(ids_list)
        out = torch.full((t, cap), -1, dtype=torch.long)
        for f_idx, id_t in enumerate(ids_list):
            if id_t.numel() == 0:
                continue
            if id_t.shape[0] > cap:
                raise RuntimeError(f"Frame has {id_t.shape[0]} entries exceeding max_boxes={cap}")
            n = id_t.shape[0]
            out[f_idx, :n] = id_t[:n].long()
        return out

    @classmethod
    def _transform_boxes_list(
        cls,
        raw_boxes_per_frame: List[List[Tuple[float, float, float, float]]],
        target: Union[int, Tuple[int, int]],
        sx: float = 1.0,
        sy: float = 1.0,
        x0: float = 0.0,
        y0: float = 0.0,
    ) -> torch.Tensor:
        if isinstance(target, int):
            th, tw = target, target
        else:
            th, tw = int(target[0]), int(target[1])
        out_boxes: List[torch.Tensor] = []
        for f_boxes in raw_boxes_per_frame:
            valid = []
            for x1, y1, x2, y2 in f_boxes:
                nx1 = max(0.0, min(float(tw), (x1 - x0) * sx))
                ny1 = max(0.0, min(float(th), (y1 - y0) * sy))
                nx2 = max(0.0, min(float(tw), (x2 - x0) * sx))
                ny2 = max(0.0, min(float(th), (y2 - y0) * sy))
                if (nx2 - nx1) >= 2.0 and (ny2 - ny1) >= 2.0:
                    valid.append([nx1, ny1, nx2, ny2])
            out_boxes.append(
                torch.tensor(valid, dtype=torch.float32)
                if valid
                else torch.zeros((0, 4), dtype=torch.float32)
            )
        return cls._pack_boxes_tensor(out_boxes)

    @classmethod
    def _transform_records_list(
        cls,
        raw_records_per_frame: List[List[Tuple[float, float, float, float, int, int]]],
        target_hw: Tuple[int, int],
        sx: float = 1.0,
        sy: float = 1.0,
        x0: float = 0.0,
        y0: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Jointly transforms `(x1, y1, x2, y2, track_id, cls_id)` into `(gt_boxes, gt_track_ids, gt_classes)`
        with exact 1-to-1 row alignment across `[T, MAX_BOXES]`.
        """
        th, tw = int(target_hw[0]), int(target_hw[1])
        out_boxes: List[torch.Tensor] = []
        out_tracks: List[torch.Tensor] = []
        out_classes: List[torch.Tensor] = []
        for f_recs in raw_records_per_frame:
            v_boxes: List[List[float]] = []
            v_tracks: List[int] = []
            v_classes: List[int] = []
            for x1, y1, x2, y2, tid, cid in f_recs:
                nx1 = max(0.0, min(float(tw), (x1 - x0) * sx))
                ny1 = max(0.0, min(float(th), (y1 - y0) * sy))
                nx2 = max(0.0, min(float(tw), (x2 - x0) * sx))
                ny2 = max(0.0, min(float(th), (y2 - y0) * sy))
                if (nx2 - nx1) >= 2.0 and (ny2 - ny1) >= 2.0:
                    v_boxes.append([nx1, ny1, nx2, ny2])
                    v_tracks.append(int(tid))
                    v_classes.append(int(cid))
            out_boxes.append(
                torch.tensor(v_boxes, dtype=torch.float32)
                if v_boxes
                else torch.zeros((0, 4), dtype=torch.float32)
            )
            out_tracks.append(
                torch.tensor(v_tracks, dtype=torch.long)
                if v_tracks
                else torch.zeros((0,), dtype=torch.long)
            )
            out_classes.append(
                torch.tensor(v_classes, dtype=torch.long)
                if v_classes
                else torch.zeros((0,), dtype=torch.long)
            )
        return (
            cls._pack_boxes_tensor(out_boxes),
            cls._pack_int_meta_tensor(out_tracks),
            cls._pack_int_meta_tensor(out_classes),
        )

    def _crop_or_resize_clip(
        self,
        video: torch.Tensor,
        m_attn: torch.Tensor,
        m_det: torch.Tensor,
        raw_records_per_frame: List[List[Tuple[float, float, float, float, int, int]]],
        raw_ignore_per_frame: List[List[Tuple[float, float, float, float]]],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Aspect-preserving `(target_h, target_w)` resize at evaluation (or 50% of train steps),
        and object-centered multi-scale `(target_h, target_w)` crop during the remaining 50% of train steps.
        """
        _, _, h, w = video.shape
        th, tw = self.target_h, self.target_w

        if h < th or w < tw or not self.is_train or torch.rand(1).item() < 0.50:
            v_out = F.interpolate(video, size=(th, tw), mode="bilinear", align_corners=False)
            ma_out = F.interpolate(m_attn, size=(th, tw), mode="nearest")
            md_out = F.interpolate(m_det, size=(th, tw), mode="nearest")
            sx, sy = float(tw) / float(w), float(th) / float(h)
            gt_boxes, gt_tracks, gt_classes = self._transform_records_list(
                raw_records_per_frame, (th, tw), sx=sx, sy=sy
            )
            ignore_boxes = self._transform_boxes_list(
                raw_ignore_per_frame, (th, tw), sx=sx, sy=sy
            )
            return v_out, ma_out, md_out, gt_boxes, gt_tracks, gt_classes, ignore_boxes

        min_scale = min(1.0, max(float(th) / float(h), float(tw) / float(w), 0.45))
        scale = min_scale + (1.0 - min_scale) * float(torch.rand(1).item())
        rh = max(th, int(round(h * scale)))
        rw = max(tw, int(round(w * scale)))
        if (rh, rw) != (h, w):
            video = F.interpolate(video, size=(rh, rw), mode="bilinear", align_corners=False)
            m_attn = F.interpolate(m_attn, size=(rh, rw), mode="nearest")
            m_det = F.interpolate(m_det, size=(rh, rw), mode="nearest")
        sx, sy = float(rw) / float(w), float(rh) / float(h)

        ys, xs = torch.where(m_attn[0, 0] > 0.5)
        if len(ys) > 0:
            jitter_y = int(torch.randint(-32, 33, (1,)).item())
            jitter_x = int(torch.randint(-32, 33, (1,)).item())
            cy = int(ys.float().mean().item()) + jitter_y
            cx = int(xs.float().mean().item()) + jitter_x
            y0 = max(0, min(rh - th, cy - th // 2))
            x0 = max(0, min(rw - tw, cx - tw // 2))
        else:
            y0 = int(torch.randint(0, rh - th + 1, (1,)).item())
            x0 = int(torch.randint(0, rw - tw + 1, (1,)).item())

        gt_boxes, gt_tracks, gt_classes = self._transform_records_list(
            raw_records_per_frame,
            (th, tw),
            sx=sx,
            sy=sy,
            x0=float(x0) / sx,
            y0=float(y0) / sy,
        )
        ignore_boxes = self._transform_boxes_list(
            raw_ignore_per_frame,
            (th, tw),
            sx=sx,
            sy=sy,
            x0=float(x0) / sx,
            y0=float(y0) / sy,
        )
        return (
            video[:, :, y0 : y0 + th, x0 : x0 + tw],
            m_attn[:, :, y0 : y0 + th, x0 : x0 + tw],
            m_det[:, :, y0 : y0 + th, x0 : x0 + tw],
            gt_boxes,
            gt_tracks,
            gt_classes,
            ignore_boxes,
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self.clips[idx]
        frames_t, mattn_t, mdet_t = [], [], []
        raw_records_per_frame: List[List[Tuple[float, float, float, float, int, int]]] = []
        raw_ignore_per_frame: List[List[Tuple[float, float, float, float]]] = []
        has_annotation_source = False
        orig_h, orig_w = self.target_h, self.target_w

        if meta["type"] == "mot_frames":
            has_annotation_source = True
            gt_records = meta["gt_records"]
            ignore_map = meta.get("ignore_map", {})
            for fpath in meta["frame_paths"]:
                frame_num = int(re.sub(r"\D", "", fpath.stem) or "1")
                img = Image.open(fpath).convert("RGB")
                w, h = img.size
                orig_h, orig_w = h, w
                tensor_img = torch.from_numpy(
                    np.array(img, dtype=np.float32) / 255.0
                ).permute(2, 0, 1)
                recs = gt_records.get(frame_num, [])
                ig_boxes = ignore_map.get(frame_num, [])
                ma, md = boxes_to_masks(recs, h, w, context_pad=16)
                frames_t.append(tensor_img)
                mattn_t.append(ma)
                mdet_t.append(md)
                raw_records_per_frame.append(recs)
                raw_ignore_per_frame.append(ig_boxes)
            frame_ids = [
                f"{meta['seq_name'].split('-FRCNN')[0]}/{int(re.sub(r'\\D', '', path.stem) or '1'):06d}"
                for path in meta["frame_paths"]
            ]

        elif meta["type"] == "kitti_frames":
            has_annotation_source = True
            gt_records = meta["gt_records"]
            ignore_map = meta.get("ignore_map", {})
            for fpath in meta["frame_paths"]:
                frame_num = int(re.sub(r"\D", "", fpath.stem) or "0")
                img = Image.open(fpath).convert("RGB")
                w, h = img.size
                orig_h, orig_w = h, w
                tensor_img = torch.from_numpy(
                    np.array(img, dtype=np.float32) / 255.0
                ).permute(2, 0, 1)
                recs = gt_records.get(frame_num, [])
                ig_boxes = ignore_map.get(frame_num, [])
                ma, md = boxes_to_masks(recs, h, w, context_pad=16)
                frames_t.append(tensor_img)
                mattn_t.append(ma)
                mdet_t.append(md)
                raw_records_per_frame.append(recs)
                raw_ignore_per_frame.append(ig_boxes)
            frame_ids = [
                f"{meta['seq_name']}/{int(re.sub(r'\\D', '', path.stem) or '0'):06d}"
                for path in meta["frame_paths"]
            ]

        elif meta["type"] == "sfu_yuv":
            w, h = meta["width"], meta["height"]
            orig_h, orig_w = h, w
            start_f = meta["start_frame"]
            for offset in range(self.clip_len):
                f_idx = start_f + offset
                tensor_img = read_raw_yuv420_frame(meta["yuv_path"], w, h, f_idx)
                lbl_file = self._resolve_sfu_label_file(meta["yuv_path"], meta["seq_name"], f_idx)
                if lbl_file is not None:
                    has_annotation_source = True
                    recs = parse_yolo_label_with_classes(lbl_file, w, h)
                else:
                    recs = []
                ma, md = boxes_to_masks(recs, h, w, context_pad=16)
                frames_t.append(tensor_img)
                mattn_t.append(ma)
                mdet_t.append(md)
                raw_records_per_frame.append(recs)
                raw_ignore_per_frame.append([])
            frame_ids = [
                f"{meta['seq_name']}/{start_f + offset:06d}" for offset in range(self.clip_len)
            ]
        else:
            raise ValueError(f"Unknown clip metadata type: {meta.get('type')}")

        video = torch.stack(frames_t, dim=0)
        m_attn = torch.stack(mattn_t, dim=0)
        m_det = torch.stack(mdet_t, dim=0)

        if not has_annotation_source and m_det.sum() == 0:
            _, _, h_v, w_v = video.shape
            m_attn[:, :, h_v // 4 : 3 * h_v // 4, w_v // 4 : 3 * w_v // 4] = 1.0
            m_det[:, :, h_v // 5 : 4 * h_v // 5, w_v // 5 : 4 * w_v // 5] = 1.0

        video, m_attn, m_det, gt_boxes, gt_track_ids, gt_classes, ignore_boxes = (
            self._crop_or_resize_clip(
                video, m_attn, m_det, raw_records_per_frame, raw_ignore_per_frame
            )
        )
        return {
            "video": video,
            "m_attn": m_attn,
            "m_det": m_det,
            "gt_boxes": gt_boxes,
            "gt_track_ids": gt_track_ids,
            "gt_classes": gt_classes,
            "gt_labels": gt_classes,
            "ignore_boxes": ignore_boxes,
            "clip_id": torch.tensor(idx, dtype=torch.long),
            "seq_name": meta["seq_name"],
            "frame_ids": frame_ids,
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.long),
        }
