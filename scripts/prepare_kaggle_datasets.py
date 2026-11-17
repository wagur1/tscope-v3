"""Auto-discover and prepare mounted VCM video datasets (MOT17, MOT20, KITTI, HEVC-B/SFU) on Kaggle or locally."""
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


VERIFIED_KAGGLE_DATASET_SLUGS: Dict[str, str] = {
    "mot17": "ahmedsamir1598/mot17challenge",
    "mot20": "ismailelbouknify/mot-20",
    "mot20_coco": "nghanguyntrn/mot20-dataset",
    "kitti": "leducnhuan/kitti-tracking",
    "hevc_b": "minhngt02/hevc-b",
    "visdrone_vid": "huseyincavus/visdrone2019-vid",
}

KAGGLE_MOT17_PATHS: List[str] = [
    "/kaggle/input/mot17challenge/MOT17/train",
    "/kaggle/input/mot17challenge",
    "/kaggle/input/mot-17/MOT17/train",
    "/kaggle/input/mot-17",
    "/kaggle/input/mot17",
    "/kaggle/input/motchallenge-mot17",
]

KAGGLE_MOT20_PATHS: List[str] = [
    "/kaggle/input/mot-20/MOT20/train",
    "/kaggle/input/mot-20",
    "/kaggle/input/mot20-dataset/MOT20/train",
    "/kaggle/input/mot20-dataset",
    "/kaggle/input/mot20traintest/MOT20/MOT20/train",
    "/kaggle/input/mot20traintest",
]

KAGGLE_KITTI_PATHS: List[str] = [
    "/kaggle/input/kitti-tracking/kitti_tracking/training",
    "/kaggle/input/kitti-tracking/training",
    "/kaggle/input/kitti-tracking",
]

KAGGLE_SFU_YUV_PATHS: List[str] = [
    "/kaggle/input/hevc-b",
    "/kaggle/input/sfu-hw-objects-v1",
]


def _scan_mot_train_roots(root: Path) -> Dict[str, Path]:
    """Deterministically discovers MOT17 and MOT20 train roots under `root`."""
    found: Dict[str, Path] = {}
    if not root.exists():
        return found
    for gt_file in sorted(root.rglob("gt.txt")):
        seq_dir = gt_file.parent.parent
        if not (seq_dir / "img1").is_dir():
            continue
        seq_name = seq_dir.name.upper()
        train_root = seq_dir.parent
        if seq_name.startswith("MOT17-") and "mot17" not in found:
            found["mot17"] = train_root
        elif seq_name.startswith("MOT20-") and "mot20" not in found:
            found["mot20"] = train_root
        elif "mot_other" not in found:
            found["mot_other"] = train_root
    return found


def _scan_kitti_root(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    for label_dir in sorted(root.rglob("label_02")):
        if label_dir.is_dir() and (label_dir.parent / "image_02").is_dir():
            return label_dir.parent
    return None


def _scan_yuv_root(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    for yuv_file in sorted(root.rglob("*.yuv")):
        if yuv_file.is_file():
            return yuv_file.parent
    return None


def find_all_mounted_kaggle_datasets(search_root: Optional[Path] = None) -> Dict[str, Path]:
    """Returns a deterministic mapping `{dataset_key: Path}` for all datasets mounted under `/kaggle/input`."""
    kaggle_in = Path(search_root) if search_root is not None else Path("/kaggle/input")
    discovered: Dict[str, Path] = {}

    mot_roots = _scan_mot_train_roots(kaggle_in)
    discovered.update(mot_roots)

    if "mot17" not in discovered:
        for p_str in KAGGLE_MOT17_PATHS:
            p = Path(p_str)
            if p.exists():
                discovered["mot17"] = p
                break

    if "mot20" not in discovered:
        for p_str in KAGGLE_MOT20_PATHS:
            p = Path(p_str)
            if p.exists():
                discovered["mot20"] = p
                break

    kitti_root = _scan_kitti_root(kaggle_in)
    if kitti_root is not None:
        discovered["kitti"] = kitti_root
    else:
        for p_str in KAGGLE_KITTI_PATHS:
            p = Path(p_str)
            if p.exists():
                discovered["kitti"] = p
                break

    yuv_root = _scan_yuv_root(kaggle_in)
    if yuv_root is not None:
        discovered["sfu"] = yuv_root
    else:
        for p_str in KAGGLE_SFU_YUV_PATHS:
            p = Path(p_str)
            if p.exists():
                discovered["sfu"] = p
                break

    return discovered


def find_mounted_kaggle_dataset(
    dataset_name: str = "mot17",
    search_root: Optional[Path] = None,
) -> Optional[Path]:
    """
    Finds a specific mounted VCM dataset inside `/kaggle/input` without cross-dataset collision
    when multiple datasets (e.g. `mot17challenge` + `mot-20` + `kitti-tracking`) are mounted simultaneously.

    Args:
        dataset_name: `"mot17"` (default, with fallback to any mounted VCM root if MOT17 is absent),
                      `"mot17_strict"`, `"mot20"`, `"kitti"`, `"sfu"`/`"hevc_b"`, `"second"`, or `"any"`.
        search_root: Optional override root for unit testing (defaults to `/kaggle/input`).
    """
    key = (dataset_name or "mot17").strip().lower()
    discovered = find_all_mounted_kaggle_datasets(search_root=search_root)

    if key == "mot17_strict":
        path = discovered.get("mot17")
    elif key == "mot17":
        path = (
            discovered.get("mot17")
            or discovered.get("mot_other")
            or discovered.get("mot20")
            or discovered.get("kitti")
            or discovered.get("sfu")
        )
    elif key == "mot20":
        path = discovered.get("mot20")
    elif key == "kitti":
        path = discovered.get("kitti")
    elif key in ("sfu", "hevc", "hevc_b"):
        path = discovered.get("sfu")
    elif key == "second":
        path = discovered.get("mot20") or discovered.get("kitti") or discovered.get("sfu")
    elif key in ("any", "auto"):
        path = (
            discovered.get("mot17")
            or discovered.get("mot20")
            or discovered.get("kitti")
            or discovered.get("sfu")
            or discovered.get("mot_other")
        )
    else:
        raise ValueError(f"Unsupported dataset_name={dataset_name!r}")

    if path is not None:
        print(f"[Dataset Auto-Detect] Resolved {key} root at: {path}")
    return path


def find_second_vcm_dataset(search_root: Optional[Path] = None) -> Optional[Path]:
    """Finds the mounted 2nd VCM benchmark dataset (`MOT20`, `KITTI Tracking`, or `HEVC-B/SFU`)."""
    return find_mounted_kaggle_dataset("second", search_root=search_root)


def download_sfu_hw_annotations(target_dir: Path, strict: bool = False) -> Optional[Path]:
    """
    Attempts to fetch SFU-HW-Objects-v1 / SFU-HW-Tracks tools if reachable.
    Non-fatal by default (`strict=False`) because official SFU-HW-Objects-v1 labels are hosted
    on Mendeley/SFU Vault rather than GitHub; `MOT20` (`ismailelbouknify/mot-20`) and `KITTI`
    (`leducnhuan/kitti-tracking`) carry complete in-dataset ground-truth annotations on Kaggle.
    """
    sfu_dir = Path(target_dir) / "SFU-HW-Objects-v1"
    if sfu_dir.exists() and any(sfu_dir.iterdir()):
        print(f"SFU-HW-Objects-v1 directory already present at {sfu_dir}")
        return sfu_dir

    res = subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/takehiro-code/SFU-HW-Tracks.git",
            str(sfu_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode == 0 and sfu_dir.exists():
        print(f"Cloned SFU-HW-Tracks utilities -> {sfu_dir}")
        return sfu_dir

    msg = (
        "[Dataset Prep] Warning: SFU-HW-Objects-v1 external label repository is unavailable via git; "
        "use mounted `ismailelbouknify/mot-20` or `leducnhuan/kitti-tracking` for self-contained 2nd-dataset evaluation."
    )
    if strict:
        raise RuntimeError(msg)
    print(msg)
    return None


def create_mot17_sample_structure(
    target_dir: Path,
    num_sequences: int = 4,
    frames_per_seq: int = 32,
    height: int = 384,
    width: int = 640,
) -> Path:
    """
    Creates a valid MOT17-formatted directory (`MOT17_sample/train/<seq>/img1/*.jpg` + `gt/gt.txt`)
    with 16:9 `640x384` frames and 32-frame GOP support for fast local CI smoke tests.
    """
    from PIL import Image
    import numpy as np

    mot_root = Path(target_dir) / "MOT17_sample" / "train"
    for seq_idx in range(1, num_sequences + 1):
        seq_dir = mot_root / f"MOT17-{seq_idx:02d}-FRCNN"
        img1_dir = seq_dir / "img1"
        gt_dir = seq_dir / "gt"
        img1_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(100 + seq_idx)
        bg = rng.integers(50, 140, size=(height, width, 3), dtype=np.uint8)

        gt_lines = []
        for f_idx in range(1, frames_per_seq + 1):
            frame = bg.copy()
            noise = rng.integers(-6, 7, size=(height, width, 3), dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            bw, bh = 56, 110
            max_x0 = max(1, width - bw - 8)
            x0 = min(max_x0, 24 + f_idx * 8)
            y0 = min(max(1, height - bh - 8), 60 + (seq_idx * 14))
            frame[y0 : y0 + bh, x0 : x0 + bw] = [210, 80, 60]

            img_path = img1_dir / f"{f_idx:06d}.jpg"
            Image.fromarray(frame).save(img_path, quality=95)
            gt_lines.append(f"{f_idx},1,{x0},{y0},{bw},{bh},1,1,1.0\n")

        with open(gt_dir / "gt.txt", "w", encoding="utf-8") as f:
            f.writelines(gt_lines)

    print(f"Prepared MOT17-formatted sample structure ({width}x{height}, {frames_per_seq} frames/seq) at: {mot_root}")
    return mot_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MOT17, MOT20, KITTI & SFU datasets on Kaggle")
    parser.add_argument("--target_dir", type=str, default="./data")
    parser.add_argument("--dataset_name", type=str, default="mot17", help="mot17 | mot20 | kitti | sfu | second | any")
    parser.add_argument("--download_sfu_labels", action="store_true")
    parser.add_argument("--init_mot_sample", action="store_true")
    parser.add_argument("--sample_frames", type=int, default=32)
    args = parser.parse_args()

    target = Path(args.target_dir)
    target.mkdir(parents=True, exist_ok=True)

    mounted = find_mounted_kaggle_dataset(args.dataset_name)
    if mounted is not None:
        print(f"Use `--data_root {mounted}` in runner scripts")

    second = find_second_vcm_dataset()
    if second is not None and second != mounted:
        print(f"Detected 2nd VCM dataset at: {second}")

    if args.download_sfu_labels:
        download_sfu_hw_annotations(target, strict=False)

    if args.init_mot_sample or (mounted is None and args.dataset_name == "mot17"):
        create_mot17_sample_structure(target, frames_per_seq=args.sample_frames)


if __name__ == "__main__":
    main()
