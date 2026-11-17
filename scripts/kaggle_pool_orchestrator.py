"""Master Orchestrator for Kaggle T4 GPU Pool (Push, Status, Pull & Manifest Collection for tscope-v3)."""
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.experiment_protocol import (
    SCHEMA_VERSION,
    VARIANTS,
    read_manifest_files,
    sha256_file,
    validate_soup_provenance,
)
from scripts.kaggle_output_download import download_kernel_outputs
from scripts.merge_shards import merge_checkpoints_soup, merge_eval_jsons_and_print_table

REPO_CLONE_URL = "https://github.com/wagur1/tscope-v3.git"
ABLATION_ROTATION: List[str] = list(VARIANTS)
SUPPORTED_RUN_MODES = (
    "train",
    "validate_soups",
    "eval_soup",
    "eval_sota_and_tracking",
    "eval_cross_backbone",
    "eval_poc",
    "both",
)


def validate_git_revision(revision: str) -> str:
    """Enforces a mandatory 40-char lowercase hexadecimal commit SHA (F22)."""
    rev = (revision or "").strip()
    if len(rev) != 40 or any(ch not in "0123456789abcdef" for ch in rev):
        raise ValueError(
            "A full 40-char lowercase hex --git_revision is mandatory for both train and eval* modes (F22)."
        )
    return rev


def resolve_use_yolov8s(run_mode: str, explicit_flag: Optional[bool] = None) -> bool:
    """
    Resolves --use_yolov8s per F04:
      - Defaults to False in `train` (single-teacher Faster R-CNN ResNet50-FPN v2 supervision, zero YOLO leakage).
      - Defaults to True in `eval*` / `validate_soups` modes.
    """
    if explicit_flag is not None:
        return bool(explicit_flag)
    return run_mode != "train"


def _redact_secrets(message: str, secrets: Sequence[str]) -> str:
    redacted = str(message)
    for secret in secrets:
        if secret and len(secret) >= 3:
            redacted = redacted.replace(secret, "[REDACTED]")
    redacted = re.sub(r"KGAT_[A-Za-z0-9_\-]+", "[REDACTED_TOKEN]", redacted)
    return redacted


def load_account_pool(pool_file: str) -> List[Tuple[str, str]]:
    path = Path(pool_file)
    if not path.exists():
        raise FileNotFoundError(f"Pool file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        return [(str(user).strip(), str(token).strip()) for user, token in raw.items()]
    if isinstance(raw, list):
        out: List[Tuple[str, str]] = []
        for item in raw:
            user = str(item.get("username", "")).strip()
            tok = str(item.get("key") or item.get("token") or "").strip()
            if user and tok:
                out.append((user, tok))
        return out
    raise ValueError("Unsupported pool JSON format")


def get_authenticated_api(username: str, token: str):
    """Authenticates a single Kaggle account in-process without writing credentials to disk."""
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = token
    os.environ["KAGGLE_API_TOKEN"] = token
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _kernel_slug_for_mode(run_mode: str, shard_id: int) -> str:
    if run_mode in ("train", "eval_soup", "eval_poc", "both"):
        return f"t-scope-v3-shard-{shard_id:02d}"
    mode_tag = run_mode.replace("_", "-")
    return f"t-scope-v3-{mode_tag}-{shard_id:02d}"


def build_worker_script(
    shard_id: int,
    num_shards: int,
    variant: str,
    epochs: int,
    batch_size: int,
    run_mode: str,
    git_revision: str,
    img_h: int = 384,
    img_w: int = 640,
    img_size: Optional[int] = None,
    clip_len: int = 4,
    gop_len: int = 32,
    bframes: int = 0,
    data_shard_id: int = -1,
    data_num_shards: int = -1,
    patience: int = 10,
    min_epochs: int = 12,
    lr: float = 1.5e-4,
    use_yolov8s: Optional[bool] = None,
    grad_accum_steps: int = 2,
    soup_subdir: str = "release_ckpts",
    sota_tasks: str = "all",
    eval_x264: bool = True,
    eval_ra: bool = True,
) -> str:
    """
    Generates the standalone Python script pushed to each Kaggle T4 GPU kernel.
    Clones https://github.com/wagur1/tscope-v3.git and strictly verifies the 40-char
    `--git_revision` checkout in BOTH `train` and all `eval*` modes (F22).
    """
    if run_mode not in SUPPORTED_RUN_MODES:
        raise ValueError(f"Unsupported run_mode={run_mode!r}; expected one of {SUPPORTED_RUN_MODES}")
    checked_revision = validate_git_revision(git_revision)
    if img_size is not None and img_h == 384 and img_w == 640 and img_size != 640:
        # Optional square override only when explicitly requested
        eff_h = int(img_size)
        eff_w = int(img_size)
    else:
        eff_h = int(img_h)
        eff_w = int(img_w)

    use_yolo_train = resolve_use_yolov8s("train", use_yolov8s)
    use_yolo_eval = resolve_use_yolov8s("eval_soup", use_yolov8s)
    needs_ultralytics = (
        use_yolo_train
        or (run_mode != "train" and use_yolo_eval)
        or run_mode in ("eval_sota_and_tracking", "eval_cross_backbone")
    )

    return f'''import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

SHARD_ID = {int(shard_id)}
NUM_SHARDS = {int(num_shards)}
VARIANT = {variant!r}
EPOCHS = {int(epochs)}
BATCH_SIZE = {int(batch_size)}
GRAD_ACCUM_STEPS = {int(grad_accum_steps)}
IMG_H = {eff_h}
IMG_W = {eff_w}
CLIP_LEN = {int(clip_len)}
GOP_LEN = {int(gop_len)}
BFRAMES = {int(bframes)}
RUN_MODE = {run_mode!r}
DATA_SHARD_ID = {int(data_shard_id)}
DATA_NUM_SHARDS = {int(data_num_shards)}
PATIENCE = {int(patience)}
MIN_EPOCHS = {int(min_epochs)}
LR = {float(lr)}
GIT_REVISION = {checked_revision!r}
USE_YOLOV8S_TRAIN = {use_yolo_train!r}
USE_YOLOV8S_EVAL = {use_yolo_eval!r}
NEEDS_ULTRALYTICS = {needs_ultralytics!r}
SOUP_SUBDIR = {soup_subdir!r}
SOTA_TASKS = {sota_tasks!r}
EVAL_X264 = {bool(eval_x264)!r}
EVAL_RA = {bool(eval_ra)!r}

if len(GIT_REVISION) != 40 or any(c not in "0123456789abcdef" for c in GIT_REVISION):
    raise RuntimeError("Mandatory 40-char hex GIT_REVISION check failed before clone")

if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
    subprocess.run(["apt-get", "update", "-qq"], check=False)
    subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], check=True)

pip_pkgs = ["scipy"]
if NEEDS_ULTRALYTICS:
    pip_pkgs.append("ultralytics")
if RUN_MODE == "eval_sota_and_tracking":
    pip_pkgs.extend(["gdown", "pyyaml"])
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *pip_pkgs], check=False)

repo_dir = Path("/tmp/tscope-v3")
if repo_dir.exists():
    shutil.rmtree(repo_dir)
subprocess.run(
    ["git", "clone", {REPO_CLONE_URL!r}, str(repo_dir)],
    check=True,
)
subprocess.run(["git", "checkout", "--detach", GIT_REVISION], cwd=str(repo_dir), check=True)
actual_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), text=True).strip()
if actual_revision != GIT_REVISION:
    raise RuntimeError(f"Pinned git revision mismatch: expected {{GIT_REVISION}}, got {{actual_revision}}")
print(f"[Provenance] Verified checkout wagur1/tscope-v3 @ {{actual_revision}} (mode={{RUN_MODE}}, Variant={{VARIANT}})", flush=True)

if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))

from scripts.prepare_kaggle_datasets import find_mounted_kaggle_dataset, find_second_vcm_dataset

mot17_root = find_mounted_kaggle_dataset("mot17")
second_root = find_second_vcm_dataset()
sfu_root = find_mounted_kaggle_dataset("sfu")

def _resolve_soup_dir() -> Path:
    kaggle_in = Path("/kaggle/input")
    if kaggle_in.exists():
        for cand in sorted(kaggle_in.rglob("tscope_soup_full_tscope.pt")):
            return cand.parent
    return repo_dir / SOUP_SUBDIR

def _resolve_constituent_dir():
    kaggle_in = Path("/kaggle/input")
    if kaggle_in.exists():
        for cand in sorted(kaggle_in.rglob("tscope_*_shard*of*.pt")):
            return cand.parent
    return None

env = os.environ.copy()
env["PYTHONPATH"] = str(repo_dir) + os.pathsep + env.get("PYTHONPATH", "")
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ckpt_path = f"/kaggle/working/tscope_{{VARIANT}}_shard{{SHARD_ID:02d}}of{{NUM_SHARDS:02d}}.pt"

if RUN_MODE in ("train", "both"):
    cmd_train = [
        sys.executable,
        str(repo_dir / "scripts" / "kaggle_shard_runner.py"),
        "--mode", "train",
        "--shard_id", str(SHARD_ID),
        "--num_shards", str(NUM_SHARDS),
        "--data_shard_id", str(DATA_SHARD_ID),
        "--data_num_shards", str(DATA_NUM_SHARDS),
        "--variant", VARIANT,
        "--img_h", str(IMG_H),
        "--img_w", str(IMG_W),
        "--clip_len", str(CLIP_LEN),
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--grad_accum_steps", str(GRAD_ACCUM_STEPS),
        "--lr", str(LR),
        "--patience", str(PATIENCE),
        "--min_epochs", str(MIN_EPOCHS),
        "--enable_x265_ste",
        "--use_pretrained_fpn",
        "--output_dir", "/kaggle/working",
    ]
    if mot17_root is not None:
        cmd_train.extend(["--data_root", str(mot17_root)])
    if USE_YOLOV8S_TRAIN:
        cmd_train.append("--use_yolov8s")
    subprocess.run(cmd_train, cwd=str(repo_dir), env=env, check=True)

if RUN_MODE in ("eval_poc", "both"):
    cmd_eval_poc = [
        sys.executable,
        str(repo_dir / "scripts" / "kaggle_shard_runner.py"),
        "--mode", "eval_poc",
        "--shard_id", str(SHARD_ID),
        "--num_shards", str(NUM_SHARDS),
        "--variant", VARIANT,
        "--img_h", str(IMG_H),
        "--img_w", str(IMG_W),
        "--gop_len", str(GOP_LEN),
        "--bframes", str(BFRAMES),
        "--ckpt", ckpt_path,
        "--use_pretrained_fpn",
        "--output_dir", "/kaggle/working",
    ]
    if mot17_root is not None:
        cmd_eval_poc.extend(["--data_root", str(mot17_root)])
    if USE_YOLOV8S_EVAL:
        cmd_eval_poc.append("--use_yolov8s")
    subprocess.run(cmd_eval_poc, cwd=str(repo_dir), env=env, check=True)

if RUN_MODE == "validate_soups":
    soup_dir = _resolve_soup_dir()
    constituent_dir = _resolve_constituent_dir()
    cmd_val = [
        sys.executable,
        str(repo_dir / "scripts" / "validate_soups.py"),
        "--data_root", str(mot17_root) if mot17_root is not None else "./data",
        "--soup_dir", str(soup_dir),
        "--output_dir", "/kaggle/working",
        "--img_h", str(IMG_H),
        "--img_w", str(IMG_W),
    ]
    if constituent_dir is not None:
        cmd_val.extend(["--constituent_dir", str(constituent_dir)])
    subprocess.run(cmd_val, cwd=str(repo_dir), env=env, check=True)

if RUN_MODE == "eval_soup":
    soup_dir = _resolve_soup_dir()
    assigned_shards = [SHARD_ID]
    cmd_eval = [
        sys.executable,
        str(repo_dir / "scripts" / "kaggle_shard_runner.py"),
        "--mode", "eval_all_variants",
        "--shard_ids", *[str(s) for s in assigned_shards],
        "--num_shards", str(NUM_SHARDS),
        "--img_h", str(IMG_H),
        "--img_w", str(IMG_W),
        "--gop_len", str(GOP_LEN),
        "--bframes", str(BFRAMES),
        "--soup_dir", str(soup_dir),
        "--use_pretrained_fpn",
        "--output_dir", "/kaggle/working",
    ]
    if mot17_root is not None:
        cmd_eval.extend(["--data_root", str(mot17_root)])
    if USE_YOLOV8S_EVAL:
        cmd_eval.append("--use_yolov8s")
    subprocess.run(cmd_eval, cwd=str(repo_dir), env=env, check=True)

if RUN_MODE == "eval_sota_and_tracking":
    soup_dir = _resolve_soup_dir()
    cmd_sota = [
        sys.executable,
        str(repo_dir / "scripts" / "eval_sota_and_tracking.py"),
        "--data_root", str(mot17_root) if mot17_root is not None else "auto",
        "--second_data_root", str(second_root) if second_root is not None else "auto",
        "--soup_dir", str(soup_dir),
        "--output_dir", "/kaggle/working",
        "--img_h", str(IMG_H),
        "--img_w", str(IMG_W),
        "--gop_len", str(GOP_LEN),
        "--bframes", str(BFRAMES),
        "--auto_download_scope",
        "--tasks", SOTA_TASKS,
        "--shard_id", str(SHARD_ID),
        "--num_shards", str(NUM_SHARDS),
    ]
    if sfu_root is not None:
        cmd_sota.extend(["--sfu_label_root", str(sfu_root)])
    subprocess.run(cmd_sota, cwd=str(repo_dir), env=env, check=True)

if RUN_MODE == "eval_cross_backbone":
    soup_dir = _resolve_soup_dir()
    cmd_cb = [
        sys.executable,
        str(repo_dir / "scripts" / "eval_cross_backbone_bdbr.py"),
        "--data_root", str(mot17_root) if mot17_root is not None else "auto",
        "--soup_dir", str(soup_dir),
        "--output_dir", "/kaggle/working",
        "--img_h", str(IMG_H),
        "--img_w", str(IMG_W),
        "--gop_len", str(GOP_LEN),
    ]
    if EVAL_X264:
        cmd_cb.append("--eval_x264")
    if EVAL_RA:
        cmd_cb.append("--eval_ra")
    subprocess.run(cmd_cb, cwd=str(repo_dir), env=env, check=True)

print(f"=== Shard {{SHARD_ID}}/{{NUM_SHARDS}} ({{VARIANT}}) finished successfully! ===", flush=True)
'''


def verify_pool(accounts: List[Tuple[str, str]], min_gpu_hours: float = 8.0) -> None:
    print(f"Verifying {len(accounts)} Kaggle accounts in pool (credentials masked)...")
    ok_count = 0
    high_quota_count = 0
    total_rem_h = 0.0
    for idx, (user, token) in enumerate(accounts):
        try:
            api = get_authenticated_api(user, token)
            q = api.quota_view()
            used_h = q.gpu_quota.time_used.total_seconds() / 3600.0
            tot_h = q.gpu_quota.total_time_allowed.total_seconds() / 3600.0
            rem_h = max(0.0, tot_h - used_h)
            total_rem_h += rem_h
            ok_count += 1
            if rem_h >= min_gpu_hours:
                high_quota_count += 1
            print(f"  [Account {idx:02d}/{len(accounts):02d}] AUTH_OK | GPU remaining: {rem_h:5.2f}h / {tot_h:4.1f}h")
        except Exception as exc:
            safe_err = _redact_secrets(type(exc).__name__, (user, token))
            print(f"  [Account {idx:02d}/{len(accounts):02d}] AUTH_FAIL ({safe_err})")
    print(
        f"\nPool Summary: {ok_count}/{len(accounts)} authenticated, "
        f"{high_quota_count} with >={min_gpu_hours:.1f}h GPU, {total_rem_h:.2f}h total GPU remaining."
    )


def _verify_local_git_and_soups_before_push(args: argparse.Namespace) -> str:
    revision = validate_git_revision(args.git_revision)
    if not getattr(args, "allow_non_head_revision", False):
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True).strip()
        if revision != head:
            raise ValueError(f"Pinned --git_revision ({revision}) must match local reviewed HEAD ({head})")
        dirty = subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=ROOT_DIR, text=True).strip()
        if dirty:
            raise ValueError("Commit and push local tracked changes before dispatching Kaggle kernels")

    if args.run_mode == "eval_soup" and not getattr(args, "skip_local_soup_check", False):
        import torch

        for variant in VARIANTS:
            path = ROOT_DIR / args.soup_subdir / f"tscope_soup_{variant}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Missing validated soup checkpoint for eval_soup: {path}")
            soup = torch.load(path, map_location="cpu", weights_only=True)
            validate_soup_provenance(soup, variant, require_validation=True)
            published = subprocess.check_output(
                ["git", "show", f"{revision}:{args.soup_subdir}/{path.name}"], cwd=ROOT_DIR
            )
            if hashlib.sha256(published).hexdigest() != sha256_file(path):
                raise ValueError(f"Local soup checkpoint differs from pinned git revision: {path.name}")
    return revision


def _resolve_dataset_sources(args: argparse.Namespace) -> List[str]:
    sources: List[str] = []
    if args.dataset_slug:
        for part in str(args.dataset_slug).split(","):
            slug = part.strip()
            if slug and slug not in sources:
                sources.append(slug)
    extra = getattr(args, "extra_dataset_slugs", None) or []
    for raw in extra:
        for part in str(raw).split(","):
            slug = part.strip()
            if slug and slug not in sources:
                sources.append(slug)
    if args.run_mode == "eval_sota_and_tracking" and not extra:
        for second_slug in ("ismailelbouknify/mot-20", "leducnhuan/kitti-tracking"):
            if second_slug not in sources:
                sources.append(second_slug)
    return sources


def _dispatch_state_path(output_dir: Path, run_mode: str) -> Path:
    return output_dir / f"dispatch_{run_mode}.json"


def select_balanced_active_accounts(
    active_accounts: List[Tuple[int, str, str, float]],
    run_mode: str,
    ablation_sweep: bool,
    expected_shards: int = 0,
    allow_unbalanced_shards: bool = False,
) -> Tuple[List[Tuple[int, str, str, float]], List[Tuple[int, str, str, float]]]:
    """
    Selects the exact shard cohort and returns `(selected_for_dispatch, reserved_for_parallel_eval)`.
    In `train` + `ablation_sweep`, enforces strictly equal shard counts across all 5 `VARIANTS`
    (e.g. 20 = 4x5 or 15 = 3x5, F07/F15) and reserves remaining GPU accounts for parallel SOTA/complexity jobs.
    """
    n_variants = len(ABLATION_ROTATION)
    if expected_shards > 0:
        if run_mode in ("train", "both") and ablation_sweep and not allow_unbalanced_shards:
            if expected_shards % n_variants != 0:
                raise ValueError(
                    f"--expected_shards={expected_shards} must be a multiple of {n_variants} "
                    f"for balanced variant training (e.g. {n_variants * 3} or {n_variants * 4})."
                )
        if len(active_accounts) < expected_shards:
            raise ValueError(
                f"Requested --expected_shards={expected_shards}, but only {len(active_accounts)} "
                f"accounts meet the GPU quota threshold."
            )
        return active_accounts[:expected_shards], active_accounts[expected_shards:]

    if run_mode in ("train", "both") and ablation_sweep and not allow_unbalanced_shards:
        balanced_n = (len(active_accounts) // n_variants) * n_variants
        if balanced_n == 0:
            raise ValueError(
                f"Need at least {n_variants} active GPU accounts for balanced {n_variants}-variant sweep; "
                f"found {len(active_accounts)}."
            )
        # Cap default balanced training sweep at 4 shards x 5 variants = 20 so remaining accounts stay free for parallel eval
        if balanced_n > n_variants * 4:
            balanced_n = n_variants * 4
        return active_accounts[:balanced_n], active_accounts[balanced_n:]

    if run_mode in ("validate_soups", "eval_cross_backbone") and expected_shards == 0:
        return active_accounts[:1], active_accounts[1:]

    return active_accounts, []


def push_all_shards(args: argparse.Namespace, accounts: List[Tuple[str, str]]) -> None:
    revision = _verify_local_git_and_soups_before_push(args)
    pool_size = len(accounts)
    scan_limit = min(pool_size, args.max_shards) if args.max_shards > 0 else pool_size
    dataset_sources = _resolve_dataset_sources(args)

    print(
        f"Scanning {scan_limit}/{pool_size} pool accounts "
        f"(mode={args.run_mode}, rev={revision[:12]}, datasets={dataset_sources}, min_gpu_hours={args.min_gpu_hours:.2f}h)..."
    )

    only_set = set(args.only_shards) if getattr(args, "only_shards", None) else None
    active_accounts: List[Tuple[int, str, str, float]] = []
    for acc_idx in range(scan_limit):
        if only_set is not None and acc_idx not in only_set:
            continue
        user, token = accounts[acc_idx]
        try:
            api = get_authenticated_api(user, token)
            q = api.quota_view()
            used_h = q.gpu_quota.time_used.total_seconds() / 3600.0
            tot_h = q.gpu_quota.total_time_allowed.total_seconds() / 3600.0
            rem_h = max(0.0, tot_h - used_h)
            if args.skip_exhausted and rem_h < args.min_gpu_hours:
                print(
                    f"  [Account {acc_idx:02d}] SKIP (GPU remaining {rem_h:5.2f}h < {args.min_gpu_hours:.2f}h)"
                )
                continue
            active_accounts.append((acc_idx, user, token, rem_h))
        except Exception as exc:
            safe_err = _redact_secrets(type(exc).__name__, (user, token))
            print(f"  [Account {acc_idx:02d}] Quota check failed ({safe_err})")

    if not active_accounts:
        raise RuntimeError("No active GPU accounts with sufficient quota found")

    selected_accounts, reserved_accounts = select_balanced_active_accounts(
        active_accounts=active_accounts,
        run_mode=args.run_mode,
        ablation_sweep=bool(args.ablation_sweep),
        expected_shards=int(args.expected_shards),
        allow_unbalanced_shards=bool(getattr(args, "allow_unbalanced_shards", False)),
    )
    if reserved_accounts:
        res_summary = ", ".join(f"acc_{idx:02d}({rem:.1f}h)" for idx, _, _, rem in reserved_accounts)
        print(f"  [Pool Reserve] {len(reserved_accounts)} active GPU account(s) reserved for parallel SOTA/eval: {res_summary}")

    num_shards = len(selected_accounts)
    active_variants = [
        ABLATION_ROTATION[idx % len(ABLATION_ROTATION)]
        if (args.ablation_sweep and args.run_mode in ("train", "both", "eval_poc"))
        else args.variant
        for idx in range(num_shards)
    ]

    task_split = [t.strip() for t in str(getattr(args, "sota_task_split", "")).split(",") if t.strip()]
    dispatch_records: List[Dict[str, Any]] = []

    for logical_shard_id, (acc_idx, user, token, rem_h) in enumerate(selected_accounts):
        variant = active_variants[logical_shard_id]
        same_variant_indices = [i for i in range(num_shards) if active_variants[i] == variant]
        data_shard_id = same_variant_indices.index(logical_shard_id)
        data_num_shards = len(same_variant_indices)
        slug = _kernel_slug_for_mode(args.run_mode, logical_shard_id)
        kernel_id = f"{user}/{slug}"
        shard_sota_tasks = (
            task_split[logical_shard_id % len(task_split)]
            if (args.run_mode == "eval_sota_and_tracking" and task_split)
            else args.sota_tasks
        )
        eff_num_shards_arg = 1 if (args.run_mode == "eval_sota_and_tracking" and task_split) else num_shards
        eff_shard_id_arg = 0 if (args.run_mode == "eval_sota_and_tracking" and task_split) else logical_shard_id

        try:
            api = get_authenticated_api(user, token)
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                worker_code = build_worker_script(
                    shard_id=eff_shard_id_arg,
                    num_shards=eff_num_shards_arg,
                    variant=variant,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    run_mode=args.run_mode,
                    git_revision=revision,
                    img_h=args.img_h,
                    img_w=args.img_w,
                    clip_len=args.clip_len,
                    gop_len=args.gop_len,
                    bframes=args.bframes,
                    data_shard_id=data_shard_id,
                    data_num_shards=data_num_shards,
                    patience=args.patience,
                    min_epochs=args.min_epochs,
                    lr=args.lr,
                    use_yolov8s=args.use_yolov8s,
                    grad_accum_steps=args.grad_accum_steps,
                    soup_subdir=args.soup_subdir,
                    sota_tasks=shard_sota_tasks,
                    eval_x264=not args.no_eval_x264,
                    eval_ra=not args.no_eval_ra,
                )
                (tmp_path / "worker.py").write_text(worker_code, encoding="utf-8")

                metadata = {
                    "id": kernel_id,
                    "title": f"T-SCOPE v3 {args.run_mode.replace('_', ' ').title()} {logical_shard_id:02d}",
                    "code_file": "worker.py",
                    "language": "python",
                    "kernel_type": "script",
                    "is_private": "true",
                    "enable_gpu": "true",
                    "enable_tpu": "false",
                    "enable_internet": "true",
                    "dataset_sources": dataset_sources,
                    "competition_sources": [],
                    "kernel_sources": [],
                    "model_sources": [],
                }
                (tmp_path / "kernel-metadata.json").write_text(
                    json.dumps(metadata, indent=2), encoding="utf-8"
                )

                res = api.kernels_push(str(tmp_path))
                err_msg = _redact_secrets(getattr(res, "error", "") or "", (user, token))
                print(
                    f"  [Shard {logical_shard_id:02d}/{num_shards:02d} -> Account {acc_idx:02d}] "
                    f"Pushed ({rem_h:5.2f}h rem) | slug={slug} | variant={variant:<18} "
                    f"(slice {data_shard_id + 1}/{data_num_shards}) {err_msg}"
                )
                dispatch_records.append(
                    {
                        "shard_id": logical_shard_id,
                        "account_index": acc_idx,
                        "kernel_slug": slug,
                        "variant": variant,
                        "data_shard_id": data_shard_id,
                        "data_num_shards": data_num_shards,
                        "sota_tasks": shard_sota_tasks if args.run_mode == "eval_sota_and_tracking" else None,
                    }
                )
                time.sleep(2.5)
        except Exception as exc:
            safe_err = _redact_secrets(type(exc).__name__, (user, token))
            print(f"  [Shard {logical_shard_id:02d}/{num_shards:02d} -> Account {acc_idx:02d}] ERROR ({safe_err})")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dispatch_payload = {
        "schema_version": SCHEMA_VERSION,
        "run_mode": args.run_mode,
        "git_revision": revision,
        "expected_shards": num_shards,
        "dispatched_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_sources": dataset_sources,
        "assignments": dispatch_records,
    }
    state_file = _dispatch_state_path(out_dir, args.run_mode)
    state_file.write_text(json.dumps(dispatch_payload, indent=2), encoding="utf-8")
    print(f"Saved credential-free dispatch map -> {state_file}")


def _load_assignments(args: argparse.Namespace, accounts: List[Tuple[str, str]]) -> Tuple[int, Optional[str], List[Dict[str, Any]]]:
    out_dir = Path(args.output_dir).resolve()
    state_file = Path(args.dispatch_map).resolve() if getattr(args, "dispatch_map", "") else _dispatch_state_path(out_dir, args.run_mode)
    if state_file.is_file():
        payload = json.loads(state_file.read_text(encoding="utf-8"))
        assignments = list(payload.get("assignments", []))
        expected = int(payload.get("expected_shards") or len(assignments))
        if args.expected_shards > 0 and args.expected_shards != expected:
            raise ValueError(f"--expected_shards={args.expected_shards} conflicts with dispatch map ({expected})")
        return expected, payload.get("git_revision"), assignments

    expected = int(args.expected_shards) if args.expected_shards > 0 else (
        min(len(accounts), args.max_shards) if args.max_shards > 0 else len(accounts)
    )
    if expected < 1 or expected > len(accounts):
        raise ValueError(f"Invalid expected_shards={expected} for pool of size {len(accounts)}")
    assignments = [
        {
            "shard_id": sid,
            "account_index": sid,
            "kernel_slug": _kernel_slug_for_mode(args.run_mode, sid),
            "variant": ABLATION_ROTATION[sid % len(ABLATION_ROTATION)],
        }
        for sid in range(expected)
    ]
    rev = args.git_revision.strip() if getattr(args, "git_revision", "") else None
    return expected, rev, assignments


def check_status_all(args: argparse.Namespace, accounts: List[Tuple[str, str]]) -> None:
    expected_shards, _, assignments = _load_assignments(args, accounts)
    print(f"Checking kernel status across {len(assignments)}/{expected_shards} assigned shard(s) (mode={args.run_mode})...")
    for item in assignments:
        sid = int(item["shard_id"])
        acc_idx = int(item["account_index"])
        slug = str(item["kernel_slug"])
        user, token = accounts[acc_idx]
        kernel_id = f"{user}/{slug}"
        try:
            api = get_authenticated_api(user, token)
            st = api.kernels_status(kernel_id)
            status_str = str(getattr(st, "status", st))
            msg = _redact_secrets(getattr(st, "failureMessage", "") or "", (user, token))
            print(f"  [Shard {sid:02d} | Account {acc_idx:02d}] {slug:<34} : {status_str} {msg}")
        except Exception as exc:
            safe_err = _redact_secrets(type(exc).__name__, (user, token))
            print(f"  [Shard {sid:02d} | Account {acc_idx:02d}] {slug:<34} : ERROR ({safe_err})")


def _collect_dir_files(directory: Path) -> List[Dict[str, Any]]:
    return [
        {
            "path": path.relative_to(directory).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]


def _resolve_manifest_entry_dir(collection: Path, dir_str: str) -> Path:
    raw = Path(dir_str)
    return raw.resolve() if raw.is_absolute() else (collection / raw).resolve()


def _route_backup_eval_outputs(manifest: Dict[str, Any], collection: Path, num_shards: int) -> None:
    """Moves recovered backup eval_shard_<sid>of<N>_*.json outputs to each target shard entry."""
    pattern = re.compile(rf"^eval_shard_(\d+)of{num_shards:02d}_.+\.json$")
    entry_by_id = {int(entry["shard_id"]): entry for entry in manifest["shards"]}
    touched_shards = set()

    for host_entry in manifest["shards"]:
        host_dir_str = host_entry.get("directory")
        if not host_dir_str:
            continue
        host_dir = _resolve_manifest_entry_dir(collection, host_dir_str)
        host_sid = int(host_entry["shard_id"])
        for path in sorted(host_dir.rglob("*.json")):
            if not path.is_file() or path.is_symlink():
                continue
            match = pattern.match(path.name)
            if not match:
                continue
            target_sid = int(match.group(1))
            if target_sid == host_sid or target_sid not in entry_by_id:
                continue
            target_entry = entry_by_id[target_sid]
            target_dir_str = target_entry.get("directory")
            if not target_dir_str:
                target_dir = Path(tempfile.mkdtemp(prefix=f"shard_{target_sid:02d}_", dir=collection))
                target_entry["directory"] = target_dir.relative_to(collection).as_posix()
            else:
                target_dir = _resolve_manifest_entry_dir(collection, target_dir_str)
            target_entry["downloaded"] = True
            target_entry["status"] = "KernelWorkerStatus.COMPLETE"
            dest = target_dir / path.name
            if dest.exists():
                path.unlink()
            else:
                shutil.move(str(path), str(dest))
                recovered = target_entry.setdefault("recovered_from_shards", [])
                if host_sid not in recovered:
                    recovered.append(host_sid)
            touched_shards.add(host_sid)
            touched_shards.add(target_sid)

    for sid in sorted(touched_shards):
        entry = entry_by_id[sid]
        resolved_dir = _resolve_manifest_entry_dir(collection, entry["directory"])
        entry["files"] = _collect_dir_files(resolved_dir)
        if entry.get("recovered_from_shards"):
            print(
                f"  [Shard {sid:02d}] Routed {len(entry['files'])} files from backup shard(s) {entry['recovered_from_shards']}"
            )


def _seed_from_prior_manifest(
    manifest: Dict[str, Any],
    collection: Path,
    seed_manifest_path: Optional[Path],
    num_shards: int,
) -> None:
    """Seeds missing shard eval JSONs from a prior verified collection_manifest.json."""
    if not seed_manifest_path:
        return
    resolved_seed = Path(seed_manifest_path).resolve()
    _, prior_files = read_manifest_files(resolved_seed, ".json")
    pattern = re.compile(rf"^eval_shard_(\d+)of{num_shards:02d}_.+\.json$")
    entry_by_id = {int(entry["shard_id"]): entry for entry in manifest["shards"]}
    seeded_shards = set()

    for prior_entry, prior_path, _ in prior_files:
        match = pattern.match(prior_path.name)
        if not match:
            continue
        target_sid = int(match.group(1))
        if target_sid != int(prior_entry["shard_id"]) or target_sid not in entry_by_id:
            continue
        target_entry = entry_by_id[target_sid]
        target_dir_str = target_entry.get("directory")
        if not target_dir_str:
            target_dir = Path(tempfile.mkdtemp(prefix=f"shard_{target_sid:02d}_", dir=collection))
            target_entry["directory"] = target_dir.relative_to(collection).as_posix()
        else:
            target_dir = _resolve_manifest_entry_dir(collection, target_dir_str)
        dest = target_dir / prior_path.name
        if not dest.exists():
            shutil.copy2(str(prior_path), str(dest))
            target_entry["downloaded"] = True
            target_entry["status"] = "KernelWorkerStatus.COMPLETE"
            target_entry["seeded_from_manifest"] = resolved_seed.name
            seeded_shards.add(target_sid)

    for sid in sorted(seeded_shards):
        entry = entry_by_id[sid]
        resolved_dir = _resolve_manifest_entry_dir(collection, entry["directory"])
        entry["files"] = _collect_dir_files(resolved_dir)
        print(f"  [Shard {sid:02d}] Seeded outputs from prior manifest -> {len(entry['files'])} files")


def pull_and_merge_all(args: argparse.Namespace, accounts: List[Tuple[str, str]]) -> Path:
    expected_shards, git_revision, assignments = _load_assignments(args, accounts)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    collection = Path(tempfile.mkdtemp(prefix=f"collection_{args.run_mode}_", dir=out_dir))
    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_mode": args.run_mode,
        "expected_shards": expected_shards,
        "git_revision": git_revision,
        "collected_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "shards": [],
    }
    print(
        f"Collecting completed outputs ({expected_shards} expected shard(s), mode={args.run_mode}) "
        f"with collection-relative paths under {collection}..."
    )

    for item in assignments:
        shard_id = int(item["shard_id"])
        acc_idx = int(item["account_index"])
        slug = str(item["kernel_slug"])
        expected_variant = str(item.get("variant") or ABLATION_ROTATION[shard_id % len(ABLATION_ROTATION)])
        user, token = accounts[acc_idx]
        kernel_id = f"{user}/{slug}"
        entry: Dict[str, Any] = {
            "shard_id": shard_id,
            "account_index": acc_idx,
            "kernel_slug": slug,
            "expected_variant": expected_variant,
            "downloaded": False,
        }
        try:
            api = get_authenticated_api(user, token)
            status = api.kernels_status(kernel_id)
            entry["status"] = str(getattr(status, "status", status))
            if entry["status"].upper().split(".")[-1] != "COMPLETE":
                print(f"  [Shard {shard_id:02d} | Account {acc_idx:02d}] {entry['status']}; skipped")
            else:
                destination = Path(tempfile.mkdtemp(prefix=f"shard_{shard_id:02d}_", dir=collection))
                # Store portable collection-relative directory path (F22 fix)
                entry["directory"] = destination.relative_to(collection).as_posix()
                download_kernel_outputs(api, kernel_id, destination)
                entry["files"] = _collect_dir_files(destination)
                entry["downloaded"] = True
                print(f"  [Shard {shard_id:02d} | Account {acc_idx:02d}] Downloaded {len(entry['files'])} files")
        except Exception as error:
            entry["error_type"] = type(error).__name__
            print(
                f"  [Shard {shard_id:02d} | Account {acc_idx:02d}] Collection failed: "
                f"{type(error).__name__}; credentials not logged"
            )
        manifest["shards"].append(entry)
        (collection / "collection_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        time.sleep(1.5)

    _route_backup_eval_outputs(manifest, collection, expected_shards)
    if getattr(args, "seed_manifest", ""):
        _seed_from_prior_manifest(manifest, collection, Path(args.seed_manifest), expected_shards)
    manifest_path = collection / "collection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Collection manifest -> {manifest_path}")
    print("Use scripts/merge_shards.py with the checked manifest for audited merge.")
    return manifest_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Master Orchestrator for Kaggle T4 GPU Pool (tscope-v3)"
    )
    parser.add_argument(
        "--pool_file",
        type=str,
        default=os.environ.get("KAGGLE_POOL_FILE", r"c:\Users\Wagur1\Desktop\pool (1).json"),
        help="Path to JSON mapping {username: KGAT_token}",
    )
    parser.add_argument(
        "--action",
        choices=["verify_pool", "push", "status", "pull", "pull_and_merge"],
        default="verify_pool",
    )
    parser.add_argument(
        "--run_mode",
        choices=list(SUPPORTED_RUN_MODES),
        default="train",
        help="Execution mode dispatched to Kaggle T4 GPU kernels",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=list(VARIANTS),
        default="full_tscope",
        help="Ablation variant when --ablation_sweep is False",
    )
    parser.add_argument(
        "--ablation_sweep",
        action="store_true",
        help="Assigns all 5 de-confounded VARIANTS with strictly equal shards/variant (F07/F15)",
    )
    parser.add_argument(
        "--expected_shards",
        "--expected-shards",
        dest="expected_shards",
        type=int,
        default=0,
        help="Parameterized total shard count (e.g. 20 = 4x5 or 15 = 3x5; 0 = auto-balanced, F22)",
    )
    parser.add_argument(
        "--allow_unbalanced_shards",
        action="store_true",
        help="Allow non-multiple-of-5 shard counts during ablation_sweep",
    )
    parser.add_argument(
        "--dataset_slug",
        type=str,
        default="ahmedsamir1598/mot17challenge",
        help="Primary Kaggle dataset slug mounted in each kernel",
    )
    parser.add_argument(
        "--extra_dataset_slugs",
        type=str,
        nargs="*",
        default=[],
        help="Additional Kaggle dataset slugs to mount (e.g. ismailelbouknify/mot-20 leducnhuan/kitti-tracking)",
    )
    parser.add_argument("--epochs", type=int, default=36, help="Max epochs (upper bound when --patience > 0)")
    parser.add_argument("--patience", type=int, default=10, help="Early-stop patience on MOT17-09 val loss")
    parser.add_argument("--min_epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--batch_size", type=int, default=2, help="Per-GPU batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--img_h", type=int, default=384, help="16:9 aspect-preserving height (divisible by 32)")
    parser.add_argument("--img_w", type=int, default=640, help="16:9 aspect-preserving width (divisible by 32)")
    parser.add_argument("--clip_len", type=int, default=4, help="Training clip length")
    parser.add_argument("--gop_len", type=int, default=32, help="Evaluation GOP / IntraPeriod length (F05)")
    parser.add_argument("--bframes", type=int, default=0, help="0 = Low-Delay P (IPPP), 4 = Random Access (IBBP)")
    parser.add_argument("--max_shards", type=int, default=0, help="0 = scan all pool accounts")
    parser.add_argument(
        "--skip_exhausted",
        action="store_true",
        default=True,
        help="Skip accounts below --min_gpu_hours",
    )
    parser.add_argument(
        "--min_gpu_hours",
        type=float,
        default=8.0,
        help="Minimum remaining weekly GPU hours required to dispatch a kernel",
    )
    parser.add_argument(
        "--use_yolov8s",
        dest="use_yolov8s",
        action="store_true",
        default=None,
        help="Force enable YOLOv8s (defaults to False in train [F04] and True in eval*)",
    )
    parser.add_argument(
        "--no_use_yolov8s",
        dest="use_yolov8s",
        action="store_false",
        help="Force disable YOLOv8s",
    )
    parser.add_argument(
        "--git_revision",
        type=str,
        default="",
        help="Mandatory 40-char commit SHA verified in both train and eval* kernels (F22)",
    )
    parser.add_argument(
        "--allow_non_head_revision",
        action="store_true",
        help="Allow dispatching a historical published commit different from local HEAD",
    )
    parser.add_argument(
        "--skip_local_soup_check",
        action="store_true",
        help="Skip local release_ckpts check when soups are mounted via Kaggle dataset",
    )
    parser.add_argument(
        "--soup_subdir",
        type=str,
        default="release_ckpts",
        help="Repository subdirectory containing soup checkpoints",
    )
    parser.add_argument(
        "--sota_tasks",
        type=str,
        default="all",
        help="--tasks value for eval_sota_and_tracking (all|sota|tracking|second_dataset|complexity)",
    )
    parser.add_argument(
        "--sota_task_split",
        type=str,
        default="",
        help="Optional comma-separated task list split across parallel eval_sota_and_tracking kernels",
    )
    parser.add_argument("--no_eval_x264", action="store_true", help="Skip libx264 pass in eval_cross_backbone")
    parser.add_argument("--no_eval_ra", action="store_true", help="Skip Random Access pass in eval_cross_backbone")
    parser.add_argument("--only_shards", type=int, nargs="+", default=None, help="Optional subset of pool account indices")
    parser.add_argument("--dispatch_map", type=str, default="", help="Optional path to dispatch_<mode>.json")
    parser.add_argument("--seed_manifest", type=str, default="", help="Optional prior collection_manifest.json")
    parser.add_argument("--output_dir", type=str, default="./outputs_kaggle_pool")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    accounts = load_account_pool(args.pool_file)
    if args.action == "verify_pool":
        verify_pool(accounts, min_gpu_hours=args.min_gpu_hours)
    elif args.action == "push":
        push_all_shards(args, accounts)
    elif args.action == "status":
        check_status_all(args, accounts)
    elif args.action in ("pull", "pull_and_merge"):
        pull_and_merge_all(args, accounts)


if __name__ == "__main__":
    main()
