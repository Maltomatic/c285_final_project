"""
modal_run.py — general-purpose Modal launcher for six-wheel TD3 training.

Usage:
    modal run scripts/modal_run.py [options]

Run detached (close terminal safely):
    modal run --detach scripts/modal_run.py [options]

Examples:
    modal run --detach scripts/modal_run.py --exp-name fault_k1  --obs-stack 1  --wandb
    modal run --detach scripts/modal_run.py --exp-name fault_k3  --obs-stack 3  --wandb
    modal run --detach scripts/modal_run.py --exp-name fault_k10 --obs-stack 10 --wandb
    modal run --detach scripts/modal_run.py --exp-name pure_w2   --pure --num-fault-wheels 2 --wandb

Check running jobs:
    modal app list

Download results:
    modal volume get six-wheel-rl-volume fault_k10-td3_checkpoint.pth .
"""

import sys
from pathlib import Path

import modal

APP_NAME    = "six-wheel-fault-rl"
NETRC_PATH  = Path("~/.netrc").expanduser()
PROJECT_DIR = "/root/project"
VOLUME_PATH = "/root/vol"

# ── Hardware ──────────────────────────────────────────────────────────────────
# CPU is the real bottleneck (MuJoCo physics across parallel envs).
# H100 keeps the small TD3 MLP updates instant; 32 CPUs saturate the env pool.
GPU    = "A10G"
CPUS   = 32.0
MEMORY = 32768   # MB (32 GB) — replay buffer peaks ~8 GB for k=10, comfortable headroom

volume = modal.Volume.from_name("six-wheel-rl-volume", create_if_missing=True)


def _gitignore_patterns() -> list[str]:
    if not modal.is_local():
        return []
    root = Path(__file__).resolve().parents[1]
    gitignore = root / ".gitignore"
    patterns = ["**/.venv/**", "**/wandb/**", "**/__pycache__/**", "**/*.pth"]
    if gitignore.is_file():
        for line in gitignore.read_text(encoding="utf-8").splitlines():
            entry = line.strip()
            if not entry or entry.startswith("#") or entry.startswith("!"):
                continue
            entry = entry.lstrip("/")
            if entry.endswith("/"):
                patterns.append(f"**/{entry.rstrip('/')}/**")
            else:
                patterns.append(f"**/{entry}")
    return patterns


# ── Container image ───────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim()
    .apt_install("libgl1", "libglib2.0-0", "libosmesa6", "libegl1", "patchelf")
    .uv_sync()
)
if NETRC_PATH.is_file():
    image = image.add_local_file(NETRC_PATH, remote_path="/root/.netrc", copy=True)
image = image.add_local_dir(".", remote_path=PROJECT_DIR, ignore=_gitignore_patterns())

app = modal.App(APP_NAME)

_env = {
    "PYTHONPATH": PROJECT_DIR,
    "MUJOCO_GL": "osmesa",       # software GL — reliable on headless containers
    "PYOPENGL_PLATFORM": "osmesa",
}


# ── Remote training function ──────────────────────────────────────────────────
@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 10,
    image=image,
    gpu=GPU,
    cpu=CPUS,
    memory=MEMORY,
    env=_env,
)
def train_remote(*args: str) -> None:
    import os

    vol = Path(VOLUME_PATH)
    vol.mkdir(parents=True, exist_ok=True)
    (vol / "eval_logs").mkdir(exist_ok=True)
    os.chdir(str(vol))

    sys.path.insert(0, PROJECT_DIR)
    sys.argv = ["main.py"] + list(args)

    from main import main as train_main
    train_main()
    volume.commit()


# ── Local entrypoint ──────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(
    # core
    exp_name: str = "fault",
    obs_stack: int = 5,
    num_envs: int = 0,   # 0 = auto-detect (cpu_count()-1 on the remote machine)
    # model
    pure: bool = False,
    no_fault: bool = False,
    ft: bool = False,
    # eval flags (pass --eval to run in eval mode instead of training)
    eval: bool = False,
    ckpt_name: str = "",
    num_fault_wheels: int = 1,
    jitter_fault: bool = False,
    same_side: bool = False,
    # wandb
    wandb: bool = False,
    wandb_project: str = "285-final-project",
) -> None:
    args = [
        "--exp-name", exp_name,
        "--obs-stack", str(obs_stack),
    ]
    if num_envs > 0:    args += ["--num-envs", str(num_envs)]
    if pure:            args.append("--pure")
    if no_fault:        args.append("--no-fault")
    if ft:              args.append("--ft")
    if eval:            args.append("--eval")
    if ckpt_name:       args += ["--ckpt-name", ckpt_name]
    if num_fault_wheels != 1:
                        args += ["--num-fault-wheels", str(num_fault_wheels)]
    if jitter_fault:    args.append("--jitter-fault")
    if same_side:       args.append("--same-side")
    if wandb:           args += ["--wandb", "--wandb-project", wandb_project]

    print(f"Launching: main.py {' '.join(args)}")
    train_remote.spawn(*args)
