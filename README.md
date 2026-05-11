# History-Conditioned Residual RL for Fault-Tolerant 6-Wheel Robot Control

A MuJoCo + Gymnasium simulation environment for training a residual RL policy that compensates for wheel faults on a 6-wheel skid-steer robot.

---

## Concept

The robot follows a fixed waypoint trajectory using a simple **base controller** (proportional heading + differential drive). On top of that, an **RL policy** outputs small wheel velocity corrections (`delta_omega`) to compensate when one wheel degrades.

The policy never observes the fault directly — it must infer it from the gap between commanded and actual wheel velocities, across multiple timesteps.

```
WaypointController → (v, ω)
     → BaseAllocator → omega_base[6]     (nominal wheel cmds)
     → (+) delta_omega                   (RL residual action)
     → fault_fn                          (hidden degradation)
     → MuJoCo data.ctrl[6]
```

---

## File Structure

```
c285_final_project/
├── assets/
│   └── robot.xml                   MuJoCo MJCF model
├── controllers/
│   ├── TD3_controller.py           TD3 agent (history-stacked obs, replay buffer)
│   ├── base_controller.py          WaypointController + BaseAllocator
│   └── utils/
│       └── model_configs.py        TD3 hyperparameters (G_STEPS, discount, etc.)
├── envs/
│   ├── __init__.py                 gym.register call
│   ├── env_configs.py              Central config (all runtime flags live here)
│   ├── six_wheel_env.py            Gymnasium environment
│   └── rewards.py                  Reward functions
├── eval_logs/
│   ├── ablation_grapher.py         Plot ablation study results
│   ├── history_grapher.py          Plot history window ablation results
│   └── eval_grapher.py             Per-alpha / per-wheel breakdown plots
├── scripts/
│   └── modal_run.py                Modal cloud training launcher
├── run_ablation.sh                 Run all 9 eval ablation experiments
├── run_history_ablation.sh         Train 3 history-window models (k=1,3,10)
├── run_history_eval.sh             Eval the 3 history-window checkpoints
└── main.py                         Training + eval entrypoint
```

---

## Setup

```bash
# requires uv (https://github.com/astral-sh/uv)
uv sync
```

---

## Robot Model (`assets/robot.xml`)

- **Chassis:** 1.0 m × 0.6 m × 0.2 m box, 20 kg, freejoint (6-DOF)
- **6 wheels:** cylinder radius=0.15 m, half-width=0.04 m, 1.5 kg each
  - Left side (y=+0.35): FL, ML, RL at x = +0.38, 0.00, −0.38
  - Right side (y=−0.35): FR, MR, RR at x = +0.38, 0.00, −0.38
- **Actuators:** velocity-controlled (`kv=100`), `ctrl[0..5]` = [FL, ML, RL, FR, MR, RR]
- **Physics:** `timestep=0.002 s`, `integrator=implicitfast`

**MuJoCo state layout:**

| Array | Indices | Content |
|---|---|---|
| `qpos` | 0:3 | chassis x, y, z |
| `qpos` | 3:7 | chassis quaternion (w, x, y, z) |
| `qpos` | 7:13 | wheel hinge angles (unused) |
| `qvel` | 0:6 | chassis linear + angular velocity (world frame) |
| `qvel` | 6:12 | wheel angular velocities (rad/s) |
| `ctrl` | 0:6 | wheel velocity commands (rad/s) |

---

## Base Controller (`controllers/base_controller.py`)

### WaypointController

Proportional heading controller that outputs `(v, omega)`:

```python
heading_error = wrap_to_pi(atan2(dy, dx) - heading)
omega = clip(kp * heading_error, ±omega_max)   # kp=2.0, omega_max=2.0
v     = v_max * max(0, cos(heading_error))      # v_max=1.5 m/s
```

Advances to the next waypoint when within `arrival_radius=0.3 m`. Returns `done=True` when all waypoints are reached.

**Default trajectory** (L-shape + return, ~22 m total):

```
(0,0) → (5,0) → (5,5) → (0,5) → (0,0)
```

### BaseAllocator

Converts `(v, omega)` to 6 wheel commands using differential-drive kinematics:

```python
v_left  = v - omega * track_width/2    # track_width = 0.70 m
v_right = v + omega * track_width/2
omega_wheel = v_side / wheel_radius    # wheel_radius = 0.15 m
```

Left wheels = indices 0,1,2. Right wheels = indices 3,4,5.

---

## Gymnasium Environment (`envs/six_wheel_env.py`)

**Registration ID:** `"SixWheelSkidSteer-v0"`

### Timing

| Level | Duration |
|---|---|
| MuJoCo timestep | 0.002 s |
| Env step (`_FRAME_SKIP=10`) | 0.02 s (50 Hz control) |
| Max episode (`max_episode_steps=2000`) | 40 s simulated |

### Observation Space

23 values per timestep, stacked over the last **K timesteps** (default K=5, i.e. 100 ms of history):

| Index | Signal | Why it's there |
|---|---|---|
| 0 | cross-track error (m) | signed lateral deviation from path |
| 1 | heading error (rad) | angular error to next waypoint |
| 2 | dist to waypoint (m) | how far to go |
| 3 | v_forward (m/s) | chassis speed in robot frame |
| 4 | omega_z (rad/s) | yaw rate |
| 5:11 | current wheel velocities (rad/s) | what the wheels actually did |
| 11:17 | base wheel velocities (rad/s) | what the base controller commanded |
| 17:23 | previous deviation (rad/s) | last RL action |

State dim = K × 20 (residual) or K × 16 (pure RL). Change K with `--obs-stack`.

### Action Space

`delta_omega` — additive wheel velocity corrections, clipped to ±15 rad/s.

### Fault Model

**Training:** one wheel is randomly degraded at each `reset()`:
```python
fault_wheel_idx ~ Uniform{0,..,5}
fault_alpha     ~ Uniform[0.0, 1.0]   # 0 = dead, 1 = healthy
omega_actual = omega_cmd * fault_alpha
```

**Evaluation:** fault injected mid-episode at steps 150 and 700. Controlled via CLI flags:

| Flag | Effect |
|---|---|
| `--num-fault-wheels 2` | fault 2 wheels simultaneously |
| `--num-fault-wheels 3` | fault 3 wheels simultaneously |
| `--same-side` | all faulted wheels on the same side (0-2 or 3-5) |
| `--jitter-fault` | fault alpha varies each step: `clip(alpha*(1+N(0,0.1²)), 0, 1)` |

### Reward

**`tracking_reward`** (training default):
```
r = -0.2|cross_track| - 0.5|heading_err| - 0.01‖δω‖² - 0.01‖δω - δω_prev‖² + 50·reached - 0.05
```

**`eval_reward`** (used automatically in eval mode) — success-focused variant.

### Termination

| Condition | Type |
|---|---|
| All waypoints reached | `terminated=True` |
| Roll or pitch > 0.7 rad | `terminated=True` |
| Position > 30 m from origin | `terminated=True` |
| 2000 steps elapsed | `truncated=True` |

---

## Training

### Local training

```bash
# Residual RL (default) — fault-trained
uv run python main.py --exp-name fault

# Pure RL (no base controller)
uv run python main.py --exp-name fault_pure --pure

# No-fault baseline
uv run python main.py --exp-name normal --no-fault

# With W&B logging
uv run python main.py --exp-name fault --wandb --wandb-project 285-final-project

# History window ablation (custom obs stack size)
uv run python main.py --exp-name fault_k3 --obs-stack 3 --wandb
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--exp-name` | `fault` | Prefix for checkpoint and log files |
| `--obs-stack` | `5` | History window length (timesteps stacked) |
| `--num-envs` | cpu_count-1 | Parallel environments |
| `--pure` | off | Pure RL instead of residual |
| `--no-fault` | off | Disable fault injection (healthy baseline) |
| `--wandb` | off | Enable Weights & Biases logging |
| `--wandb-project` | `285-final-project` | W&B project name |

Checkpoints saved as `{exp_name}-td3_checkpoint.pth`. Training logs saved as `{exp_name}-training_log.csv` and `{exp_name}-episode_returns.csv`.

### Cloud training on Modal

Run a single experiment on an A10G GPU (32 CPUs, 32 GB RAM) — detached so you can close your terminal:

```bash
modal run --detach scripts/modal_run.py --exp-name fault_k1 --obs-stack 1 --wandb
```

Run the 3 history ablation experiments in parallel (open 3 terminals):

```bash
# Terminal 1
modal run --detach scripts/modal_run.py --exp-name fault_k1  --obs-stack 1  --wandb

# Terminal 2
modal run --detach scripts/modal_run.py --exp-name fault_k3  --obs-stack 3  --wandb

# Terminal 3
modal run --detach scripts/modal_run.py --exp-name fault_k10 --obs-stack 10 --wandb
```

Monitor jobs: `modal app list`

Download results after training:
```bash
modal volume get six-wheel-rl-volume fault_k10-td3_checkpoint.pth .
modal volume get six-wheel-rl-volume fault_k10-training_log.csv .
```

Reset the volume (start fresh, removes all checkpoints):
```bash
modal volume delete six-wheel-rl-volume
```

All Modal flags mirror `main.py` flags — see `modal run scripts/modal_run.py --help`.

---

## Evaluation

### Single eval run

```bash
# Evaluate default fault checkpoint (1 wheel, random placement)
uv run python main.py --eval --exp-name fault_w1 --ckpt-name fault

# Multi-wheel fault
uv run python main.py --eval --exp-name fault_w2 --ckpt-name fault --num-fault-wheels 2
uv run python main.py --eval --exp-name fault_w3 --ckpt-name fault --num-fault-wheels 3

# Same-side fault placement
uv run python main.py --eval --exp-name fault_w2_sameside --ckpt-name fault --num-fault-wheels 2 --same-side

# Fault jitter robustness
uv run python main.py --eval --exp-name fault_jitter --ckpt-name fault --jitter-fault

# Pure RL vs residual comparison
uv run python main.py --eval --exp-name pure_w2 --ckpt-name fault_pure --pure --num-fault-wheels 2
```

Results saved to `eval_logs/{exp_name}-eval_log.csv`.

### Full ablation study (9 experiments)

```bash
bash run_ablation.sh
```

Runs all 9 experiments sequentially. Logs to `ablation_logs/`. Results to `eval_logs/`.

### History window ablation (train + eval)

```bash
# Step 1: train (or use Modal above)
bash run_history_ablation.sh   # trains fault_k1, fault_k3, fault_k10

# Step 2: eval each checkpoint
bash run_history_eval.sh
```

---

## Plotting

### Ablation study overview (9 experiments)

```bash
uv run python eval_logs/ablation_grapher.py
```

Outputs to `eval_logs/`:
- `ablation_overall.png` — success rate across all 9 experiments
- `ablation_multiwheels.png` — residual vs pure RL as fault count increases
- `ablation_sameside.png` — same-side vs random fault placement
- `ablation_jitter.png` — constant vs jittered fault

### History window ablation

```bash
uv run python eval_logs/history_grapher.py
```

Outputs to `eval_logs/`:
- `history_success.png` — success rate vs window size (k=1, 3, 5, 10)
- `history_reward.png` — avg episode reward vs window size
- `history_steps.png` — avg episode length vs window size

### Per-alpha / per-wheel breakdown

```bash
uv run python eval_logs/eval_grapher.py \
    --experiment fault_w1=fault_w1-eval_log.csv \
    --experiment fault_w2=fault_w2-eval_log.csv \
    --experiment fault_w3=fault_w3-eval_log.csv
```

---

## W&B Logging

Login (one-time):
```bash
uv run wandb login
```

Add `--wandb` to any training command. Runs appear at [wandb.ai](https://wandb.ai) under project `285-final-project`.

Metrics logged:
- `train/critic1_loss`, `train/critic2_loss`, `train/actor_loss` — every 1000 steps
- `train/epsilon`, `train/replay_buffer_size` — every 1000 steps
- `episode/total_reward`, `episode/discounted_reward`, `episode/steps`, `episode/expected_q` — every episode
