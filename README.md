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
│   └── robot.xml               MuJoCo MJCF model
├── controllers/
│   └── base_controller.py      WaypointController + BaseAllocator
├── envs/
│   ├── __init__.py             gym.register call (set max episode length here)
│   ├── six_wheel_env.py        Gymnasium environment
│   └── rewards.py              Reward functions definition (two versions)
└── test_robot.ipynb            MuJoCo physics & base controller test
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

```python
import envs  # must import to trigger gym.register
import gymnasium as gym
env = gym.make("SixWheelSkidSteer-v0")
```

### Timing

| Level | Duration |
|---|---|
| MuJoCo timestep | 0.002 s |
| Env step (`_FRAME_SKIP=10`) | 0.02 s (50 Hz control) |
| Max episode (`max_episode_steps=2000`) | 40 s simulated |

### Observation Space — `Box(115,)` float32

23 values per timestep, stacked over the last K=5 timesteps (100 ms of history):

| Index | Signal | Why it's there |
|---|---|---|
| 0 | cross-track error (m) | signed lateral deviation from path |
| 1 | heading error (rad) | angular error to next waypoint |
| 2 | dist to waypoint (m) | how far to go |
| 3 | v_forward (m/s) | chassis speed in robot frame |
| 4 | omega_z (rad/s) | yaw rate |
| 5:11 | **actual wheel velocities** (rad/s) | what the wheels actually did |
| 11:17 | **omega_base** — base allocator output (rad/s) | what the controller commanded |
| 17:23 | prev_delta_omega (rad/s) | last RL action |

> **Fault detection relies on indices 5:17.** When a wheel is degraded, `actual[i] < omega_base[i] + delta_omega[i]`. The 5-frame stack lets the policy see this discrepancy persisting over time, distinguishing a permanent fault from a transient slip.

### Action Space — `Box(6,)` float32, clipped to ±5 rad/s

`delta_omega` — additive wheel velocity corrections on top of `omega_base`. The actual command sent to MuJoCo is:

```python
omega_cmd = omega_base + delta_omega
# fault applied (hidden from policy):
omega_cmd[fault_idx] *= fault_alpha
data.ctrl[:] = omega_cmd
```

### Fault Model

Sampled fresh at every `reset()`:

```python
fault_wheel_idx ~ Uniform{0, 1, 2, 3, 4, 5}
fault_alpha     ~ Uniform[0.0, 1.0]   # 0 = dead wheel, 1 = healthy
```

Applied as a multiplicative degradation to one wheel's command. The policy never sees `fault_wheel_idx` or `fault_alpha` directly.

For evaluation, inject faults mid-episode:

```python
env.unwrapped.inject_fault(wheel_idx=2, alpha=0.0)  # kill wheel 2
```

### Reward (`envs/rewards.py`)

Two reward functions are available, passed via `reward_fn=` at construction:

**`tracking_reward`** (default) — weights `(1.0, 0.5, 0.01, 0.01, 50.0, 0.05)`:

```
r = -w1*|cross_track_err| - w2*|heading_err|
  - w3*sum(delta_omega²) - w4*sum((delta_omega - prev_delta_omega)²)
  + w5*(waypoint_reached) - w6
```

**`sparse_reward`** — weights `(100.0, 0.1)`:

```
r = +w1*(waypoint_reached) - w2
```

### Termination

| Condition | Type |
|---|---|
| All waypoints reached | `terminated=True` |
| Roll or pitch > 0.7 rad (~40°) | `terminated=True` |
| Position > 15 m from origin | `terminated=True` |
| 2000 env steps elapsed | `truncated=True` (TimeLimit wrapper) |

---

## Setup

```bash
# requires uv (https://github.com/astral-sh/uv)
uv sync        # installs mujoco, gymnasium, numpy
uv run jupyter notebook test_robot.ipynb   # physics sanity checks
```

---

## What's Left To Do

### SAC + training

- [ ] Implement SAC
- [ ] Wrap env with `gym.make("SixWheelSkidSteer-v0")` — `TimeLimit` is already applied
- [ ] Train a **no-fault baseline** (`inject_fault(idx, alpha=1.0)` at reset) — verify the base controller + zero residual gets reasonable reward before adding fault randomness
- [ ] Train with full fault distribution (default: fault sampled at each reset)
- [ ] Evaluation: inject specific faults mid-episode via `env.unwrapped.inject_fault(idx, alpha)`, measure cross-track error vs. base-controller-only baseline

### Environment

- [ ] Expand waypoint trajectory to include a curved segment (currently only straight legs + 90° corners)
- [ ] Tune reward weights once training begins (current defaults are untested under RL)
- [ ] Consider `gymnasium.wrappers.NormalizeObservation` if training is unstable
