"""
Gymnasium environment for the 6-wheel skid-steer robot.

Control pipeline per step
─────────────────────────
  WaypointController → (v, ω)
       → BaseAllocator → omega_base[6]
       → (+) delta_omega  (RL action)
       → fault_fn         (multiplicative wheel degradation)
       → data.ctrl        (MuJoCo velocity actuators)

Observation (23 values per timestep, stacked K=5 → 115 total)
──────────────────────────────────────────────────────────────
  [0]  cross_track_error      (m)
  [1]  heading_error          (rad)
  [2]  dist_to_waypoint       (m)
  [3]  v_forward              (m/s, robot frame)
  [4]  omega_z                (rad/s)
  [5:11]  actual wheel velocities qvel[6:12]   (rad/s)
  [11:17] base allocator commands omega_base   (rad/s)
  [17:23] previous residual action delta_omega (rad/s)

Action
──────
  delta_omega : (6,) wheel velocity corrections, clipped to ±15 rad/s

Fault model
───────────
  Sampled at reset:
    fault_wheel_idx ~ Uniform{0,...,5}
    fault_alpha     ~ Uniform[0, 1]
  Applied:
    omega_actual[fault_wheel_idx] *= fault_alpha
  The policy does NOT observe fault parameters.
"""

import os
from collections import deque

import mujoco
from mujoco import viewer
import numpy as np
import gymnasium as gym

from controllers.base_controller import WaypointController, BaseAllocator, ZeroAllocator, WAYPOINTS, REVERSE_WAYPOINTS, EVAL_WAYPOINTS
from envs.rewards import tracking_reward, sparse_reward

from envs.env_configs import _ACTION_DIM, _OBS_DIM, _ACTION_CLIP, _OBS_STACK, _FRAME_SKIP,\
    _WHEEL_RADIUS, _TRACK_WIDTH, _RESET_SETTLE_STEPS, _MAX_CTRL_ACCEL,\
    EVAL, EVAL_EPISODES, FAULT_STEPS
import envs.env_configs as env_config

# constants
_XML_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "robot.xml")
_CAM_LOOKAT = np.array([2.5, 2.5, 0.2], dtype=np.float32)
_CAM_DISTANCE = 10.0
_CAM_AZIMUTH = 90.0
_CAM_ELEVATION = -25.0
eval_fault_types = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

class SixWheelEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        reward_fn=tracking_reward,
        reward_weights: tuple | None = None,
        env_id = 0,
        no_fault: bool | None = None,
        pure_rl: bool | None = None,
    ):
        super().__init__()

        # validate render_mode
        assert render_mode is None or render_mode in self.metadata["render_modes"], (
            f"Invalid render_mode '{render_mode}'. "
            f"Supported: {self.metadata['render_modes']}"
        )
        self.render_mode = render_mode
        self.reward_fn = reward_fn
        self.reward_weights = reward_weights if reward_weights is not None else None
        self.env_id = env_id
        self.pure_rl = env_config.PURE_RL if pure_rl is None else bool(pure_rl)
        # Keep reward helpers in this process consistent with the env mode.
        env_config.PURE_RL = self.pure_rl
        self.no_fault = bool(no_fault)
        # if eval, no_fault until step 150, then inject a random fault and keep it until the end of the episode
        if EVAL:
            self.no_fault = True
        self._steps = 0

        # load MuJoCo model (once, shared across resets)
        self.model = mujoco.MjModel.from_xml_path(_XML_PATH)
        self.data  = mujoco.MjData(self.model)
        self.model.opt.iterations = 10
        self.model.opt.ls_iterations = 5
        self._ctrl_dt = float(self.model.opt.timestep * _FRAME_SKIP)
        self._max_ctrl_delta = float(_MAX_CTRL_ACCEL * self._ctrl_dt)

        # gymnasium spaces
        self.obs_size = _OBS_DIM if not self.pure_rl else (_OBS_DIM-6) # no base vel, no cross-track
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-(_ACTION_CLIP if not self.pure_rl else _ACTION_CLIP*2), high=(_ACTION_CLIP if not self.pure_rl else _ACTION_CLIP*2),
            shape=(_ACTION_DIM,), dtype=np.float32,
        )

        # controllers (recreated on reset)
        wp = WAYPOINTS if env_id % 2 else REVERSE_WAYPOINTS
        self.wp_controller  = WaypointController(wp if not EVAL else EVAL_WAYPOINTS)
        self.allocator = BaseAllocator(_WHEEL_RADIUS, _TRACK_WIDTH) if not self.pure_rl else ZeroAllocator()

        # history buffer
        self._obs_history: deque = deque(maxlen=_OBS_STACK)

        # episode state
        self._prev_delta_omega = np.zeros(_ACTION_DIM, dtype=np.float32)
        self._prev_omega_base = np.zeros(_ACTION_DIM, dtype=np.float32)
        self._prev_ctrl_cmd = np.zeros(_ACTION_DIM, dtype=np.float32)
        self.fault_wheel_idx: int = 0
        self.fault_alpha: float   = 1.0
        self.inject_step = FAULT_STEPS[0] if EVAL else 0

        # rendering handles
        self._viewer   = None   # mujoco.viewer passive handle (human mode)
        self._renderer = None   # mujoco.Renderer              (rgb_array mode)
        self._camera_initialized = False

    # Gymnasium API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # seeds self.np_random

        self._steps = 0

        # Reset physics
        mujoco.mj_resetData(self.model, self.data)

        # Ensure no dynamic or actuator state leaks across episodes.
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.qacc_warmstart[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        if self.data.act.size > 0:
            self.data.act[:] = 0.0

        # Place chassis at origin, flat orientation
        self.data.qpos[0:3] = [0.0, 0.0, 0.25]   # x, y, z
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # quaternion (identity)
        mujoco.mj_forward(self.model, self.data)

        # Let contacts settle with zero control before first policy action.
        self.data.ctrl[:] = 0.0
        for _ in range(_RESET_SETTLE_STEPS):
            mujoco.mj_step(self.model, self.data)

        # Fault injection (sampled fresh every episode)
        self.fault_wheel_idx = int(self.np_random.integers(0, _ACTION_DIM))
        self.fault_alpha      = 1.0 if self.no_fault else float(self.np_random.uniform(0.0, 1.0))
        if EVAL:
            injection_offset = int(self.np_random.integers(-5, 5))
            self.inject_step = np.random.choice(FAULT_STEPS) + injection_offset

        # if not self.no_fault:
        #     print(f"Fault in env {self.env_id}: wheel {self.fault_wheel_idx} at {self.fault_alpha:.2f}x effectiveness")

        # Reset controllers
        self.wp_controller.reset()

        # Reset episode state
        self._prev_delta_omega = np.zeros(_ACTION_DIM, dtype=np.float32)
        self._prev_omega_base = np.zeros(_ACTION_DIM, dtype=np.float32)
        self._prev_ctrl_cmd = np.zeros(_ACTION_DIM, dtype=np.float32)

        # Fill history buffer with zeros
        self._obs_history.clear()
        for _ in range(_OBS_STACK):
            self._obs_history.append(np.zeros(self.obs_size, dtype=np.float32))

        # stack = self._get_stacked_obs()
        obs = self._single_obs(*self._chassis_pose(), self._prev_omega_base, self._prev_delta_omega)
        return obs, {}

    def step(self, action: np.ndarray):
        delta_omega = np.clip(action, -_ACTION_CLIP, _ACTION_CLIP).astype(np.float32) if not self.pure_rl else action.astype(np.float32)

        # 1. Post-deviation speed
        omega_cmd_target = self._prev_omega_base + delta_omega
        # Slew-rate limit wheel commands to avoid large startup/transition impulses.
        cmd_delta = np.clip(omega_cmd_target - self._prev_ctrl_cmd, -self._max_ctrl_delta, self._max_ctrl_delta)
        omega_cmd = self._prev_ctrl_cmd + cmd_delta
        # omega_cmd = omega_cmd_target.copy()

        # print(f"\nLast timestep base speed: {self._prev_omega_base}, delta_omega: {delta_omega}, output speed: {omega_cmd}")

        if self._steps == self.inject_step and EVAL:
            self.inject_fault(
                wheel_idx=int(self.np_random.integers(0, _ACTION_DIM)),
                alpha=float(self.np_random.choice(eval_fault_types))
            )
        # 2. Apply fault (hidden from policy)
        omega_faulted = omega_cmd.copy()
        omega_faulted[self.fault_wheel_idx] *= self.fault_alpha

        # 3. Step physics
        self.data.ctrl[:] = omega_faulted
        for _ in range(_FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        self._prev_ctrl_cmd = omega_cmd.copy()
        
        # 4. Body controller
        pos_xy, heading = self._chassis_pose()
        prev_wp_idx = self.wp_controller._idx          # snapshot before compute
        v, omega, _ = self.wp_controller.compute(pos_xy, heading)
        waypoint_reached = self.wp_controller._idx > prev_wp_idx  # True if we have advanced to the next waypoint
        
        # 5. New base speed for next step
        omega_base = self.allocator.allocate(v, omega)
        # print(f"New base speed for obs: {omega_base}")
        self._prev_omega_base = omega_base

        # 6. Build obs
        obs_t = self._single_obs(pos_xy, heading, omega_base, self._prev_delta_omega)
        self._obs_history.append(obs_t)
        # stacked = self._get_stacked_obs()

        # 7. Reward
        if self.reward_weights is not None:    
            reward = self.reward_fn(
                obs_t, action, self._prev_delta_omega,
                waypoint_reached, self.reward_weights
            )
        else:
            reward = self.reward_fn(
                obs_t, action, self._prev_delta_omega,
                waypoint_reached
            )
        # add completion reward in EVAL
        if EVAL and self.wp_controller.is_done():
            reward += 300.0

        # 8. Termination
        success = self.wp_controller.is_done()
        terminated = self._is_terminated(pos_xy)
        truncated  = False  # handled externally by TimeLimit wrapper

        self._prev_delta_omega = delta_omega.copy()

        if self.render_mode == "human":
            self.render()

        info = {
            "step": self._steps,
            "success": bool(success),
            "fault_wheel_idx": int(self.fault_wheel_idx),
            "fault_alpha": float(self.fault_alpha),
        }
        self._steps += 1
        # return stacked, reward, terminated, truncated, {}
        return obs_t, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = viewer.launch_passive(self.model, self.data)
            if not self._camera_initialized:
                self._viewer.cam.lookat[:] = _CAM_LOOKAT
                self._viewer.cam.distance = _CAM_DISTANCE
                self._viewer.cam.azimuth = _CAM_AZIMUTH
                self._viewer.cam.elevation = _CAM_ELEVATION
                self._camera_initialized = True
            self._viewer.sync()

        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()  # np.ndarray (H, W, 3) uint8

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
            self._camera_initialized = False
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # mid-episode fault injection (evaluation only)

    def inject_fault(self, wheel_idx: int, alpha: float) -> None:
        """Inject or change a fault mid-episode (for evaluation experiments)."""
        assert 0 <= wheel_idx < _ACTION_DIM, "wheel_idx must be in [0, 5]"
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]" # TODO: allow -1~1 for special malfunction? eg. blown tire, cause drag
        # print(f"Injecting fault in env {self.env_id} at step {self._steps}: wheel {wheel_idx} at {alpha:.2f}x effectiveness")
        self.fault_wheel_idx = wheel_idx
        self.fault_alpha     = alpha

    # internal helpers

    def _chassis_pose(self):
        """Return (pos_xy, heading) from freejoint qpos."""
        x, y = self.data.qpos[0], self.data.qpos[1]
        qw, qx, qy, qz = self.data.qpos[3:7]
        heading = np.arctan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy ** 2 + qz ** 2),
        )
        return np.array([x, y], dtype=np.float64), float(heading)

    def _single_obs(
        self,
        pos_xy: np.ndarray,
        heading: float,
        omega_base: np.ndarray,
        prev_delta_omega: np.ndarray,
    ) -> np.ndarray:
        """Build one 23-dim observation vector."""
        # print(f"Residual: {not self.pure_rl}")
        wp = self.wp_controller.current_waypoint()
        dx, dy = wp - pos_xy
        dist    = float(np.hypot(dx, dy))

        desired_heading = np.arctan2(dy, dx)
        heading_error   = float(_wrap_to_pi(desired_heading - heading))

        # Cross-track error: signed perpendicular distance (approx)
        cross_track_error = float(np.sin(heading_error) * dist)

        # Body-frame velocities from freejoint qvel
        vx_w = float(self.data.qvel[0])
        vy_w = float(self.data.qvel[1])
        wz   = float(self.data.qvel[5])
        v_forward = vx_w * np.cos(heading) + vy_w * np.sin(heading)

        # Actual wheel velocities: qvel[6:12]
        wheel_qvel = self.data.qvel[6:12].astype(np.float32)

        obs = np.concatenate([
            [cross_track_error, heading_error, dist, v_forward, wz],  # 5
            wheel_qvel,        # 6 — actual
            omega_base,        # 6 — base commanded
            prev_delta_omega,  # 6 — previous residual
        ]).astype(np.float32) if not self.pure_rl else np.concatenate([
            [cross_track_error, heading_error, dist, v_forward, wz],  # 5
            wheel_qvel,        # 6 — actual
            prev_delta_omega,  # 6 — previous action
        ]).astype(np.float32)
        assert obs.shape == (self.obs_size,), f"obs shape mismatch: {obs.shape}"
        return obs

    def _get_stacked_obs(self) -> np.ndarray:
        return np.concatenate(list(self._obs_history), axis=0).astype(np.float32)

    def _is_terminated(self, pos_xy: np.ndarray) -> bool:
        # All waypoints reached
        if self.wp_controller.is_done():
            return True
        # Robot flipped (large roll or pitch)
        qw, qx, qy, qz = self.data.qpos[3:7]
        roll  = np.arctan2(2.0 * (qw * qx + qy * qz),
                           1.0 - 2.0 * (qx ** 2 + qy ** 2))
        pitch = np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
        if abs(roll) > 1.0 or abs(pitch) > 1.0:
            return True
        # Out of bounds
        if np.any(np.abs(pos_xy) > 30.0):
            return True
        return False


def _wrap_to_pi(angle: float) -> float:
    return ((angle + np.pi) % (2.0 * np.pi)) - np.pi
