"""
Base controller for the 6-wheel skid-steer robot.

WaypointController  — tracks a fixed waypoint list, outputs (v, omega)
BaseAllocator       — converts (v, omega) to 6 wheel angular velocities

These are intentionally simple; the RL residual corrects for faults on top.
"""

import numpy as np

# ── Waypoint trajectory ──────────────────────────────────────────────────────
# L-shape + return: straight → 90° corner → back → home
WAYPOINTS = np.array([
    [5.0, 0.0],   # straight along X
    [5.0, 5.0],   # 90° left turn, then straight along Y
    [0.0, 5.0],   # 90° left turn, back toward X=0
    [0.0, 0.0],   # home
], dtype=np.float64)


def _wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


class WaypointController:
    """
    Proportional waypoint-following controller.

    compute(pos_xy, heading) → (v, omega, done)
      v      : desired forward speed  [m/s]
      omega  : desired yaw rate       [rad/s]
      done   : True when all waypoints have been reached
    """

    def __init__(
        self,
        waypoints: np.ndarray = WAYPOINTS,
        arrival_radius: float = 0.3,
        kp_heading: float = 3.0,
        v_max: float = 2.0,
        omega_max: float = 2.5,
    ):
        self.waypoints = np.array(waypoints, dtype=np.float64)
        self.arrival_radius = arrival_radius
        self.kp_heading = kp_heading
        self.v_max = v_max
        self.omega_max = omega_max
        self._idx = 0

    def reset(self) -> None:
        self._idx = 0

    def is_done(self) -> bool:
        return self._idx >= len(self.waypoints)

    def current_waypoint(self) -> np.ndarray:
        """Return the current target waypoint (x, y)."""
        if self.is_done():
            return self.waypoints[-1]
        return self.waypoints[self._idx]

    def compute(
        self, pos_xy: np.ndarray, heading: float
    ):
        """
        Parameters
        ----------
        pos_xy  : (2,) array, robot position in world frame [m]
        heading : robot yaw angle [rad]

        Returns
        -------
        v      : float, forward speed command [m/s]
        omega  : float, yaw rate command [rad/s]
        done   : bool
        """
        if self.is_done():
            return 0.0, 0.0, True

        wp = self.waypoints[self._idx]
        dx, dy = wp - pos_xy
        dist = np.hypot(dx, dy)

        # Advance waypoint when close enough
        if dist < self.arrival_radius:
            self._idx += 1
            if self.is_done():
                return 0.0, 0.0, True
            wp = self.waypoints[self._idx]
            dx, dy = wp - pos_xy
            dist = np.hypot(dx, dy)

        desired_heading = np.arctan2(dy, dx)
        heading_error = _wrap_to_pi(desired_heading - heading)

        # Proportional heading control
        omega = float(np.clip(self.kp_heading * heading_error,
                               -self.omega_max, self.omega_max))

        # Slow down when pointing away from goal
        v = float(self.v_max * max(0.0, np.cos(heading_error)))

        return v, omega, False


class BaseAllocator:
    """
    Differential-drive base allocator.

    Converts (v, omega) → 6 wheel angular velocity commands [rad/s]:
      [omega_fl, omega_ml, omega_rl, omega_fr, omega_mr, omega_rr]

    Left side  = indices 0,1,2
    Right side = indices 3,4,5

    Sign convention:
      Positive omega_wheel → wheel rotates to move robot FORWARD.
    """

    def __init__(self, wheel_radius: float = 0.15, track_width: float = 0.7):
        self.wheel_radius = wheel_radius
        self.track_width = track_width

    def allocate(self, v: float, omega: float) -> np.ndarray:
        """
        Parameters
        ----------
        v     : desired body forward speed [m/s]
        omega : desired body yaw rate [rad/s]  (positive = turn left)

        Returns
        -------
        omega_wheels : (6,) array of wheel angular velocities [rad/s]
        """
        v_left  = v - omega * self.track_width / 2.0
        v_right = v + omega * self.track_width / 2.0

        omega_left  = v_left  / self.wheel_radius
        omega_right = v_right / self.wheel_radius

        return np.array(
            [omega_left, omega_left, omega_left,
             omega_right, omega_right, omega_right],
            dtype=np.float32,
        )
