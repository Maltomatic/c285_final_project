"""
Reward functions for SixWheelEnv.

Each function has the same signature:
    reward_fn(obs_t, delta_omega, prev_delta_omega, waypoint_reached, weights)
    -> float

Pass the desired function to SixWheelEnv via reward_fn=...
"""

import numpy as np

from envs.env_configs import _ACTION_CLIP

def tracking_reward(
    obs_t: np.ndarray,
    delta_omega: np.ndarray,
    prev_delta_omega: np.ndarray,
    waypoint_reached: bool,
    weights: tuple = (0.2, 0.5, 0.01, 0.01, 50.0, 0.1),
) -> float:
    """
    Default reward.

    weights:
        w1  cross-track error penalty     — stay on path
        w2  heading error penalty         — point toward goal
        w3  action magnitude penalty      — keep residuals small
        w4  action smoothness penalty     — avoid jerk
        w5  waypoint reached bonus        — incentivize progress
        w6  time step penalty             — incentivize speed
    """
    w1, w2, w3, w4, w5, w6 = weights
    cross_track_err = float(obs_t[0])
    heading_err     = float(obs_t[1])

    r = 0.0
    r += -w1 * abs(cross_track_err)
    # print("Cross-track error: -", cross_track_err)
    r += -w2 * abs(heading_err)
    # print("Heading error: -", heading_err)
    r += -w3 * float(np.sum(delta_omega ** 2))
    # print("Action magnitude: -", float(np.sum(delta_omega ** 2)))
    r += -w4 * float(np.sum((delta_omega - prev_delta_omega) ** 2))
    # print("Action smoothness: -", float(np.sum((delta_omega - prev_delta_omega) ** 2)))
    r += +w5 * float(waypoint_reached)
    # print("Waypoint reached bonus: ", float(waypoint_reached))
    r += -w6
    # print("Total step reward: ", r)
    return r

def sparse_reward(
    obs_t: np.ndarray,
    delta_omega: np.ndarray,
    prev_delta_omega: np.ndarray,
    waypoint_reached: bool,
    weights: tuple = (20.0, 10.0, 0.1),
) -> float:
    """
    Sparse reward — only illegal action magnitude penalty, waypoint bonuses, and a time penalty.
    Harder to learn but produces cleaner behavior.

    weights:
        w1  waypoint reached bonus
        w2  time step penalty
    """
    w1, w2, w3 = weights
    r  = -w1 * float(np.sum(delta_omega) - ((_ACTION_CLIP)*2 - 1) ** 2)
    r += +w2 * float(waypoint_reached)
    r += -w3
    return r

def eval_reward(
    obs_t: np.ndarray,
    delta_omega: np.ndarray,
    prev_delta_omega: np.ndarray,
    waypoint_reached: bool,
    weights: tuple = (0.0, 10.0, 0.1),
) -> float:
    """
    Only time penalty; env adds completion reward. Only for eval.

    weights:
        w1  waypoint reached bonus
        w2  time step penalty
    """
    w1, w2, w3 = weights
    r = 0.0
    r += -w3
    return r
