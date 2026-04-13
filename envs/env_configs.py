# centralize parameters
RENDER_TRAINING = False
DEBUG = False
GPU_THREAD = False # True may be faster if GPU is strong and CPU is meh; on laptop 4070 basically no difference


NO_FAULT = False
EVAL = True
EVAL_EPISODES = 100
NO_OP = False # if True, agent takes no action. deviation is 0

_ACTION_DIM = 6
_OBS_DIM = 23
STACK_OBS_DIM = 6 * 3 * 5 + 2 * 5 # 6 wheels, base/dev/actual, 5 steps; heading and dist, 5 steps
_ACTION_CLIP = 15.0

_OBS_STACK = 5           # timesteps stacked
_FRAME_SKIP = 10         # physics steps per env step → 0.01 * 10 = 0.1 s (10 Hz control)
_WHEEL_RADIUS = 0.15     # m
_TRACK_WIDTH = 0.70      # m
_RESET_SETTLE_STEPS = 10
_MAX_CTRL_ACCEL = 20   # max wheel-speed command slew [rad/s^2]