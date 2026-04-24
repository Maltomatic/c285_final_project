################# centralize parameters #################

# hardware setup controls
RENDER_TRAINING = False
DEBUG = False
GPU_THREAD = False # True may be faster if GPU is strong and CPU is meh; on laptop 4070 basically no difference

# script controls - set through cmd args
NUM_ENVS = 16
NO_FAULT = False
NO_OP = False # if True, agent takes no action. deviation is 0
PURE_RL = False # if false, use residual policy w/ baseAllocator; if true, RL does everything

# evaluation controls
EVAL = False
EVAL_EPISODES = 1000
FAULT_STEP = 150

# env constants
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