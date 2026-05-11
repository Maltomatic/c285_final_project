################# centralize parameters #################

# hardware setup controls
RENDER_TRAINING = False
DEBUG = False
GPU_THREAD = True  # True may be faster if GPU is strong and CPU is meh; on laptop 4070 basically no difference

# script controls - set through cmd args
NUM_ENVS = 16
NO_FAULT = False
NO_OP = False # if True, agent takes no action. deviation is 0
PURE_RL = True # if false, use residual policy w/ baseAllocator; if true, RL does everything
FINE_TUNE = True # if true, no-fault for 70% G_STEPS, fault for remianing 30%

# evaluation controls
EVAL = False
EVAL_EPISODES = 2000
FAULT_STEPS = [150, 700]
NUM_FAULT_WHEELS = 1     # eval only: number of wheels to fault simultaneously (1=single, 2-3=multi)
FAULT_JITTER = False     # eval only: fault alpha varies each step: omega_f = clip(alpha*(1+noise),0,1)*omega
JITTER_STD = 0.05       # std of per-step Gaussian noise on faulted wheel alpha (FAULT_JITTER only)
SAME_SIDE_FAULT = False  # eval only: if True, all faulted wheels are on the same side (0,1,2 or 3,4,5)
                         #            if False, wheels drawn uniformly from all 6 (no side constraint)

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