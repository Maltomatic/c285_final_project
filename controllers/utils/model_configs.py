# agent hyperparameters
import math

EPS_START = 0.99
EPS_MIN = 0.05
EXPLORE_FRACTION = 0.7
DECAY_INTERVAL = 1000

NOISE_STD = 0.3
NOISE_MIN = 0.05

G_STEPS = 30_000_000
FT_RATIO = 0.7
DISCOUNT = 0.997 # agent gamma
CAPACITY = 5_000_000

# def get_noise_std(step, ft=False):
#     total_steps = G_STEPS
#     if ft:
#         total_steps = G_STEPS * 0.7
#     frac = min(step/(total_steps * 0.85), 1.0)
#     std = NOISE_STD - (NOISE_STD - NOISE_MIN) * frac
#     return std

# def ft_noise(step):
#     total_steps = G_STEPS * 0.3
#     frac = min(step/(total_steps * 0.85), 1.0)
#     std = NOISE_STD - (NOISE_STD - NOISE_MIN) * frac
#     return std

RWD_FN = 'tracking' # 'tracking', 'sparse', 'eval'

_decay_events = G_STEPS * EXPLORE_FRACTION / DECAY_INTERVAL # how many times to decay eps
_decay_amount = (EPS_MIN/EPS_START)**(1.0 / _decay_events)
EPS_DECAY = round(_decay_amount, 5)

def verify_setup(dec = EPS_DECAY):
    # check how many steps to decay from EPS_START to EPS_MIN with EPS_DECAY
    ratio = EPS_MIN / EPS_START
    steps = math.log(ratio, dec)
    final_explore_step = steps * DECAY_INTERVAL
    print(f"With decay {dec}, it would take {steps:.2f} decay steps, or {final_explore_step:.0f} global steps to reach EPS_MIN.")

if __name__ == "__main__":
    print("Verifying epsilon decay setup...")
    verify_setup()
    # 0.99986 w/o tuning run
    # 0.99978 pre-tuning
    # 0.99945 for tuning