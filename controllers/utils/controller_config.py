# agent hyperparameters
import math

EPS_START = 0.99
EPS_MIN = 0.05
EXPLORE_FRACTION = 0.7
DECAY_INTERVAL = 1000

G_STEPS = 30_000_000
DISCOUNT = 0.997 # agent gamma
CAPACITY = 5_000_000

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
    # test_decays = 0.9955
    # print("Testing with decay =", test_decays)
    # verify_setup(test_decays)