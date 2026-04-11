import numpy as np
import torch
import gymnasium as gym
import multiprocessing
import time
import csv

from envs.six_wheel_env import SixWheelEnv
from controllers.TD3_controller import Agent

EPISODES    = 100_000
DISCOUNT    = 0.97
NUM_ENVS    = max(1, multiprocessing.cpu_count() - 1)
CSV_PATH    = "training_log.csv"

print(f"Launching {NUM_ENVS} parallel environments.")


# ── Environment factory ───────────────────────────────────────────────────────

def make_env():
    def _init():
        return gym.make("SixWheelSkidSteer-v0")
    return _init


# ── Logging ───────────────────────────────────────────────────────────────────

def log_row(episode, step, reward, losses):
    write_header = False
    try:
        with open(CSV_PATH, "r") as f:
            write_header = f.read(1) == ""
    except FileNotFoundError:
        write_header = True

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Episode", "Step", "Reward", "Losses"])
        writer.writerow([episode, step, reward, losses])


def last_logged_episode():
    try:
        with open(CSV_PATH, "r") as f:
            reader = csv.reader(f)
            last = 0
            for row in reader:
                if row and row[0] != "Episode":
                    try:
                        last = max(last, int(row[0]))
                    except ValueError:
                        pass
            return last
    except FileNotFoundError:
        return 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    vec_env = gym.vector.AsyncVectorEnv([make_env() for _ in range(NUM_ENVS)])
    agent   = Agent(num_envs=NUM_ENVS)

    global_step  = 0
    episode_step = np.zeros(NUM_ENVS, dtype=np.int32)
    episode_idx  = np.arange(NUM_ENVS, dtype=np.int32) + last_logged_episode()
    sum_rewards  = np.zeros(NUM_ENVS, dtype=np.float64)

    raw_obs, _ = vec_env.reset()                             # (N, 23)
    obs_t = torch.stack([                                    # (N, state_dim)
        agent.parse_obs(raw_obs[i], env_idx=i)
        for i in range(NUM_ENVS)
    ])

    try:
        while episode_idx.min() < EPISODES:
            actions   = agent.make_decision(obs_t)           # (N, 6)
            actions_np = actions.detach().cpu().numpy()

            raw_next, rewards, terminated, truncated, _ = vec_env.step(actions_np)

            next_obs_t = torch.stack([
                agent.parse_obs(raw_next[i], env_idx=i)
                for i in range(NUM_ENVS)
            ])

            # Store all N transitions into the shared replay buffer
            for i in range(NUM_ENVS):
                done = bool(terminated[i]) or bool(truncated[i])
                agent.store_transition(
                    obs_t[i], actions_np[i], rewards[i], done, next_obs_t[i]
                )

            # Single gradient update drawn from the pooled buffer
            losses = agent.update()

            sum_rewards  += rewards
            episode_step += 1
            global_step  += 1

            if global_step % 100 == 0:
                print(
                    f"Step {global_step} | "
                    f"eps {agent.epsilon:.3f} | "
                    f"replay {len(agent.replay):,} | "
                    f"losses {losses}"
                )
                log_row(int(episode_idx.mean()), global_step,
                        float(rewards.mean()), losses)

            # Handle episode resets for envs that finished
            for i in range(NUM_ENVS):
                if terminated[i] or truncated[i]:
                    print(
                        f"  Env {i} | episode {episode_idx[i]} done "
                        f"at step {episode_step[i]} | "
                        f"sum_reward {sum_rewards[i]:.2f}"
                    )
                    agent.reset_env_hist(i)
                    episode_idx[i]  += NUM_ENVS        # stride by num_envs
                    episode_step[i]  = 0
                    sum_rewards[i]   = 0.0

            obs_t = next_obs_t

    except KeyboardInterrupt:
        print("Interrupted — saving checkpoint.")
        agent.checkpoint_save("td3_checkpoint")

    vec_env.close()


main()
