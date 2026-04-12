import os
import numpy as np
import torch, envs, gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from controllers.TD3_controller import Agent
import csv
import time

EPISODES = 100_000
DISCOUNT = 0.97
RENDER_TRAINING = False
NUM_ENVS = 16
USE_ASYNC_VECTOR_ENV = True
OBS_MODE = "PARTIAL_OBS"
GRADIENT_STEPS_PER_VECTOR_STEP = 2  # good default for 24 envs; increase if stable
THROUGHPUT_LOG_INTERVAL_SEC = 5.0

csv_path = 'training_log.csv'

def logging(episode, step, reward, disc_rewards, losses):
    # create if not exist, with column names

    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as log_file:
        log_writer = csv.writer(log_file)
        if (not file_exists) or log_file.tell() == 0:
            log_writer.writerow(['Episode', 'Step', 'Reward', 'Discounted', 'Losses'])
        log_writer.writerow([episode, step, reward, disc_rewards, losses])

def _make_env(rank: int):
    def _thunk():
        return gym.make(
            "SixWheelSkidSteer-v0",
            render_mode=None,
            obs_mode=OBS_MODE,
        )
    return _thunk


def _build_env():
    if NUM_ENVS > 1:
        env_fns = [_make_env(i) for i in range(NUM_ENVS)]
        if USE_ASYNC_VECTOR_ENV:
            return AsyncVectorEnv(env_fns)
        return SyncVectorEnv(env_fns)
    return gym.make(
        "SixWheelSkidSteer-v0",
        render_mode="human" if RENDER_TRAINING else None,
        obs_mode=OBS_MODE,
    )

def main():
    env = _build_env()
    agent = Agent(num_envs=NUM_ENVS)
    vector_mode = NUM_ENVS > 1

    if vector_mode and OBS_MODE == "FULL_OBS":
        raise ValueError("FULL_OBS is not supported in parallel mode. Use PARTIAL_OBS for vectorized training.")

    print(
        f"Running with device={agent.device}, num_envs={NUM_ENVS}, "
        f"vector_backend={'async' if USE_ASYNC_VECTOR_ENV and vector_mode else 'sync' if vector_mode else 'single'}, "
        f"obs_mode={OBS_MODE}"
    )

    episode = 0
    global_steps = 0
    # use log to find last episode
    try:
        reader = csv.reader(open(csv_path))
        last_episode = 0
        for row in reader:
            if row[0] != 'Episode':
                last_episode = max(last_episode, int(row[0]))
        episode = last_episode
    except Exception as e:
        print(e)
        print("No existing log file found, starting from episode 0.")

    try:
        if vector_mode:
            obs, _ = env.reset()
            ep_steps = np.zeros(NUM_ENVS, dtype=np.int64)
            ep_rewards = np.zeros(NUM_ENVS, dtype=np.float64)
            ep_discounted = np.zeros(NUM_ENVS, dtype=np.float64)
            meter_start_t = time.perf_counter()
            meter_start_global_steps = global_steps
            meter_vector_steps = 0

            while episode < EPISODES:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
                actions = agent.make_decision(obs_t).detach().cpu().numpy()
                # teaching mode
                actions = np.zeros_like(actions)

                next_obs, rewards, terminated, truncated, infos = env.step(actions)
                dones = np.logical_or(terminated, truncated)
                rewards_f = np.asarray(rewards, dtype=np.float32)

                # Add transitions once, then run multiple optimizer-only steps.
                agent.add_transitions(obs, actions, rewards_f, dones.astype(np.float32), next_obs)
                losses = None
                for _ in range(GRADIENT_STEPS_PER_VECTOR_STEP):
                    losses = agent.optimize_step()

                ep_rewards += rewards_f
                ep_discounted += (DISCOUNT ** ep_steps) * rewards_f
                ep_steps += 1
                global_steps += NUM_ENVS
                meter_vector_steps += 1

                now_t = time.perf_counter()
                elapsed = now_t - meter_start_t
                if elapsed >= THROUGHPUT_LOG_INTERVAL_SEC:
                    stepped_transitions = global_steps - meter_start_global_steps
                    steps_per_sec = stepped_transitions / max(elapsed, 1e-9)
                    vec_step_ms = (elapsed / max(meter_vector_steps, 1)) * 1000.0
                    per_env_step_ms = (elapsed / max(stepped_transitions, 1)) * 1000.0
                    print(
                        f"Perf {elapsed:.1f}s | transitions/s={steps_per_sec:.1f} "
                        f"| vec_step_ms={vec_step_ms:.3f} | env_step_ms={per_env_step_ms:.3f}"
                    )
                    meter_start_t = now_t
                    meter_start_global_steps = global_steps
                    meter_vector_steps = 0

                finished_idx = np.where(dones)[0]
                for idx in finished_idx:
                    episode += 1
                    print(
                        f"Episode {episode} finished at step {int(ep_steps[idx])} "
                        f"with sum reward {ep_rewards[idx]:.3f} and discounted reward {ep_discounted[idx]:.3f}"
                    )
                    logging(episode, int(ep_steps[idx]), float(ep_rewards[idx]), float(ep_discounted[idx]), losses)
                    ep_steps[idx] = 0
                    ep_rewards[idx] = 0.0
                    ep_discounted[idx] = 0.0
                    if episode >= EPISODES:
                        break

                if global_steps % (100 * NUM_ENVS) == 0:
                    mean_r = float(np.mean(rewards_f))
                    print(f"GlobalStep {global_steps}, Mean step reward: {mean_r:.3f}, Losses: {losses}")

                obs = next_obs
        else:
            while episode < EPISODES:
                obs, _ = env.reset()
                episode += 1
                steps = 0
                sum_rewards = 0.0
                discounted_reward = 0.0
                done = False

                while not done:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
                    action = agent.make_decision(obs_t).detach().cpu().numpy()
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    reward_f = float(reward)
                    done = bool(terminated or truncated)
                    agent.add_transitions(obs_t, action, reward_f, float(done), next_obs)
                    losses = agent.optimize_step()
                    obs = next_obs
                    sum_rewards += reward_f
                    discounted_reward += (DISCOUNT ** steps) * reward_f
                    if steps % 100 == 0:
                        print(f"Episode {episode}, Step {steps}, Reward: {reward_f:.3f}, Losses: {losses}")
                        logging(episode, steps, reward_f, discounted_reward, losses)
                    steps += 1

                print(
                    f"Episode {episode} finished at step {steps} "
                    f"with sum reward {sum_rewards:.3f} and discounted reward {discounted_reward:.3f}"
                )
    except KeyboardInterrupt:
        print("Training interrupted, saving checkpoint.")
        agent.checkpoint_save(path="td3_checkpoint")

    env.close()

if __name__ == "__main__":
    main()