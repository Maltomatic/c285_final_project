import numpy as np
import torch, envs, gymnasium as gym
from envs.six_wheel_env import SixWheelEnv
from controllers.TD3_controller import Agent
import time, csv

EPISODES = 100_000
DISCOUNT = 0.97
RENDER_TRAINING = True

csv_path = 'training_log.csv'

def logging(episode, step, reward, losses):
    # create if not exist, with column names

    with open(csv_path, mode='w+', newline='') as log_file:
        log_writer = csv.writer(log_file)
        if log_file.tell() == 0:
            log_writer.writerow(['Episode', 'Step', 'Reward', 'Losses'])
        log_writer.writerow([episode, step, reward, losses])

def reset():
    env = gym.make("SixWheelSkidSteer-v0")
    obs = env.reset()
    return obs

def main():
    env = gym.make("SixWheelSkidSteer-v0", render_mode="human" if RENDER_TRAINING else None)
    agent = Agent()
    for episode in range(EPISODES):
        obs, _ = env.reset()
        steps = 0
        sum_rewards = 0
        discounted_reward = 0
        done = False
        obs_t = agent.parse_obs(obs)[1]
        
        while not done:
            action = agent.make_decision(obs_t).detach().numpy()
            # action = np.zeros_like(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_obs_t = agent.parse_obs(obs)[1]
            losses = agent.train_step(obs_t, action, reward, done, next_obs_t)
            obs_t = next_obs_t
            sum_rewards += reward
            discounted_reward += (DISCOUNT ** steps) * reward
            if steps % 100 == 0:
                print(f"Episode {episode}, Step {steps}, Reward: {reward:.3f}, Losses: {losses}")
                logging(episode, steps, reward, losses)
            steps += 1
        agent.hist_reset()
        print(f"Episode {episode} finished at step {steps} with sum reward {sum_rewards} and discounted reward {discounted_reward}")

    env.close()

main()