import numpy as np
import torch
import envs, gymnasium
from envs.six_wheel_env import SixWheelEnv
from controllers.TD3_controller import Agent

EPISODES = 100_000
DISCOUNT = 0.97
RENDER_TRAINING = True

def reset():
    env = SixWheelEnv()
    obs = env.reset()
    return obs

def main():
    env = SixWheelEnv(render_mode="human" if RENDER_TRAINING else None)
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
            agent.train_step(obs_t, action, reward, done, next_obs_t)
            obs_t = next_obs_t
            sum_rewards += reward
            discounted_reward += (DISCOUNT ** steps) * reward
            steps += 1
        print(f"Episode {episode} finished at step {steps} with sum reward {sum_rewards} and discounted reward {discounted_reward}")

    env.close()

main()