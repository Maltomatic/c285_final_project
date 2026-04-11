import torch
import torch.nn as nn
import numpy as np
from collections import deque

from controllers.models.actor import Actor
from controllers.models.critic import Critic

OBS_DIM = 23
ACT_DIM = 6

class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        state_dim = 6*5*3 + 1*5*2 # 6 wheels, base/dev/actual ; 1 vel ; 1 ang
        action_dim = ACT_DIM
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.replay = []
        self.wheel_hist = deque(maxlen=5)
        self.base_hist = deque(maxlen=5)
        self.dev_hist = deque(maxlen=5)
        self.vels_hist = deque(maxlen=5)
        self.ang_hist = deque(maxlen=5)
        self.hist_reset()

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
    
    def hist_reset(self):
        self.wheel_hist.clear()
        self.base_hist.clear()
        self.dev_hist.clear()
        self.vels_hist.clear()
        self.ang_hist.clear()
        for _ in range(5):
            self.wheel_hist.append(np.zeros(6, dtype=np.float32))
            self.base_hist.append(np.zeros(6, dtype=np.float32))
            self.dev_hist.append(np.zeros(6, dtype=np.float32))
            self.vels_hist.append(0.0)
            self.ang_hist.append(0.0)

    def make_decision(self, obs):
        if type(obs) is not torch.Tensor:
            obs_dict, obs_t = self.parse_obs(obs)
        else:
            obs_t = obs
        action = self.actor(obs_t)
        return action

    def train_step(self, obs, action, reward, done, next_obs):
        # Placeholder for training step
        pass

    def parse_obs(self, obs):
        obs_dict = {}
        obs_dict["cross_track_err"] = obs[0]
        obs_dict["heading_err"] = obs[1]
        obs_dict["waypoint_dist"] = obs[2]
        obs_dict["bot_vel"] = obs[3]
        self.vels_hist.append(obs[3])
        obs_dict["bot_ang"] = obs[4]
        self.ang_hist.append(obs[4])
        obs_dict["curr_wheel"] = obs[5:11]
        self.wheel_hist.append(obs[5:11])
        obs_dict["curr_base"] = obs[11:17]
        self.base_hist.append(obs[11:17])
        obs_dict["curr_dev"] = obs[17:23]
        self.dev_hist.append(obs[17:23])

        _vels_hist = np.array(self.vels_hist)
        _ang_hist = np.array(self.ang_hist)
        _wheel_hist = np.concatenate(list(self.wheel_hist))
        _base_hist = np.concatenate(list(self.base_hist))
        _dev_hist = np.concatenate(list(self.dev_hist))

        obs_arr = np.concatenate((_vels_hist, _ang_hist, _wheel_hist, _base_hist, _dev_hist)).astype(np.float32)
        obs_t = torch.from_numpy(obs_arr)

        return obs_dict, obs_t