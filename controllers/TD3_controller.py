from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import copy

from controllers.models.actor import Actor
from controllers.models.critic import Critic

from controllers.utils.replay import ReplayBuffer

from envs.env_configs import _ACTION_CLIP, _ACTION_DIM as ACT_DIM, EVAL
import envs.env_configs as env_config

from controllers.utils.model_configs import EPS_START, EPS_DECAY, EPS_MIN, CAPACITY, G_STEPS, FT_RATIO

COMPILE = False

class Agent(nn.Module):
    def __init__(self, num_envs = 1, checkpoint_path="td3_checkpoint"):
        super(Agent, self).__init__()
        self.num_envs = num_envs
        if not env_config.EVAL:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu") # without training CPU is faster
        k = env_config._OBS_STACK
        # Residual obs: k × (vel + ang + 6 wheels + 6 base + 6 dev) = k × 20
        # Pure RL obs:  k × (heading + dist + vel + ang + 6 wheels + 6 act) = k × 16
        self.state_dim = k * 16 if env_config.PURE_RL else k * 20
        action_dim = ACT_DIM
        self.actor = Actor(self.state_dim, action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic1 = Critic(self.state_dim, action_dim).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2 = Critic(self.state_dim, action_dim).to(self.device)
        self.critic2_target = copy.deepcopy(self.critic2)
        # compile for faster training
        if self.device.type == 'cuda' and COMPILE:
            self.actor = torch.compile(self.actor)
            self.critic1 = torch.compile(self.critic1)
            self.critic2 = torch.compile(self.critic2)
            self.actor_target = torch.compile(self.actor_target)
            self.critic1_target = torch.compile(self.critic1_target)
            self.critic2_target = torch.compile(self.critic2_target)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=1e-3)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=1e-3)

        self.replay = ReplayBuffer(CAPACITY, self.state_dim, ACT_DIM) if not EVAL else None
        self.batch_size = 1024
        self.gamma = 0.995
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.max_action = _ACTION_CLIP if not env_config.PURE_RL else _ACTION_CLIP * 2
        self.action_scale = self.max_action
        self.max_grad_norm = 3.0
        self.total_it = 0
        self._loss = None # store most recent loss for logging

        self.wheel_hist = [deque(maxlen=env_config._OBS_STACK) for _ in range(num_envs)]
        if not env_config.PURE_RL:
            self.base_hist = [deque(maxlen=env_config._OBS_STACK) for _ in range(num_envs)]
            self.dev_hist = [deque(maxlen=env_config._OBS_STACK) for _ in range(num_envs)]
        else:
            self.act_hist = [deque(maxlen=env_config._OBS_STACK) for _ in range(num_envs)]
            self.heading_hist = [deque(maxlen=env_config._OBS_STACK) for _ in range(num_envs)]
            self.dist_hist = [deque(maxlen=env_config._OBS_STACK) for _ in range(num_envs)]
        self.vels_hist = [deque(maxlen=env_config._OBS_STACK) for _ in range(num_envs)]
        self.ang_hist = [deque(maxlen=env_config._OBS_STACK) for _ in range(num_envs)]
        self.hist_reset()

        self.epsilon = EPS_START
        self.epsilon_decay = EPS_DECAY
        if env_config.FINE_TUNE:
            self.epsilon_decay = 0.99978
        self.epsilon_min = EPS_MIN
        self.ft_start_step = int(G_STEPS * FT_RATIO)
        self.ft_reset = False

        self.checkpoint_load(checkpoint_path)

        self.printout()
    
    def printout(self):
        # print out hyperparameters
        print(f"TD3 {'Non-Residual' if env_config.PURE_RL else 'Residual'} Agent initialized with:")
        print(f"\tDevice: {self.device}")
        print(f"\tReplay buffer capacity: {CAPACITY}")
        print(f"\tBatch size: {self.batch_size}")
        print(f"\tGamma (discount): {self.gamma}")
        print(f"\tTau (target update rate): {self.tau}")
        print(f"\tPolicy noise: {self.policy_noise}")
        print(f"\tNoise clip: {self.noise_clip}")
        print(f"\tPolicy update frequency: {self.policy_freq}")
        print(f"\tMax action: {self.max_action}")
        print(f"\tMax grad norm: {self.max_grad_norm}")
        # print(f"\tEpsilon start: {EPS_START}")
        # print(f"\tEpsilon decay: {EPS_DECAY}")
        # print(f"\tEpsilon min: {EPS_MIN}")
    
    def set_for_fine_tune(self):
        # set learning rate
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = 1e-5
        for param_group in self.critic1_optimizer.param_groups:
            param_group['lr'] = 1e-4
        for param_group in self.critic2_optimizer.param_groups:
            param_group['lr'] = 1e-4
        # # freeze all but last 2 layers of actor and critics
        # for param in self.actor.parameters():
        #     param.requires_grad = False
        # for param in list(self.actor.parameters())[-4:]:
        #     param.requires_grad = True
        # for critic in [self.critic1, self.critic2]:
        #     for param in critic.parameters():
        #         param.requires_grad = False
        #     for param in list(critic.parameters())[-4:]:
        #         param.requires_grad = True
        # clear histories
        self.hist_reset()
        # clear replay buffer
        del self.replay
        self.replay = ReplayBuffer(CAPACITY, self.state_dim, ACT_DIM)

        self.epsilon = EPS_START
        self.epsilon_decay = 0.99945

        print("Agent is reset for fine tuning.")

        self.ft_reset = True

    def hist_reset(self):
        for env_id in range(self.num_envs):
            self.hist_reset_single_env(env_id)
    
    def hist_reset_single_env(self, env_id):
        self.wheel_hist[env_id].clear()
        if not env_config.PURE_RL:
            self.base_hist[env_id].clear()
            self.dev_hist[env_id].clear()
        else:
            self.act_hist[env_id].clear()
            self.heading_hist[env_id].clear()
            self.dist_hist[env_id].clear()
        self.vels_hist[env_id].clear()
        self.ang_hist[env_id].clear()
        for _ in range(env_config._OBS_STACK):
            self.wheel_hist[env_id].append(np.zeros(6, dtype=np.float32))
            if not env_config.PURE_RL:
                self.base_hist[env_id].append(np.zeros(6, dtype=np.float32))
                self.dev_hist[env_id].append(np.zeros(6, dtype=np.float32))
            else:
                self.act_hist[env_id].append(np.zeros(6, dtype=np.float32))
                self.heading_hist[env_id].append(0.0)
                self.dist_hist[env_id].append(0.0)
            self.vels_hist[env_id].append(0.0)
            self.ang_hist[env_id].append(0.0)
    
    def parse_obs(self, obs, device=None, env_id=0):
        # if obs is tensor, return None and obs
        if type(obs) is torch.Tensor:
            return None, obs.to(self.device)
        obs_dict = {}
        # obs_dict["cross_track_err"] = obs[0]
        if env_config.PURE_RL:
            obs_dict["heading_err"] = obs[1]
            self.heading_hist[env_id].append(obs[1])
            obs_dict["waypoint_dist"] = obs[2]
            self.dist_hist[env_id].append(obs[2])
        obs_dict["bot_vel"] = obs[3]
        self.vels_hist[env_id].append(obs[3])
        obs_dict["bot_ang"] = obs[4]
        self.ang_hist[env_id].append(obs[4])
        obs_dict["curr_wheel"] = obs[5:11]
        self.wheel_hist[env_id].append(obs[5:11])
        if not env_config.PURE_RL:
            obs_dict["curr_base"] = obs[11:17]
            self.base_hist[env_id].append(obs[11:17])
            obs_dict["curr_dev"] = obs[17:23]
            self.dev_hist[env_id].append(obs[17:23])
            _base_hist = np.concatenate(list(self.base_hist[env_id]))
            _dev_hist = np.concatenate(list(self.dev_hist[env_id]))
        else:
            obs_dict["prev_act"] = obs[11:17]
            self.act_hist[env_id].append(obs[11:17])
            _act_hist = np.concatenate(list(self.act_hist[env_id]))
            _heading_hist = np.array(self.heading_hist[env_id])
            _dist_hist = np.array(self.dist_hist[env_id])

        _wheel_hist = np.concatenate(list(self.wheel_hist[env_id]))
        _vels_hist = np.array(self.vels_hist[env_id])
        _ang_hist = np.array(self.ang_hist[env_id])

        if not env_config.PURE_RL:
            obs_arr = np.concatenate((_vels_hist, _ang_hist, _wheel_hist, _base_hist, _dev_hist)).astype(np.float32)
        else:
            obs_arr = np.concatenate((_heading_hist, _dist_hist, _vels_hist, _ang_hist, _wheel_hist, _act_hist)).astype(np.float32)
        target_device = self.device if device is None else torch.device(device)
        obs_t = torch.from_numpy(obs_arr).to(target_device)

        return obs_dict, obs_t

    def make_decision(self, obs, explore: bool = True, env_id=0, steps=0):
        if type(obs) is not torch.Tensor:
            obs_t = torch.stack([self.parse_obs(obs[i], env_id=i)[1] for i in range(self.num_envs)])
        else:
            obs_t = obs.to(self.device)
        # obs_t is batched over N envs
        N = obs_t.shape[0]
        # print(f"Shape of obs_t in make_decision: {obs_t.shape}")
        if not EVAL and env_config.FINE_TUNE and steps >= self.ft_start_step and not self.ft_reset:
            self.set_for_fine_tune()
        if explore and np.random.rand() < self.epsilon:
            # if steps < 0.05 * G_STEPS or (self.ft_reset and steps < self.ft_start_step + 0.001 * G_STEPS):
            action = torch.empty(N, ACT_DIM, dtype=torch.float32, device=self.device).uniform_(-self.max_action, self.max_action)
            # else:
            #     with torch.no_grad():
            #         action = self.actor(obs_t)
        else:
            with torch.no_grad():
                action = self.actor(obs_t)# * self.max_action # target instead of actor to avoid gpu/cpu race
                # print(f"Action before clamp: {action}")
                # action = torch.clamp(action, -self.max_action, self.max_action)

        return action

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def save_transition(self, obs, action, reward, done, next_obs):
        obs_np = obs.detach().cpu().numpy() if isinstance(obs, torch.Tensor) else np.asarray(obs, dtype=np.float32)
        next_obs_np = (next_obs.detach().cpu().numpy() if isinstance(next_obs, torch.Tensor) else np.asarray(next_obs, dtype=np.float32))
        action_np = np.asarray(action, dtype=np.float32)
        done = float(done)
        self.replay.add(obs_np, action_np, reward, done, next_obs_np)

    def train_step(self):
        if len(self.replay) < self.batch_size:
            return None

        self.total_it += 1

        # Sample batch
        states, actions, rewards, dones, next_states = self.replay.sample(self.batch_size, device=self.device)
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise) * self.action_scale
            noise_clip = torch.full_like(actions, self.noise_clip) * self.action_scale
            noise = torch.clamp(noise, -noise_clip, noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)
            next_state_action = torch.cat([next_states, next_actions], dim=1)
            target_q1 = self.critic1_target(next_state_action)
            target_q2 = self.critic2_target(next_state_action)
            target_q = rewards + (1.0 - dones) * self.gamma * torch.min(target_q1, target_q2)

        state_action = torch.cat([states, actions], dim=1)
        current_q1 = self.critic1(state_action)
        current_q2 = self.critic2(state_action)

        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), self.max_grad_norm)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        nn.utils.clip_grad_norm_(self.critic2.parameters(), self.max_grad_norm)
        self.critic2_optimizer.step()

        actor_loss = self._loss
        if self.total_it % self.policy_freq == 0:
            pi_raw = self.actor(states)
            pi_actions = pi_raw.clamp(-self.max_action, self.max_action)
            pi_state_action = torch.cat([states, pi_actions], dim=1)
            actor_loss = -self.critic1(pi_state_action).mean() + 1e-3 * (pi_raw**2).mean()
            self._loss = actor_loss.detach().cpu()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

        return {
            "critic1_loss": float(critic1_loss.item()),
            "critic2_loss": float(critic2_loss.item()),
            "actor_loss": float(actor_loss.item()) if actor_loss is not None else None,
        }

    def unwrap(self, m):
        return m._orig_mod if hasattr(m, '_orig_mod') else m

    def checkpoint_save(self, path):
        torch.save({
            'actor': self.unwrap(self.actor).state_dict(),
            'critic1': self.unwrap(self.critic1).state_dict(),
            'critic2': self.unwrap(self.critic2).state_dict(),
            'actor_target': self.unwrap(self.actor_target).state_dict(),
            'critic1_target': self.unwrap(self.critic1_target).state_dict(),
            'critic2_target': self.unwrap(self.critic2_target).state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path + ".pth")
        # save_replay = input("Save replay buffer? (y/n): ")
        sz = len(self.replay)
        np.savez(path + "_replay.npz",
            obs=self.replay.obs[:sz],
            next_obs=self.replay.next_obs[:sz],
            actions=self.replay.actions[:sz],
            rewards=self.replay.rewards[:sz],
            dones=self.replay.dones[:sz],
        )
    
    def checkpoint_load(self, path):
        try:
            ckpt_path = path + ".pth"
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.unwrap(self.actor).load_state_dict(checkpoint['actor'])
            self.unwrap(self.critic1).load_state_dict(checkpoint['critic1'])
            self.unwrap(self.critic2).load_state_dict(checkpoint['critic2'])
            self.unwrap(self.actor_target).load_state_dict(checkpoint['actor_target'])
            self.unwrap(self.critic1_target).load_state_dict(checkpoint['critic1_target'])
            self.unwrap(self.critic2_target).load_state_dict(checkpoint['critic2_target'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
            self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
            self.epsilon = float(checkpoint.get('epsilon', self.epsilon))
            self._optimizer_to_device(self.actor_optimizer)
            self._optimizer_to_device(self.critic1_optimizer)
            self._optimizer_to_device(self.critic2_optimizer)
            print(f"Model checkpoint loaded from {ckpt_path}.")
        except FileNotFoundError:
            print(f"No checkpoint found at {ckpt_path}. Starting with uninitialized model.")
        if not EVAL:
            try:
                replay_path = path + "_replay.npz"
                data = np.load(replay_path)
                self.replay.obs[:len(data['obs'])] = data['obs']
                self.replay.next_obs[:len(data['next_obs'])] = data['next_obs']
                self.replay.actions[:len(data['actions'])] = data['actions']
                self.replay.rewards[:len(data['rewards'])] = data['rewards']
                self.replay.dones[:len(data['dones'])] = data['dones']
                self.replay.size = len(data['obs'])
                self.replay.ptr = self.replay.size % self.replay.capacity
                print(f"Replay buffer loaded from {replay_path} with size {self.replay.size}.")
            except FileNotFoundError:
                print(f"No replay buffer found at {replay_path}, starting with empty buffer.")

    def _soft_update(self, source: nn.Module, target: nn.Module):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    def _optimizer_to_device(self, optimizer):
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
