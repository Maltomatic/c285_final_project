import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import copy

from controllers.models.actor import Actor
from controllers.models.critic import Critic

from controllers.utils.replay import ReplayBuffer

OBS_DIM = 23
STACK_OBS_DIM = 6*5*3 + 1*5*2 # 6 wheels, base/dev/actual ; 1 vel ; 1 ang
ACT_DIM = 6

COMPILE = False
CAPACITY = 1_000_000

class Agent(nn.Module):
    def __init__(self, num_envs = 1, checkpoint_path="td3_checkpoint"):
        super(Agent, self).__init__()
        self.num_envs = num_envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dim = STACK_OBS_DIM
        action_dim = ACT_DIM
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2 = Critic(state_dim, action_dim).to(self.device)
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

        self.replay = ReplayBuffer(CAPACITY, STACK_OBS_DIM, ACT_DIM)
        self.batch_size = 1024
        self.gamma = 0.997
        self.tau = 0.005
        self.policy_noise = 0.5
        self.noise_clip = 1.5
        self.policy_freq = 2
        self.max_action = 15.0
        self.max_grad_norm = 3.0
        self.total_it = 0
        self._loss = None # store most recent loss for logging

        self.wheel_hist = [deque(maxlen=5) for _ in range(num_envs)]
        self.base_hist = [deque(maxlen=5) for _ in range(num_envs)]
        self.dev_hist = [deque(maxlen=5) for _ in range(num_envs)]
        self.vels_hist = [deque(maxlen=5) for _ in range(num_envs)]
        self.ang_hist = [deque(maxlen=5) for _ in range(num_envs)]
        self.hist_reset()

        self.epsilon = 0.45
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.decay_steps = 10

        self.checkpoint_load(checkpoint_path)
    def hist_reset(self):
        for env_id in range(self.num_envs):
            self.hist_reset_single_env(env_id)
    
    def hist_reset_single_env(self, env_id):
        self.wheel_hist[env_id].clear()
        self.base_hist[env_id].clear()
        self.dev_hist[env_id].clear()
        self.vels_hist[env_id].clear()
        self.ang_hist[env_id].clear()
        for _ in range(5):
            self.wheel_hist[env_id].append(np.zeros(6, dtype=np.float32))
            self.base_hist[env_id].append(np.zeros(6, dtype=np.float32))
            self.dev_hist[env_id].append(np.zeros(6, dtype=np.float32))
            self.vels_hist[env_id].append(0.0)
            self.ang_hist[env_id].append(0.0)
    
    def parse_obs(self, obs, device=None, env_id=0):
        obs_dict = {}
        obs_dict["cross_track_err"] = obs[0]
        obs_dict["heading_err"] = obs[1]
        obs_dict["waypoint_dist"] = obs[2]
        obs_dict["bot_vel"] = obs[3]
        self.vels_hist[env_id].append(obs[3])
        obs_dict["bot_ang"] = obs[4]
        self.ang_hist[env_id].append(obs[4])
        obs_dict["curr_wheel"] = obs[5:11]
        self.wheel_hist[env_id].append(obs[5:11])
        obs_dict["curr_base"] = obs[11:17]
        self.base_hist[env_id].append(obs[11:17])
        obs_dict["curr_dev"] = obs[17:23]
        self.dev_hist[env_id].append(obs[17:23])

        _vels_hist = np.array(self.vels_hist[env_id])
        _ang_hist = np.array(self.ang_hist[env_id])
        _wheel_hist = np.concatenate(list(self.wheel_hist[env_id]))
        _base_hist = np.concatenate(list(self.base_hist[env_id]))
        _dev_hist = np.concatenate(list(self.dev_hist[env_id]))

        obs_arr = np.concatenate((_vels_hist, _ang_hist, _wheel_hist, _base_hist, _dev_hist)).astype(np.float32)
        target_device = self.device if device is None else torch.device(device)
        obs_t = torch.from_numpy(obs_arr).to(target_device)

        return obs_dict, obs_t

    def make_decision(self, obs, explore: bool = True, env_id=0):
        if type(obs) is not torch.Tensor:
            obs_t = torch.stack([self.parse_obs(obs[i], env_id=i)[1] for i in range(self.num_envs)])
        else:
            obs_t = obs.to(self.device)
        # obs_t is batched over N envs
        N = obs_t.shape[0]
        if explore and np.random.rand() < self.epsilon:
            action = torch.empty(N, ACT_DIM, dtype=torch.float32, device=self.device).uniform_(-self.max_action, self.max_action)
        else:
            with torch.no_grad():
                action = self.actor(obs_t)# * self.max_action # target instead of actor to avoid gpu/cpu race
                # print(f"Action before clamp: {action}")
                # action = torch.clamp(action, -self.max_action, self.max_action)

        return action

    def decay_epsilon(self):
        self.decay_steps -= 1
        if self.decay_steps <= 0:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.decay_steps = 10

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
        idx = np.random.choice(len(self.replay), size=self.batch_size, replace=False)
        states, actions, rewards, dones, next_states = self.replay.sample(self.batch_size, device=self.device)
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) * self.max_action + noise).clamp(-self.max_action, self.max_action)
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
            pi_actions = self.actor(states) * self.max_action
            pi_state_action = torch.cat([states, pi_actions], dim=1)
            actor_loss = -self.critic1(pi_state_action).mean()
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
        except FileNotFoundError:
            print(f"No checkpoint found at {ckpt_path}. Starting with uninitialized model.")
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
