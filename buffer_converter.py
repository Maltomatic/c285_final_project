import torch
import numpy as np
from controllers.TD3_controller import ReplayBuffer

OBS_DIM = 100  # 6*5*3 + 1*5*2
ACT_DIM = 6
CAPACITY = 1_000_000

EXP = "fault"

# Load old deque-based buffer
old_data = torch.load(f"{EXP}-td3_checkpoint_replay.pth", map_location="cpu", weights_only=False)
print(f"Loaded {len(old_data)} transitions")

# Create new buffer
new_buffer = ReplayBuffer(CAPACITY, OBS_DIM, ACT_DIM)

# Pour old tuples into new buffer
for obs, action, reward, done, next_obs in old_data:
    new_buffer.add(
        np.asarray(obs,      dtype=np.float32),
        np.asarray(action,   dtype=np.float32),
        float(reward),
        float(done),
        np.asarray(next_obs, dtype=np.float32),
    )

print(f"Converted {new_buffer.size} transitions")

# Save in new format
np.savez(
    f"{EXP}-td3_checkpoint_replay.npz",
    obs=new_buffer.obs[:new_buffer.size],
    next_obs=new_buffer.next_obs[:new_buffer.size],
    actions=new_buffer.actions[:new_buffer.size],
    rewards=new_buffer.rewards[:new_buffer.size],
    dones=new_buffer.dones[:new_buffer.size],
    ptr=np.array(new_buffer.ptr),
    size=np.array(new_buffer.size),
)
print("Saved to td3_checkpoint_replay.npz")