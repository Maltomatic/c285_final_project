import gymnasium as gym

gym.register(
    id="SixWheelSkidSteer-v0",
    entry_point="envs.six_wheel_env:SixWheelEnv",
    max_episode_steps=2000,
)
