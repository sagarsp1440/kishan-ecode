from gym.envs.registration import register

register(
    id='HalfCheetahPT-v2',
    entry_point='policy_transfer.envs.mujoco:HalfCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='HopperPT-v2',
    entry_point='policy_transfer.envs.mujoco:HopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='Walker2dPT-v2',
    max_episode_steps=1000,
    entry_point='policy_transfer.envs.mujoco:Walker2dEnv',
)

register(
    id='Cartpole2dPT-v1',
    entry_point='policy_transfer.envs.gym.envs.classic_control:Cartpole2dTEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)