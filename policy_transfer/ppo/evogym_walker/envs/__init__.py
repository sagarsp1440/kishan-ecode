
# import envs and necessary gym packages
from policy_transfer.ppo.evogym_walker.envs.simple_env import SimpleWalkerEnv
from gym.envs.registration import register

# register the env using gym's interface
register(
    id = 'SimpleWalkingEnv-v0',
    entry_point = 'policy_transfer.ppo.evogym_walker.envs.simple_env:SimpleWalkerEnv',
    max_episode_steps = 1000
)
