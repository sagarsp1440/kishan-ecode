import gym
import numpy as np
from gym import error, spaces

class EnvWrapper(gym.Wrapper):
    def __init__(self, env, up_dim):
        super().__init__(env)
        self.wrapped_env = env
        self.env = env.env # skip a wrapper for retaining other apis
        self.up_dim = up_dim


        # high = np.inf * np.ones(int(self.env.obs_dim / up_dim))
        # low = -high
        self.observation_space = env.observation_space
        
        self.action_space = env.action_space

    def process_raw_obs(self, raw_o):
        # one_obs_len = int((len(raw_o) - len(self.env.control_bounds[0]) * self.env.include_act_history) / self.env.include_obs_history)
        # pred_mu = self.osi.predict(raw_o)[0]
        # temp = np.concatenate([raw_o[0], self.up_dim])
        # return (temp, *raw_o[1:])
        return np.concatenate([raw_o, self.up_dim])

    def step(self, a):
        raw_o, r, terminated, d, dict = self.wrapped_env.step(a)
        return self.process_raw_obs(raw_o), r, terminated, d, dict

    def reset(self):
        raw_o, info = self.wrapped_env.reset()
        return self.process_raw_obs(raw_o), info

    # def render(self):
    #     return self.wrapped_env.render()
    
    # def render(self, mode):
    #     return self.wrapped_env.render(mode)

    def render(self, *args, **kwargs):
        default_args = {}
        kwargs = {**default_args, **kwargs}  #merging the two dictionaries
        return self.wrapped_env.render(*args, **kwargs)


    def close(self):
        self.wrapped_env.close()   

    @property
    def seed(self):
        return self.env.seed

    def pad_action(self, a):
        return self.env.pad_action(a)

    def unpad_action(self, a):
        return self.env.unpad_action(a)

    def about_to_contact(self):
        return self.env.about_to_contact()

    def state_vector(self):
        return self.env.state_vector()

    def set_state_vector(self, s):
        self.env.set_state_vector(s)

    def set_sim_parameters(self, pm):
        self.env.set_sim_parameters(pm)

    def get_sim_parameters(self):
        return self.env.get_sim_parameters()