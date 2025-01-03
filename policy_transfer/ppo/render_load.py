import gym
import numpy as np
import random
from gym.wrappers.record_video import RecordVideo
from policy_transfer.policies.mlp_policy import MlpPolicy
from pyvirtualdisplay import Display
import tensorflow as tf
import glob
import pickle
import joblib
from baselines.common import set_global_seeds, tf_util as U
from gym import spaces
from env_wrapper_render import EnvWrapper


################################################################
#
# This file is copied over to policy_transfer project folder from Rendering_test folder
#
################################################################

def init_policy(pi, init_policy_params):

    U.initialize()

    cur_scope = pi.get_variables()[0].name[0:pi.get_variables()[0].name.find('/')]
    orig_scope = list(init_policy_params.keys())[0][0:list(init_policy_params.keys())[0].find('/')]
    print(cur_scope, orig_scope)
    for i in range(len(pi.get_variables())):
        if pi.get_variables()[i].name.replace(cur_scope, orig_scope, 1) in init_policy_params:
            assign_op = pi.get_variables()[i].assign(init_policy_params[pi.get_variables()[i].name.replace(cur_scope, orig_scope, 1)])
            tf.get_default_session().run(assign_op)
    
    return pi

sess = U.make_session(num_cpu=1)
sess.__enter__()
seed = 0
set_global_seeds(seed)

#running gym session 

make_env = "Humanoid" 

if make_env == "Humanoid":
    env = gym.make("Humanoid-v4", render_mode="rgb_array")
    env.env.obs_dim = 382  #orig + latent   #382 - 6d or 436 - 16d
elif make_env =="Ant":
    env = gym.make("Ant-v3", render_mode="human")
    env.env.obs_dim = 111 + 8 

high = np.inf * np.ones(env.env.obs_dim)
low = -high
env.env.observation_space = spaces.Box(low, high)
env.observation_space = spaces.Box(low, high)

# env = RecordVideo(env, './video',  episode_trigger = lambda episode_number: True, video_length=100)
# env.render_mode = "human"
# env.env.render_mode = "human"

ob_space = env.observation_space
ac_space = env.action_space

def policy_fn(name, ob_space, ac_space):
    return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                            hid_size=64, num_hid_layers=3)

pi_init = policy_fn('pi', ob_space, ac_space) # Construct network for new policy


if make_env == "Humanoid":

    pretrained_policy = "./render/humanoid/flat/policy_params.pkl"
    # pretrained_policy = "./render/humanoid/incline/policy_params.pkl"    
    # pretrained_policy = "./render/humanoid/decline/policy_params.pkl"
    # pretrained_policy = "./render/humanoid/16params/policy_params.pkl"	    

elif make_env =="Ant":

    pretrained_policy = "./render/ant/policy_params.pkl"    

pi = init_policy(pi_init, joblib.load(pretrained_policy))


if make_env == "Humanoid":
# Humanoid ---------------------

    #16params
    # l_th, l_sh, l_f_s, l_t_s, l_s_s, r_th, r_sh, r_f_s, r_t_s, r_s_s, l_u_a, l_l_a, l_h, r_u_a, r_l_a, r_h = -0.4514, -0.40205, 0.0846, 0.0676, 0.0295, -0.3680, -0.3295, 0.08271, 0.04202, 0.04734, 0.2558, 0.1243, 0.0437, 0.2209, 0.0857, 0.0738
    # extn_env = EnvWrapper(env, up_dim=[l_th, l_sh, l_f_s, l_t_s, l_s_s, r_th, r_sh, r_f_s, r_t_s, r_s_s, l_u_a, l_l_a, l_h, r_u_a, r_l_a, r_h])   #wrapped env

    #flat
    # l_th = -0.3886
    # l_sh = -0.23
    # r_th = -0.2697
    # r_sh = -0.3182
    # l_f_s = 0.0895
    # r_f_s = 0.0428

    #incline
    # l_th = -0.2979
    # l_sh = -0.2416
    # r_th = -0.3556
    # r_sh = -0.2942
    # l_f_s = 0.0596
    # r_f_s = 0.099
    # r_s_s = 0.049
    # r_t_s = 0.06
    # l_s_s = 0.049
    # l_t_s = 0.06


    # decline
    # l_th = -0.4091
    # l_sh = -0.175
    # r_th = -0.4247
    # r_sh = -0.3371
    # l_f_s = 0.0525
    # r_f_s = 0.0565
    # r_s_s = 0.049
    # r_t_s = 0.06
    # l_s_s = 0.049
    # l_t_s = 0.06

    #default
    l_th = -0.34 
    l_sh = -0.3 
    r_th = -0.34 
    r_sh = -0.3 
    r_s_s = 0.049
    r_t_s = 0.06
    l_s_s = 0.049
    l_t_s = 0.06
    l_f_s = 0.075
    r_f_s = 0.075
    l_u_a = .24
    l_l_a = .17
    l_h = 0.06 
    r_u_a = .24 
    r_l_a = .17
    r_h = 0.06

    extn_env = EnvWrapper(env, up_dim=[l_th, l_sh, l_f_s, r_th, r_sh, r_f_s])   #wrapped env

    
    
elif make_env =="Ant":
# Ant ---------------------
    fll,lfll,frl,lfrl,bll,lbll,brl,lbrl = 0.34, 0.345, -0.343, -0.328, -0.432, -0.272, 0.444, 0.347	#ant best params

    extn_env = EnvWrapper(env, up_dim=[fll,lfll,frl,lfrl,bll,lbll,brl,lbrl])   #wrapped env

#extn_env = RecordVideo(extn_env, './video/humanoid', step_trigger=lambda step: step % 100 == 0, video_length=0)

# observation = extn_env.reset()

observation, _ = extn_env.reset()
eps_count = 0
sor = 0

while eps_count<5: 
    #your agent goes here
    action, _ = pi.act(0.1, observation)
    #action_sample = env.action_space.sample()     
    observation, reward, _, done, info = extn_env.step(action)  
    if done:
        eps_count = eps_count + 1
        observation, _ = extn_env.reset()
    #extn_env.render()
    sor = sor + reward
    #print("reward", reward)
    extn_env.render()
avg_r = sor/eps_count
print("average reward", avg_r)
extn_env.close()

