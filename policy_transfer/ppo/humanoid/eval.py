import gym
import numpy as np
import random
# from gym.wrappers.record_video import RecordVideo
from policy_transfer.policies.mlp_policy import MlpPolicy
# from pyvirtualdisplay import Display
import tensorflow as tf
import glob
import pickle
import joblib
from baselines.common import set_global_seeds, tf_util as U
from gym import spaces
from env_wrapper_render import EnvWrapper
import json
import time
import csv
# l_th = -0.34, l_sh = -0.3, l_f_s = 0.075, l_s_s = 0.049,l_t_s = 0.06 ------> Legs
# r_th = -0.34, r_sh = -0.3, r_f_s = 0.075, r_s_s = 0.049, r_t_s = 0.06
# l_u_a = .24, l_l_a = .17, l_h = 0.06 ----> Hands 
# r_u_a = .24, r_l_a = .17, r_h = 0.06

########## STABILITY ANALYSIS ##############
# - a +/-2% noise is added to each of the parameters to get the mean and std. dev from a collected 100point distribution for 16d humanoid vs 6d humanoid.
# - Results show, 16d's mean in lower than 6d's mean but 16d's std.dev is lower than 6d's.    

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

make_env = "Ant"
# "flat" # "incline" # "decline" # "16params"   
slope =  "flat" 

if make_env == "Humanoid":
    env = gym.make("Humanoid-v3", render_mode="human")
    env.metadata['render_fps'] = 120
    if slope == "16params":
        env.env.obs_dim = 436
    else:
        env.env.obs_dim = 382           #orig + latent   
elif make_env =="Ant":
    env = gym.make("Ant-v3", render_mode="human")
    env.metadata['render_fps'] = 24
    env.env.obs_dim = 111 # + 8

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

r_s_s = 0.049
r_t_s = 0.06
l_s_s = 0.049
l_t_s = 0.06

def add_noise(*params):
    np_params = np.asarray(params)
    np_params += np.random.uniform(np_params*0.02, -np_params*0.02, size=np_params.shape)
    return np_params
    

if make_env == "Humanoid":

    # all trained for 10.2 mil steps
    if slope == 'flat':

        # With ctrl_weight
        # pretrained_policy = "./render/humanoid/flat/policy_params.pkl"
        # l_th = -0.3886
        # l_sh = -0.23
        # r_th = -0.2697
        # r_sh = -0.3182
        # l_f_s = 0.0895
        # r_f_s = 0.0428

        # With NO ctrl_weight
        pretrained_policy = "./runs_humanoid/run_seed324_18750_1712719951/data/policy_params.pkl"

        l_th = 0.26
        l_sh = 0.33
        r_th = 0.4
        r_sh = 0.2
        l_f_s = 0.1
        r_f_s = 0.07

        # l_th, l_sh, l_f_s, r_th, r_sh, r_f_s = add_noise(l_th, l_sh, l_f_s, r_th, r_sh, r_f_s)
        extn_env = EnvWrapper(env, up_dim=[l_th, l_sh, l_f_s, r_th, r_sh, r_f_s])   #wrapped env

    if slope == 'incline':
        # With ctrl_weight
        # pretrained_policy = "./render/humanoid/incline/policy_params.pkl"   
        # l_th = -0.2979
        # l_sh = -0.2416
        # r_th = -0.3556
        # r_sh = -0.2942
        # l_f_s = 0.0596
        # r_f_s = 0.099

        # With NO ctrl_weight
        pretrained_policy = "./runs_humanoid/run_seed775_18750_1712715947/data/policy_params.pkl"   
        l_th = -0.5
        l_sh = -0.27
        r_th = -0.35
        r_sh = -0.31
        l_f_s = 0.075
        r_f_s = 0.056

        # l_th, l_sh, l_f_s, r_th, r_sh, r_f_s = add_noise(l_th, l_sh, l_f_s, r_th, r_sh, r_f_s)
        extn_env = EnvWrapper(env, up_dim=[l_th, l_sh, l_f_s, r_th, r_sh, r_f_s])   #wrapped env

    if slope == 'decline':
        # With ctrl_weight
        # pretrained_policy = "./render/humanoid/decline/policy_params.pkl"
        # l_th = -0.4091
        # l_sh = -0.175
        # r_th = -0.4247
        # r_sh = -0.3371
        # l_f_s = 0.0525
        # r_f_s = 0.0565
        
        # With NO ctrl_weight
        pretrained_policy = "./runs_humanoid/run_seed775_18750_1712716042/data/policy_params.pkl"
        l_th = -0.44
        l_sh = -0.17
        r_th = -0.44
        r_sh = -0.44
        l_f_s = 0.05
        r_f_s = 0.06

        # l_th, l_sh, l_f_s, r_th, r_sh, r_f_s = add_noise(l_th, l_sh, l_f_s, r_th, r_sh, r_f_s)
        extn_env = EnvWrapper(env, up_dim=[l_th, l_sh, l_f_s, r_th, r_sh, r_f_s])   #wrapped env

    elif slope == '16params':
        pretrained_policy = "./render/humanoid/16params/policy_params.pkl"	    
        l_th, l_sh, l_f_s, l_t_s, l_s_s, r_th, r_sh, r_f_s, r_t_s, r_s_s, l_u_a, l_l_a, l_h, r_u_a, r_l_a, r_h = -0.4514, -0.40205, 0.0846, 0.0676, 0.0295, -0.3680, -0.3295, 0.08271, 0.04202, 0.04734, 0.2558, 0.1243, 0.0437, 0.2209, 0.0857, 0.0738
        
        l_th, l_sh, l_f_s, l_t_s, l_s_s, r_th, r_sh, r_f_s, r_t_s, r_s_s, l_u_a, l_l_a, l_h, r_u_a, r_l_a, r_h = add_noise(l_th, l_sh, l_f_s, l_t_s, l_s_s, r_th, r_sh, r_f_s, r_t_s, r_s_s, l_u_a, l_l_a, l_h, r_u_a, r_l_a, r_h)

        extn_env = EnvWrapper(env, up_dim=[l_th, l_sh, l_f_s, l_t_s, l_s_s, r_th, r_sh, r_f_s, r_t_s, r_s_s, l_u_a, l_l_a, l_h, r_u_a, r_l_a, r_h])   #wrapped env


elif make_env =="Ant":
    # pretrained_policy = "./render/ant/policy_params.pkl"    # - with ctrl weight
    # fll,lfll,frl,lfrl,bll,lbll,brl,lbrl = 0.34, 0.345, -0.343, -0.328, -0.432, -0.272, 0.444, 0.347	#ant best params 
    

    pretrained_policy = "/home/knagiredla/.mujoco/mujoco210/myprojects/upnhbr/policy_transfer/ppo/logs444_10000000/policy_params.pkl"    
    # fll = 0.2
    # lfll = 0.1
    # frl = -0.2
    # lfrl = -0.15
    # bll = -0.2
    # lbll = -0.45
    # brl = 0.2
    # lbrl = 0.47	#ant with ctrl

    # pretrained_policy = "./runs_ant/run_seed67_110000_1712714333/data/policy_params.pkl"   #noctrl
    # fll,lfll,frl,lfrl,bll,lbll,brl,lbrl = 0.3, 0.1, 0.38, 0.27, 0.39, 0.22, 0.11, 0.19

    # fll,lfll,frl,lfrl,bll,lbll,brl,lbrl
    extn_env = EnvWrapper(env, up_dim=[])   #wrapped env

#extn_env = RecordVideo(extn_env, './video/humanoid', step_trigger=lambda step: step % 100 == 0, video_length=0)
# observation = extn_env.reset()

pi = init_policy(pi_init, joblib.load(pretrained_policy))

observation, _ = extn_env.reset()
eps_count = 0
counter = 0
dist = []
collect_actions = []

while counter < 1:
    eps_count = 0
    avg_r = 0
    sor = 0
    while eps_count<3: 
        #your agent goes here
        action, _ = pi.act(0.1, observation)
        # print("action", action, type(action))
        collect_actions.append(list(action))
        #action_sample = env.action_space.sample()     
        observation, reward, _, done, info = extn_env.step(action)  
        if done:
            eps_count = eps_count + 1
            observation, _ = extn_env.reset()
        extn_env.render()
        # time.sleep(.05)
        sor = sor + reward
        #print("reward", reward)
    # avg_r = sor/eps_count
    # print("average reward", avg_r)
    # dist.append(avg_r)
    counter = counter + 1
    extn_env.close()

# csv_file = 'actions.csv'

# with open(csv_file, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(collect_actions)



# np_dist = np.asarray(dist)
# dist_mean = np.mean(np_dist)
# dist_std = np.std(np_dist)

# print("dist_mean", np.mean(np_dist))
# print("dist_std", np.std(np_dist))


# with open("dist.json", 'w') as outfile:
#     json.dump(dist, outfile)

# data = {'dist_mean': dist_mean, 'dist_std': dist_std}
# with open("dist_result_16params.json", 'w+') as outfile:
#     json.dump(data, outfile)


