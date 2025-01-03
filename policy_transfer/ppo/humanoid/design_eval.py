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
import json
import os

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


def policy_fn(name, ob_space, ac_space):
    return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                hid_size=64, num_hid_layers=3)


def add_noise(*params):
    np_params = np.asarray(params)
    np_params += np.random.uniform(np_params*0.02, -np_params*0.02, size=np_params.shape)
    return np_params


def main():
    import json
    import argparse

    parser2 = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser2.add_argument('--l_th', help='length', type=float, default=None)
    parser2.add_argument('--l_sh', help='length', type=float, default=None)
    parser2.add_argument('--r_th', help='length', type=float, default=None)
    parser2.add_argument('--r_sh', help='length', type=float, default=None)
    parser2.add_argument('--l_f_s', help='length', type=float, default=None)
    parser2.add_argument('--r_f_s', help='length', type=float, default=None)
    # parser2.add_argument('--l_t_s', help='mass', type=float, default=None)
    # parser2.add_argument('--r_t_s', help='mass', type=float, default=None)
    # parser2.add_argument('--l_s_s', help='mass', type=float, default=None)
    # parser2.add_argument('--r_s_s', help='mass', type=float, default=None)
    # parser2.add_argument('--l_u_a', help='length', type=float, default=None)
    # parser2.add_argument('--l_l_a', help='length', type=float, default=None)
    # parser2.add_argument('--l_h', help='length', type=float, default=None)
    # parser2.add_argument('--r_u_a', help='length', type=float, default=None)
    # parser2.add_argument('--r_l_a', help='length', type=float, default=None)
    # parser2.add_argument('--r_h', help='length', type=float, default=None)
    parser2.add_argument('--path', help='path for data')
    parser2.add_argument('--xml_path', help='path for xml')

    args3 = parser2.parse_args()
    folder_path = args3.path
    print("######### Length values for training #########", args3.l_th, args3.l_sh, args3.r_th, args3.r_sh) #, args3.l_u_a, args3.l_l_a, args3.l_h, args3.r_u_a, args3.r_l_a, args3.r_h)
    print("######### Size values for training #########", args3.l_f_s, args3.r_f_s) #, args3.l_t_s, args3.r_t_s,args3.l_s_s, args3.r_s_s)
    # print("######### n_iter value for training #########", args3.n_iter)

    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    seed = 0
    set_global_seeds(seed)

    #running gym session 

    make_env = "Humanoid"
    # "flat" # "incline" # "decline" # "16params"   
    slope =  "incline" 

    if make_env == "Humanoid":
        env = gym.make("Humanoid-v3", xml_file=args3.xml_path)
        if slope == "16params":
            env.env.obs_dim = 436
        else:
            env.env.obs_dim = 386           #orig + latent   

    high = np.inf * np.ones(env.env.obs_dim)
    low = -high
    env.env.observation_space = spaces.Box(low, high)
    env.observation_space = spaces.Box(low, high)

    ob_space = env.observation_space
    ac_space = env.action_space
    
    pi_init = policy_fn('pi', ob_space, ac_space) # Construct network for new policy

    l_th = args3.l_th
    l_sh = args3.l_sh
    r_th = args3.r_th
    r_sh = args3.l_sh
    l_f_s = args3.l_f_s
    r_f_s = args3.r_f_s
    l_t_s = 0.06
    r_t_s = 0.06
    l_s_s = 0.049
    r_s_s = 0.049
    
    # l_t_s = args3.l_t_s
    # r_t_s = args3.r_t_s
    # l_s_s = args3.l_s_s
    # r_s_s = args3.r_s_s
    # l_u_a = args3.l_u_a
    # l_l_a = args3.l_l_a
    # l_h = args3.l_h
    # r_u_a = args3.r_u_a
    # r_l_a = args3.r_l_a
    # r_h = args3.r_h

    extn_env = EnvWrapper(env, up_dim=[l_th, l_sh, l_f_s, l_t_s, l_s_s, r_th, r_sh, r_f_s, r_t_s, r_s_s])   #wrapped env

    pretrained_policy = "./render/humanoid/incline/policy_params.pkl"	   
    pi = init_policy(pi_init, joblib.load(pretrained_policy))

    observation, _ = extn_env.reset()
    eps_count = 0
    counter = 0
    dist = []

    observation, _ = extn_env.reset()
    eps_count = 0
    sor = 0

    while eps_count<5: 
        action, _ = pi.act(0.1, observation)
        observation, reward, _, done, info = extn_env.step(action)  
        if done:
            eps_count = eps_count + 1
            observation, _ = extn_env.reset()
        #extn_env.render()
        sor = sor + reward
        #print("reward", reward)
    avg_r = sor/eps_count
    print("average reward", avg_r)
    extn_env.close()

    # np_dist = np.asarray(dist)
    # dist_mean = np.mean(np_dist)
    # dist_std = np.std(np_dist)

    # print("dist_mean", np.mean(np_dist))
    # print("dist_std", np.std(np_dist))

    with open(os.path.join(folder_path, "last_return.json"), 'w+') as outfile:
        json.dump(avg_r, outfile)

    # data = {'dist_mean': dist_mean, 'dist_std': dist_std}
    # with open("dist_result_16params.json", 'w+') as outfile:
    #     json.dump(data, outfile)



#######################################################################################

if __name__ == '__main__':
    main()

