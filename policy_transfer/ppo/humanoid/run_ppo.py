#!/usr/bin/env python
from statistics import mean
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os
import os.path as osp
import gym, logging
#import policy_transfer.envs
import sys
import joblib
import tensorflow as tf
import numpy as np
from mpi4py import MPI
from policy_transfer.policies.mirror_policy import *
from policy_transfer.policies.mlp_policy import MlpPolicy
from env_wrapper import EnvWrapper
from baselines import logger
from matplotlib import pyplot as plt


output_interval = 10
# log_dir = None

def callback(localv, globalv):
    global log_dir
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(globalv.keys())
    save_dict = {}
    variables = localv['pi'].get_variables()
    for i in range(len(variables)):
        cur_val = variables[i].eval()
        save_dict[variables[i].name] = cur_val
    print("########log_dir", log_dir)
    joblib.dump(save_dict, log_dir + '/policy_params' + '.pkl', compress=True)
    if localv['iters_so_far'] % output_interval != 0:
        return
    joblib.dump(save_dict, log_dir + '/policy_params_'+str(localv['iters_so_far'])+'.pkl', compress=True)


def policy_fn(name, ob_space, ac_space):
    return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        hid_size=64, num_hid_layers=3)


def train(l_th, l_sh, l_f_s, r_th, r_sh, r_f_s, n_iterations, env_id, num_timesteps, seed, batch_size, clip, schedule, mirror, warmstart, train_up, folder_path, xml_path):
    #l_t_s, l_s_s,  r_t_s, r_s_s,
        # l_u_a, l_l_a, l_h, r_u_a, r_l_a, r_h,
    
    from policy_transfer.ppo.humanoid import ppo_sgd
    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    set_global_seeds(seed)

    env = gym.make(env_id, xml_file=xml_path)
    if train_up:
        if env.env.train_UP is not True:
            env.env.train_UP = True
            from gym import spaces

            env.env.obs_dim = 376 + 6  # 376 because obs size is 376; +16 because 16 latent parameters

            high = np.inf * np.ones(env.env.obs_dim)
            low = -high
            env.env.observation_space = spaces.Box(low, high)
            env.observation_space = spaces.Box(low, high)

            # if hasattr(env.env, 'obs_perm'):
            #     obpermapp = np.arange(len(env.env.obs_perm), len(env.env.obs_perm) + len(env.env.param_manager.activated_param))
            #     env.env.obs_perm = np.concatenate([env.env.obs_perm, obpermapp])

    with open(logger.get_dir()+"/envinfo.txt", "w") as text_file:
        text_file.write(str(env.env.__dict__))

    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), "monitor.json"), allow_early_resets=True)
    env.seed(seed+MPI.COMM_WORLD.Get_rank())

    gym.logger.setLevel(logging.WARN)

    pol_func = policy_fn

    if len(warmstart) > 0:
        warmstart_params = joblib.load(warmstart)
        print("Loaded parameters from UPN_PI path")
    else:
        warmstart_params = None

    # extn_env = EnvWrapper(env, up_dim=[l_th, l_sh, l_f_s, l_t_s, l_s_s, r_th, r_sh, r_f_s, r_t_s, r_s_s, l_u_a, l_l_a, l_h, r_u_a, r_l_a, r_h])   #wrapped env

    extn_env = EnvWrapper(env, up_dim=[l_th, l_sh, l_f_s, r_th, r_sh, r_f_s])   #wrapped env

    timesteps = num_timesteps * n_iterations
    batch_size = 4000

    pi, avg_training_reward = ppo_sgd.learn(extn_env, pol_func, folder_path, max_timesteps=timesteps, timesteps_per_batch=int(batch_size),
            clip_param=clip, entcoeff=0.0, optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=4000, gamma=0.99, lam=0.95, 
            schedule=schedule, callback=callback, init_policy_params=warmstart_params)


    print("pi", pi)
    print("mean rew", avg_training_reward)
    
    sess.close()

    return pi, avg_training_reward


def main():
    import json
    import argparse
    global log_dir

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

    parser2.add_argument('--n_iter', help='n_iterations', type=float, default=None)
    parser2.add_argument('--warmstart', help='path to warmstart policies', type=str, default="")
    parser2.add_argument('--env', help='environment ID', default=None)
    parser2.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser2.add_argument('--name', help='name of experiments', type=str, default="")
    parser2.add_argument('--min_timesteps', help='maximum step size', type=int)
    parser2.add_argument('--batch_size', help='batch size', type=int, default=32)
    parser2.add_argument('--clip', help='clip', type=float, default=0.2)
    parser2.add_argument('--schedule', help='schedule', default='constant')
    parser2.add_argument('--train_up', help='whether train up', default='True')
    parser2.add_argument('--output_interval', help='interval of outputting policies', type=int, default=10)
    parser2.add_argument('--mirror', help='whether to use mirror, (0: not mirror, 1: hard mirror, 2: soft mirror)', type=int, default=0)
    parser2.add_argument('--path', help='path for data')
    parser2.add_argument('--xml_path', help='path for xml')

    args3 = parser2.parse_args()
    # print("######### Length values for training #########", args3.l_th, args3.l_sh, args3.r_th, args3.r_sh, args3.l_u_a, args3.l_l_a, args3.l_h, args3.r_u_a, args3.r_l_a, args3.r_h)
    # print("######### Size values for training #########", args3.l_f_s, args3.r_f_s, args3.l_t_s, args3.r_t_s,args3.l_s_s, args3.r_s_s)
    print("######### n_iter value for training #########", args3.n_iter)

    l_th = args3.l_th
    l_sh = args3.l_sh
    r_th = args3.r_th
    r_sh = args3.l_sh
    l_f_s = args3.l_f_s
    r_f_s = args3.r_f_s
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

    n_iterations = args3.n_iter
    log_dir = args3.path

    pi, avg_rew = train(l_th, l_sh, l_f_s, r_th, r_sh, r_f_s, 
                        n_iterations, args3.env, num_timesteps=int(args3.min_timesteps), 
                        seed=args3.seed, batch_size=args3.batch_size, clip=args3.clip, 
                        schedule=args3.schedule, mirror=args3.mirror, warmstart=args3.warmstart, 
                        train_up=args3.train_up, folder_path = args3.path, xml_path=args3.xml_path)
    
    #l_t_s, l_s_s, r_t_s, r_s_s,
    # l_u_a, l_l_a, l_h, r_u_a, r_l_a, r_h,

    print("PI", pi)
    print("log_dir", log_dir)
    config_name = args3.path
    data = {'pi': log_dir + '/policy_params' + '.pkl', 'avg_rew': avg_rew}
    with open(os.path.join(config_name, 'test.json'), 'w') as outfile:
        json.dump(data, outfile)
    
if __name__ == '__main__':
    main()
