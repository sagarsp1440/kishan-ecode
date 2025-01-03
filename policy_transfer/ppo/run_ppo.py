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
from gym import spaces


output_interval = 10


def callback(localv, globalv):
    global log_dir
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(globalv.keys())
    save_dict = {}
    variables = localv['pi'].get_variables()
    for i in range(len(variables)):
        cur_val = variables[i].eval()
        save_dict[variables[i].name] = cur_val
    print("#log_dir", log_dir)
    joblib.dump(save_dict, log_dir + '/policy_params' + '.pkl', compress=True)
    if localv['iters_so_far'] % output_interval != 0:
        return
    joblib.dump(save_dict, log_dir + '/policy_params_'+str(localv['iters_so_far'])+'.pkl', compress=True)


def policy_fn(name, ob_space, ac_space):
    return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        hid_size=64, num_hid_layers=3)


def train(params, n_iterations, env_name, num_timesteps, seed, batch_size, clip, schedule, warmstart, train_up, folder_path, xml_path):
    from policy_transfer.ppo import ppo_sgd
    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    set_global_seeds(seed)

    env = gym.make(env_name, xml_file=xml_path)
    if train_up:
        if env.env.train_UP is not True:
            env.env.train_UP = True
            
            if env_name == "Hopper-v3":
                env.env.obs_dim = 11 + len(params)
            elif env_name == "Walker2d-v3":
                env.env.obs_dim = 17 + len(params)
            elif env_name == "Ant-v3":
                env.env.obs_dim = 111 + len(params)
            elif env_name == "Humanoid-v3":
                if len(params) == 6: 
                    env.env.obs_dim = 376 + len(params)  # 376 because obs size is 376; +16 because 16 latent parameters
                elif len(params) == 16: env.env.obs_dim = 436

            high = np.inf * np.ones(env.env.obs_dim)
            low = -high
            env.env.observation_space = spaces.Box(low, high)
            env.observation_space = spaces.Box(low, high)
    
    pol_func = policy_fn

    if len(warmstart) > 0:
        warmstart_params = joblib.load(warmstart)
        #print("Loaded parameters from UPN_PI path")
    else:
        warmstart_params = None

    extn_env = EnvWrapper(env, up_dim=params)   #wrapped env

    total_timesteps_per_round = num_timesteps * n_iterations

    pi, avg_training_reward = ppo_sgd.learn(extn_env, pol_func, folder_path, max_timesteps=total_timesteps_per_round, timesteps_per_batch=int(batch_size),
            clip_param=clip, entcoeff=0.0, optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=4000, gamma=0.99, lam=0.95, 
            schedule=schedule, callback=callback, init_policy_params=warmstart_params)

    print("pi", pi)
    print("mean_rew", avg_training_reward)
    
    sess.close()

    return pi, avg_training_reward


def list_of_ints(arg):
    return arg[1:-1].split(',')

def main():
    import json
    import argparse
    global log_dir

    parser2 = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser2.add_argument('--params_list', help='list of params', type=list_of_ints, default=None)
    parser2.add_argument('--n_iter', help='n_iterations', type=float, default=None)
    parser2.add_argument('--warmstart', help='path to warmstart policies', type=str, default="")
    parser2.add_argument('--env_name', help='environment ID', default=None)
    parser2.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser2.add_argument('--name', help='name of experiments', type=str, default="")
    parser2.add_argument('--min_timesteps', help='maximum step size', type=int)
    parser2.add_argument('--batch_size', help='batch size', type=int, default=4000)
    parser2.add_argument('--clip', help='clip', type=float, default=0.2)
    parser2.add_argument('--schedule', help='schedule', default='constant')
    parser2.add_argument('--train_up', help='whether train up', default='True')
    parser2.add_argument('--output_interval', help='interval of outputting policies', type=int, default=10)
    parser2.add_argument('--path', help='path for data')
    parser2.add_argument('--xml_path', help='path for xml')

    args3 = parser2.parse_args()

    params = args3.params_list
    params = [float(x) for x in params]

    n_iterations = args3.n_iter
    log_dir = args3.path

    pi, avg_rew = train(params, n_iterations, args3.env_name, num_timesteps=int(args3.min_timesteps), 
                        seed=args3.seed, batch_size=args3.batch_size, clip=args3.clip, 
                        schedule=args3.schedule, warmstart=args3.warmstart, 
                        train_up=args3.train_up, folder_path = args3.path, xml_path=args3.xml_path)
    
    # print("PI", pi)
    # print("log_dir", log_dir)
    config_name = args3.path
    data = {'pi': log_dir + '/policy_params' + '.pkl', 'avg_rew': avg_rew}
    with open(os.path.join(config_name, 'test.json'), 'w') as outfile:
        json.dump(data, outfile)
    
if __name__ == '__main__':
    main()
