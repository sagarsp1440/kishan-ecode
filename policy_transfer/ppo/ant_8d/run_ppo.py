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
from gym.wrappers import Monitor

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
    print("########log_dir", log_dir)
    joblib.dump(save_dict, log_dir + '/policy_params' + '.pkl', compress=True)
    if localv['iters_so_far'] % output_interval != 0:
        return
    joblib.dump(save_dict, log_dir + '/policy_params_'+str(localv['iters_so_far'])+'.pkl', compress=True)


def policy_fn(name, ob_space, ac_space):
    return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        hid_size=64, num_hid_layers=3)


def train(fll,lfll,frl,lfrl,bll,lbll,brl,lbrl, n_iterations, env_id, num_timesteps, seed, batch_size, clip, 
                schedule, mirror, warmstart, train_up, folder_path, xml_path):
    from policy_transfer.ppo.ant_8d import ppo_sgd
    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    set_global_seeds(seed)
    # import importlib
    # importlib.reload(mujoco)
    # importlib.reload(mujoco_py)

    env = gym.make(env_id, xml_file=xml_path)
    if train_up:
        if env.env.train_UP is not True:
            env.env.train_UP = True
            from gym import spaces

            # env.env.param_manager.activated_param = dyn_params
            # env.env.param_manager.controllable_param = dyn_params
            env.env.obs_dim = 111 + 8  # 111 because obs size; +8 because 8 latent parameters

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
    # env.seed(seed+MPI.COMM_WORLD.Get_rank())

    env.seed(seed)


    gym.logger.setLevel(logging.WARN)

    pol_func = policy_fn

    if len(warmstart) > 0:
        warmstart_params = joblib.load(warmstart)
        print("Loaded parameters from UPN_PI path")
    else:
        warmstart_params = None

    extn_env = EnvWrapper(env, up_dim=[fll,lfll,frl,lfrl,bll,lbll,brl,lbrl])   #wrapped env

    timesteps = num_timesteps * n_iterations
    batch_size = 4000

    pi, avg_training_reward = ppo_sgd.learn(extn_env, pol_func, folder_path,
            max_timesteps=timesteps,
            timesteps_per_batch=int(batch_size),
            clip_param=clip, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=4000,
            gamma=0.99, lam=0.95, schedule=schedule,
                        callback=callback, init_policy_params=warmstart_params)    #otpim_stepsize was = 3e-4

    print("pi", pi)
    print("mean rew", avg_training_reward)
    
    sess.close()

    return pi, avg_training_reward


def main():
    import json
    import argparse
    global log_dir


    parser2 = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser2.add_argument('--fll', help='length', type=float, default=None)
    parser2.add_argument('--lfll', help='length', type=float, default=None)
    parser2.add_argument('--frl', help='length', type=float)
    parser2.add_argument('--lfrl', help='length', type=float)
    parser2.add_argument('--bll', help='mass', type=float, default=None)
    parser2.add_argument('--lbll', help='mass', type=float, default=None)
    parser2.add_argument('--brl', help='mass', type=float)
    parser2.add_argument('--lbrl', help='mass', type=float)
    parser2.add_argument('--n_iter', help='n_iterations', type=float, default=None)
    parser2.add_argument('--warmstart', help='path to warmstart policies', type=str, default="")
    parser2.add_argument('--env', help='environment ID', default=None)
    parser2.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser2.add_argument('--name', help='name of experiments', default=None)
    parser2.add_argument('--min_timesteps', help='maximum step size', type=int)
    parser2.add_argument('--batch_size', help='batch size', type=int, default=32)
    parser2.add_argument('--clip', help='clip', type=float, default=0.2)
    parser2.add_argument('--schedule', help='schedule', default='constant')
    parser2.add_argument('--train_up', help='whether train up', default='True')
    parser2.add_argument('--output_interval', help='interval of outputting policies', type=int, default=10)
    parser2.add_argument('--mirror', help='whether to use mirror, (0: not mirror, 1: hard mirror, 2: soft mirror)', type=int, default=0)
    parser2.add_argument('--path', help='path for data')
    parser2.add_argument('--xml_path', help='path for data')




    args3 = parser2.parse_args()
    print("######### Front leg values for training #########", args3.fll, args3.lfll, args3.frl, args3.lfrl)
    print("######### Back leg values for training #########", args3.bll, args3.lbll, args3.brl, args3.lbrl)
    print("######### n_iter value for training #########", args3.n_iter)

    fll = args3.fll
    lfll = args3.lfll
    frl = args3.frl
    lfrl = args3.lfrl
    bll = args3.bll
    lbll = args3.lbll
    brl = args3.brl
    lbrl = args3.lbrl

    log_dir = args3.path
    print("received logdir", log_dir)
    


    n_iterations = args3.n_iter

    pi, avg_rew = train(fll,lfll,frl,lfrl,bll,lbll,brl,lbrl, n_iterations, args3.env, num_timesteps=int(args3.min_timesteps), 
                        seed=args3.seed, batch_size=args3.batch_size, clip=args3.clip, 
                        schedule=args3.schedule, mirror=args3.mirror, warmstart=args3.warmstart, 
                        train_up=args3.train_up, folder_path = args3.path, xml_path = args3.xml_path)
                        

    print("PI", pi)
    print("log_dir", log_dir)
    config_name = args3.path
    data = {'pi': log_dir + '/policy_params' + '.pkl', 'avg_rew': avg_rew}
    with open(os.path.join(config_name, 'test.json'), 'w') as outfile:
        json.dump(data, outfile)


    
if __name__ == '__main__':
    main()
