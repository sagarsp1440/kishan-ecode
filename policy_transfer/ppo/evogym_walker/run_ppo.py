#!/usr/bin/env python
from statistics import mean
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os
import os.path as osp
import gym
# env_dict = gym.envs.registration.registry.env_specs.copy()
# for env in env_dict:
#     if 'SimpleWalkingEnv-v0' in env:
#         print("Remove {} from registry".format(env))
#         del gym.envs.registration.registry.env_specs[env]
from policy_transfer.ppo.evogym_walker.envs import simple_env

import logging
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
from evogym import sample_robot


output_interval = 10

def callback(localv, globalv):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(globalv.keys())
    save_dict = {}
    variables = localv['pi'].get_variables()
    for i in range(len(variables)):
        cur_val = variables[i].eval()
        save_dict[variables[i].name] = cur_val
    joblib.dump(save_dict, logger.get_dir() + '/policy_params' + '.pkl', compress=True)
    if localv['iters_so_far'] % output_interval != 0:
        return
    joblib.dump(save_dict, logger.get_dir()+'/policy_params_'+str(localv['iters_so_far'])+'.pkl', compress=True)


def policy_fn(name, ob_space, ac_space):
    print("#########THis is obs space sent to policy_fn", ob_space)
    return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        hid_size=64, num_hid_layers=3)

def train(robot_values, n_iterations, env_id, num_timesteps, seed, batch_size, clip, 
                schedule, mirror, warmstart, train_up):
    from policy_transfer.ppo.evogym_walker import ppo_sgd
    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    set_global_seeds(seed)

    #robot_values = list(params.values())
    values = list(robot_values.values())
    print("################## Values ################ \n", robot_values)


    body, connections, flag = sample_robot((3,3), values)
    print("flag is", flag)

    if flag:
        print("Entered line 56")
        env = gym.make(env_id, body=body)
        if train_up:
            print("Entered line59, train_up")
            if env.env.train_UP is not True:
                env.env.train_UP = True
                from gym import spaces
                print("Importing spaces")

                # env.env.param_manager.activated_param = dyn_params
                # env.env.param_manager.controllable_param = dyn_params
                obs_size = env.env.reset().size
                env.env.obs_dim = obs_size + 9      # 200 because self.state size is 4 +1 because only length
                print("set env dim to", env.env.obs_dim)
                obs_size_new = obs_size + 9

                # high = np.inf * np.ones(env.env.obs_dim)
                # low = -high
                # env.env.observation_space = spaces.Box(low, high)
                # env.observation_space = spaces.Box(low, high)

                num_actuators = env.env.get_actuator_indices('robot').size

                env.env.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(obs_size_new,), dtype=np.float)
                env.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(obs_size_new,), dtype=np.float)
                print("Created observation space as required for UPN")

        extn_env = EnvWrapper(env, up_dim=values)  # wrapped env

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

        timesteps = 10000 * n_iterations

        pi, rew = ppo_sgd.learn(extn_env, pol_func,
                max_timesteps=timesteps,
                timesteps_per_batch=int(batch_size),
                clip_param=clip, entcoeff=0.0,
                optim_epochs=5, optim_stepsize=3e-4, optim_batchsize=64,
                gamma=0.99, lam=0.95, schedule=schedule,
                            callback=callback,
                    init_policy_params=warmstart_params,
        )

        print("pi", pi)
        print("rew", rew)
        
        sess.close()

        return pi

    
   
def main():
    import json
    import argparse

    parser2 = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser2.add_argument('--n_iter', help='n_iterations', type=float, default=None)
    parser2.add_argument('--warmstart', help='path to warmstart policies', type=str, default="")
    parser2.add_argument('--env', help='environment ID', default=None)
    parser2.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser2.add_argument('--name', help='name of experiments', type=str, default="")
    parser2.add_argument('--max_step', help='maximum step size', type=int, default = 1000)
    parser2.add_argument('--batch_size', help='batch size', type=int, default=512)
    parser2.add_argument('--clip', help='clip', type=float, default=0.2)
    parser2.add_argument('--schedule', help='schedule', default='constant')
    parser2.add_argument('--train_up', help='whether train up', default='')
    parser2.add_argument('--output_interval', help='interval of outputting policies', type=int, default=10)
    parser2.add_argument('--mirror', help='whether to use mirror, (0: not mirror, 1: hard mirror, 2: soft mirror)', type=int, default=0)



    j = open('./data/robot.json')
    robot_values = json.load(j)
    print('########### robot ############', robot_values)

    args3 = parser2.parse_args()
    print("args3", args3)

    n_iterations = args3.n_iter


    pi = train(robot_values, n_iterations, args3.env, num_timesteps=int(args3.max_step), 
                        seed=args3.seed, batch_size=args3.batch_size, clip=args3.clip, 
                        schedule=args3.schedule, mirror=args3.mirror, warmstart=args3.warmstart, 
                        train_up=args3.train_up)

    print("PI", pi)
            
    config_name = "./data"
    data = {'pi': logger.get_dir() + '/policy_params' + '.pkl'}
    with open(os.path.join(config_name, 'test.json'), 'w') as outfile:
            json.dump(data, outfile)


if __name__ == '__main__':
    main()