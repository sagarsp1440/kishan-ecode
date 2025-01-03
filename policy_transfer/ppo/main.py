# pip install hyperopt == 0.1.1
from ast import arg
from email import policy
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from hyperband import Hyperband2
import argparse
from baselines import logger
from baselines.common import set_global_seeds
import subprocess
import gym
from env_wrapper import EnvWrapper
import json
import numpy as np
import baselines.common.tf_util as U
import joblib
from gym import spaces
from run_ppo import policy_fn
import tensorflow as tf
import os
import shutil
import time
import random
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
from time import strftime, localtime

from hopper import Hopper
from humanoid_16d import Humanoid16D
from humanoid_6d import Humanoid6D
from ant import AntEnv
from walker2d import Walker2D

#random.seed(3)

UPN_PI = ""

def space(env):
    if env == "Hopper-v3":
        space = { 'l_b3': hp.uniform('l_b3', 0.1, 2.0), 'l_b4': hp.uniform('l_b4', 0.1, 1.6)}                                  # 1/4 - 4x range Hopper
    elif env == "Walker2d-v3":
        space = {'l_b1': hp.uniform('l_b1', 0.2, 0.6), 'l_b2': hp.uniform('l_b2', 0.225, 0.675), 
                'l_b3': hp.uniform('l_b3', 0.25, 0.75), 'l_b4': hp.uniform('l_b4', 0.1, 0.3)}       # Walker
    elif env == "Ant-v3":
        space = { 'fll': hp.uniform('fll', 0.1, 0.5), 'lfll': hp.uniform('lfll', 0.1, 0.5),
                'frl': hp.uniform('frl', -0.1, -0.5), 'lfrl': hp.uniform('lfrl', -0.1, -0.5),
                'bll': hp.uniform('bll', -0.1, -0.5), 'lbll': hp.uniform('lbll', -0.1, -0.5),
                'brl': hp.uniform('brl', 0.1, 0.5), 'lbrl': hp.uniform('lbrl', 0.1, 0.5)}  
    elif env == "Humanoid-v3":
        space = {'l_th': hp.uniform('l_th', -0.34*0.5, -0.34*1.5), 'l_sh': hp.uniform('l_sh', -0.3*0.5, -0.3*1.5),
                'l_f_s': hp.uniform('l_f_s', 0.075*0.5, 0.075*1.5), 'l_t_s': hp.uniform('l_t_s', 0.06*0.5, 0.06*1.5),
                'l_s_s': hp.uniform('l_f_s', 0.049*0.5, 0.049*1.5), 
                'r_th': hp.uniform('r_th', -0.34*0.5, -0.34*1.5), 'r_sh': hp.uniform('r_sh', -0.3*0.5, -0.3*1.5), 
                'r_f_s': hp.uniform('r_f_s', 0.075*0.5, 0.075*1.5), 'r_t_s': hp.uniform('l_t_s', 0.06*0.5, 0.06*1.5),
                'r_s_s': hp.uniform('l_f_s', 0.049*0.5, 0.049*1.5), # 10 parts of 2 legs
                'l_u_a': hp.uniform('l_u_a', 0.24*0.5, 0.24*1.5), 'l_l_a': hp.uniform('l_l_a', 0.17*0.5, 0.17*1.5),
                'l_h': hp.uniform('l_h', 0.06*0.5, 0.06*1.5),
                'r_u_a': hp.uniform('r_u_a', 0.24*0.5, 0.24*1.5), 'r_l_a': hp.uniform('r_l_a', 0.17*0.5, 0.17*1.5),
                'r_h': hp.uniform('r_h', 0.06*0.5, 0.06*1.5)}       # 6 parts of 2 hands   # Humanoid
    else:
        print("Exiting! Trying to create search space for unknown environment.\
              Please provide only from one of these 6 environments:\
              Hopper-v3, Walker2d-v3, Ant-v3, Humanoid-v3")
        exit() 
        
    return space
        
def handle_integers(params):
    new_params = {}
    for k, v in params.items():
        if type(v) == float and int(v) == v:
            new_params[k] = int(v)
        else:
            new_params[k] = v
    return new_params

#rng = np.random.RandomState(seed=4)
def get_params(env, rng=None):
    params = sample(space(env), rng=rng)
    # print("a", params)
    return handle_integers(params)


def run(args2):
    args2 = [str(x) for x in args2]
    #print("Run", args2)

    try:
        subprocess.check_call(args2)
        return 0
    except subprocess.CalledProcessError as e:
        print(e)
        return e.returncode


def train_params(args):
    def _train_params(env_name, n_iterations, params, xml_path, folder_path):
        global UPN_PI

        print(env_name)

        if env_name == "Hopper-v3":
            hopper = Hopper(env_name)
            new_params = hopper.construct_xml(params, xml_path)
        if env_name == "Walker2d-v3":
            walker = Walker2D(env_name)
            new_params = walker.construct_xml(params, xml_path)
        if env_name == "Ant-v3":
            ant = AntEnv(env_name)
            new_params = ant.construct_xml(params, xml_path)
        if env_name == "Humanoid-v3":
            if args.param_count == 16:
                humanoid_16d = Humanoid16D(env_name)
                new_params = humanoid_16d.construct_xml(params, xml_path)
            elif args.param_count == 6:
                humanoid_6d = Humanoid6D(env_name)
                new_params = humanoid_6d.construct_xml(params, xml_path)
        
        py_file = ["python", "policy_transfer/ppo/run_ppo.py"]
        args2 = py_file 
        args2 += ["--params_list", new_params]
        args2 += ["--n_iter", n_iterations, "--warmstart", UPN_PI, "--env_name", args.env_name]
        args2 += ["--seed", args.seed, "--name", args.name, "--min_timesteps", args.min_timesteps]
        args2 += ["--batch_size", args.batch_size]
        args2 += ["--clip", args.clip]
        args2 += ["--schedule", args.schedule]
        args2 += ["--train_up", args.train_up]
        args2 += ["--output_interval", args.output_interval]
        args2 += ["--path", folder_path]
        args2 += ["--xml_path", xml_path]
        
        run(args2)

        subprocess.run("pwd", shell=True)

        j = open(os.path.join(folder_path, './test.json'))
        data = json.load(j)
        #print('round_pi', data['pi'])

        #save pi to global variable
        UPN_PI = data['pi']
        avg_training_rew = data['avg_rew']

        ep = open(os.path.join(folder_path,'./train_ep_rets.json'))
        train_ep_rets = np.array(json.load(ep))

        plt.figure()
        plt.plot(train_ep_rets[:,0], train_ep_rets[:,1])
        plt.savefig(os.path.join(folder_path,'./graphs/graph_{}_{}.png'.format(strftime("%Y_%m_%d_%H_%M_%S", localtime()), new_params[0])))                  #location
        
        return avg_training_rew

    return _train_params

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_name', help='environment ID', default=None)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--config_count', help='Max configs in a filter', type=int, default=27)
    parser.add_argument('--name', help='name of experiments', type=str, default="")
    parser.add_argument('--min_timesteps', help='mimimum budget size', type=int)
    parser.add_argument('--batch_size', help='batch size', type=int, default=4000)
    parser.add_argument('--clip', help='clip', type=float, default=0.2)
    parser.add_argument('--schedule', help='schedule', default='constant')
    parser.add_argument('--train_up', help='whether train up', default='True')
    parser.add_argument('--output_interval', help='interval of outputting policies', type=int, default=10)
    parser.add_argument('--warmstart', help='path to warmstart policies', type=str, default="")
    parser.add_argument('--xml_path')
    parser.add_argument("--run_path", default="./")
    parser.add_argument("--order")
    parser.add_argument('--param_count', help='number of design parameters', type=int)


    args = parser.parse_args()

    

    set_global_seeds(args.seed)

    print("NAME OF EXPERIMENT:", args.env_name)
    
    # Step 1: Create a new folder
    postfix = int(time.time())
    folder_name = "run_seed{}_{}_{}".format(args.seed, args.min_timesteps, postfix)
    new_folder_path = os.path.join(args.run_path, folder_name)
    os.mkdir(new_folder_path)

    # Step 2: Copy the xml file to this folder
    shutil.copy(args.xml_path, os.path.join(os.path.abspath(new_folder_path), "env_"+args.env_name+".xml"))
    xml_path = os.path.join(os.path.abspath(new_folder_path), "env_"+args.env_name+".xml")

    log_folder = os.path.join(new_folder_path, "data")
    os.mkdir(log_folder)

    graphs_folder = os.path.join(log_folder, "graphs")
    os.mkdir(graphs_folder)

    #logger.configure(new_folder_path)

    global output_interval
    output_interval = args.output_interval
    logger.reset()
    config_name = os.path.join(new_folder_path, 'data/ppo_'+args.env_name+str(args.seed)+'_'+args.name)

    if len(args.warmstart) > 0:
        config_name += '_warmstart'

    if args.train_up == 'True':
        config_name += '_UP'

    logger.configure(config_name, ['json','stdout'])

    print("Order of algorithm (UPNHB-R or UPNHB-F)", args.order)
    print("Timesteps per unit resource", args.min_timesteps)

    hb = Hyperband2(args.env_name, get_params, train_params(args), xml_path, log_folder, args.seed, args.config_count)
    results, total_iterations, best_filter_return = hb.run(skip_last=1)

if __name__ == '__main__':
    main()
