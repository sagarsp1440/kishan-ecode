# pip install hyperopt == 0.1.1
from ast import arg
from email import policy
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from hyperband import Hyperband
import argparse
from baselines import logger
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
import subprocess
# subprocess.run("source /home/knagiredla/.mujoco/mujoco210/myprojects/policy_transfer/setup.sh", shell=True)
import random
from matplotlib import pyplot as plt
import time


# handle floats which should be integers
# works with flat params

#random.seed(3)

UPN_PI = ""
space = {'l1': hp.uniform('l1', 0.1, 2.0), 'l2': hp.uniform('l2', 0.1, 2.0), 'ml1': hp.uniform('ml1', 0.1, 2.0), 'ml2': hp.uniform('ml2', 0.1, 2.0)
         }

def handle_integers(params):
    new_params = {}
    for k, v in params.items():
        if type(v) == float and int(v) == v:
            new_params[k] = int(v)
        else:
            new_params[k] = v
    return new_params


#rng = np.random.RandomState(seed=4)

def get_params(rng=None):
    params = sample(space, rng=rng)
    # print("a", params)
    return handle_integers(params)


def run(args2):
    args2 = [str(x) for x in args2]
    print("Run", args2)

    try:
        subprocess.check_call(args2)
        return 0
    except subprocess.CalledProcessError as e:
        print(e)
        return e.returncode

# def init_policy(pi, init_policy_params):

#     U.initialize()

#     cur_scope = pi.get_variables()[0].name[0:pi.get_variables()[0].name.find('/')]
#     orig_scope = list(init_policy_params.keys())[0][0:list(init_policy_params.keys())[0].find('/')]
#     print(cur_scope, orig_scope)
#     for i in range(len(pi.get_variables())):
#         if pi.get_variables()[i].name.replace(cur_scope, orig_scope, 1) in init_policy_params:
#             assign_op = pi.get_variables()[i].assign(init_policy_params[pi.get_variables()[i].name.replace(cur_scope, orig_scope, 1)])
#             tf.get_default_session().run(assign_op)
    
#     return pi
def train_params(args):
    def _train_params(n_iterations, params, folder_path):

        #from run_ppo import train
        global UPN_PI 

        l1 = round(params.get('l1'), 4)
        l2 = round(params.get('l2'), 4)
        ml1 = round(params.get('ml1'), 4)
        ml2 = round(params.get('ml2'), 4)
        
        # t = t.get('length')
        # print("t", t)
        print("n_iter", n_iterations)


        #Calling a script run_ppo.py
        args2 = ["python", "policy_transfer/ppo/acrobot/run_ppo.py", "--l1", l1, "--l2", l2, "--ml1", ml1, "--ml2", ml2, "--n_iter",n_iterations, "--warmstart", UPN_PI, "--env", args.env, "--seed", args.seed, "--name", args.name, "--min_timesteps", args.min_timesteps]
        args2 += ["--batch_size", args.batch_size]
        args2 += ["--clip", args.clip]
        args2 += ["--schedule", args.schedule]
        args2 += ["--train_up", args.train_up]
        args2 += ["--output_interval", args.output_interval]
        args2 += ["--mirror", args.mirror]
        args2 += ["--path", folder_path]

        run(args2)

        subprocess.run("pwd", shell=True)
        j = open(os.path.join(folder_path, 'test.json'))
        data = json.load(j)
        print('round_pi', data['pi'])


        #save pi to global variable

        UPN_PI = data['pi']
        avg_training_rew = data['avg_rew']


        ep = open(os.path.join(folder_path,'./train_ep_rets.json'))
        train_ep_rets = np.array(json.load(ep))

        plt.figure()
        plt.plot(train_ep_rets[:,0], train_ep_rets[:,1])
        plt.savefig(os.path.join(folder_path,'./graphs/graph_{}_{}.png'.format(l1, l1-np.random.random())))                  #location

        return avg_training_rew

    return _train_params

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default=None)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--name', help='name of experiments', type=str, default="")
    parser.add_argument('--min_timesteps', help='maximum step size', type=int, default = 1000)
    parser.add_argument('--batch_size', help='batch size', type=int, default=4000)
    parser.add_argument('--clip', help='clip', type=float, default=0.2)
    parser.add_argument('--schedule', help='schedule', default='constant')
    parser.add_argument('--train_up', help='whether train up', default='True')
    parser.add_argument('--output_interval', help='interval of outputting policies', type=int, default=10)
    parser.add_argument('--mirror', help='whether to use mirror, (0: not mirror, 1: hard mirror, 2: soft mirror)', type=int, default=0)
    parser.add_argument('--warmstart', help='path to warmstart policies', type=str, default="")
    parser.add_argument("--run_path", default="./")
    parser.add_argument("--order")


    args = parser.parse_args()

# Step 1: Create a new folder
    postfix = int(time.time())
    folder_name = "run_seed{}_{}".format(args.seed, postfix)
    new_folder_path = os.path.join(args.run_path, folder_name)
    os.mkdir(new_folder_path)

    data_folder = os.path.join(new_folder_path, "data")
    os.mkdir(data_folder)

    graphs_folder = os.path.join(data_folder, "graphs")
    os.mkdir(graphs_folder)

    global output_interval
    output_interval = args.output_interval
    logger.reset()
    config_name = 'data/ppo_'+args.env+str(args.seed)+'_'+args.name

    if len(args.warmstart) > 0:
        config_name += '_warmstart'

    if args.train_up == 'True':
        config_name += '_UP'

    logger.configure(config_name, ['json','stdout'])

    print("ORDER", args.order)
    print("MIN_TIMESTEPS", args.min_timesteps)
    
    hb = Hyperband(get_params, train_params(args), data_folder, args.seed)
    results, total_iterations, best_filter_return = hb.run(skip_last=1)

    #writing final output to file
    # f = open("out_summary.txt", 'w+')
    # f.write("Total Resources Used:" + str(total_iterations)+ "\n")
    # f.write("Results ="+ str(results) + "\n")
    # f.write("-->" + str(len(results)) + " total runs, best:\n")
    # for r in sorted(results, key=lambda x: x['Return'], reverse=True)[:20]:
    #     f.write("Return:" + str(r['Return']) +" " + str(r['params']) + " " + str(r['iterations'])+ "Iterations " + "ID:" + str(r['id']) + "\n")
    #for i in best_filter_return:
    #    f.write("Best Filter return " + str(i)+ "\n")

if __name__ == '__main__':
    main()
