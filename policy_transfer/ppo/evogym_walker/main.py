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

# handle floats which should be integers
# works with flat params

UPN_PI = ""
x = [0,1,2,3,4]
space = {'0': hp.choice('0', x), '1': hp.choice('1', x), '2': hp.choice('2', x), '3': hp.choice('3', x), '4': hp.choice('4', x), '5': hp.choice('5', x),
            '6': hp.choice('6', x), '7': hp.choice('7', x), '8': hp.choice('8', x)}
        # '9': hp.choice('9', x),
        # '10': hp.choice('10', x),
        # '11': hp.choice('11', x),
        # '12': hp.choice('12', x),
        # '13': hp.choice('13', x),
        # '14': hp.choice('14', x),
        # '15': hp.choice('15', x)
        #  }

def handle_integers(params):
    new_params = {}
    for k, v in params.items():
        if type(v) == float and int(v) == v:
            new_params[k] = int(v)
        else:
            new_params[k] = v
    return new_params


def get_params():  
    params = sample(space)
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
    def _train_params(n_iterations, t):
        #from run_ppo import train
        global UPN_PI 
        #t = t.get('length')

        robot_values = t
        print("############# t values before run_ppo.py ############", t)
        config_name = "./data"
        with open(os.path.join(config_name, 'robot.json'), 'w') as outfile:
                json.dump(t, outfile)

        #Calling a script run_ppo.py
        args2 = ["python", "policy_transfer/ppo/evogym_walker/run_ppo.py", "--n_iter", n_iterations, "--warmstart", UPN_PI, "--env", args.env, "--seed", args.seed, "--name", args.name, "--max_step", args.max_step]
        args2 += ["--batch_size", args.batch_size]
        args2 += ["--clip", args.clip]
        args2 += ["--schedule", args.schedule]
        args2 += ["--train_up", args.train_up]
        args2 += ["--output_interval", args.output_interval]
        args2 += ["--mirror", args.mirror]
        run(args2)

        j = open('./data/test.json')
        data = json.load(j)
        print('round_pi', data['pi'])

        #save pi to global variable

        UPN_PI = data['pi']
    
    return _train_params

def eval_params(args):
    def _eval_params(all_params, n_iterations):
        config_name = "./data"
        with open(os.path.join(config_name, 'params.json'), 'w') as outfile:
            json.dump(all_params, outfile)
        # T [{'length': 1.0219238067563368}, {'length': 2.7154676388113264}, {'length': 2.3166319449990627}, {'length': 1.9017891657490518}, {'length': 1.8830407815605827}, {'length': 1.0602292431529876}, {'length': 2.1644915234921376}, {'length': 2.777048912329862}, {'length': 2.9923178802312376}] <class 'list'>
        _args = ["python", "policy_transfer/ppo/evogym_walker/evaluate.py", "--n_iter", n_iterations]        
        run(_args)

        j = open('./data/return.json')
        avg_rlreturns = json.load(j)
        
        return avg_rlreturns

    return _eval_params


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default=None)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--name', help='name of experiments', type=str, default="")
    parser.add_argument('--max_step', help='maximum step size', type=int, default = 10)
    parser.add_argument('--batch_size', help='batch size', type=int, default=4000)
    parser.add_argument('--clip', help='clip', type=float, default=0.2)
    parser.add_argument('--schedule', help='schedule', default='constant')
    parser.add_argument('--train_up', help='whether train up', default='True')
    parser.add_argument('--output_interval', help='interval of outputting policies', type=int, default=10)
    parser.add_argument('--mirror', help='whether to use mirror, (0: not mirror, 1: hard mirror, 2: soft mirror)', type=int, default=0)
    parser.add_argument('--warmstart', help='path to warmstart policies', type=str, default="")

    args = parser.parse_args()
    global output_interval
    output_interval = args.output_interval
    logger.reset()
    config_name = 'data/ppo_'+args.env+str(args.seed)+'_'+args.name

    if len(args.warmstart) > 0:
        config_name += '_warmstart'

    if args.train_up == 'True':
        config_name += '_UP'

    logger.configure(config_name, ['json','stdout'])

    hb = Hyperband(get_params, train_params(args), eval_params(args))
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
