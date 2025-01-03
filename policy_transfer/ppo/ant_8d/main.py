# pip install hyperopt == 0.1.1
from ast import arg
from email import policy
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from baselines.common import set_global_seeds
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
import shutil
import time
import subprocess
import random
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET


UPN_PI = ""
'''
Ant original parameters (x-cordinates): 
fll = 0.2, lfll = 0.4, 
frl = -0.2, lfrl = -0.4,
bll = -0.2, lbll = -0.4, 
brl = 0.2, lbrl = 0.4 
'''
#initial 8d config  
space = { 'fll': hp.uniform('fll', 0.1, 0.5), 'lfll': hp.uniform('lfll', 0.1, 0.5),
        'frl': hp.uniform('frl', -0.1, -0.5), 'lfrl': hp.uniform('lfrl', -0.1, -0.5),
        'bll': hp.uniform('bll', -0.1, -0.5), 'lbll': hp.uniform('lbll', -0.1, -0.5),
        'brl': hp.uniform('brl', 0.1, 0.5), 'lbrl': hp.uniform('lbrl', 0.1, 0.5),
         }                             

#for broken leg
# space = { 
#         'lfrl': hp.uniform('lfrl', -0.1, -0.5),
#         'lbll': hp.uniform('lbll', -0.1, -0.5),
#         'lbrl': hp.uniform('lbrl', 0.1, 0.5),
#          }                             


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


def train_params(args):
    def _train_params(n_iterations, params, xml_path, folder_path):
        '''
        Ant original parameters (x-cordinates): 
        fll = 0.2, lfll = 0.4, 
        frl = -0.2, lfrl = -0.4,
        bll = -0.2, lbll = -0.4, 
        brl = 0.2, lbrl = 0.4 
        '''

        #from run_ppo import train
        global UPN_PI

        #original 8d config

        fll = round(params.get('fll'), 3)
        lfll = 0.1
        frl = round(params.get('frl'), 3)
        lfrl = round(params.get('lfrl'), 3)
        bll = round(params.get('bll'), 3)
        lbll = round(params.get('lbll'), 3)
        brl = round(params.get('brl'), 3)
        lbrl = round(params.get('lbrl'), 3)

        #broken leg config
        # fll = 0.2
        # lfll = 0.1
        # frl = 0.2
        # bll = 0.2
        # brl = 0.2
        # lfrl = 0.15
        # lbll = 0.45
        # lbrl = 0.47
            
        ########### Xml file modification part - start ###############

        tree = ET.parse(xml_path)
        root = tree.getroot()

        worldbody = root[5]
        torso = worldbody[2]

        bodypart={}
        body_list = []

        leg_pos_list = {'front_left_leg':'0.0 0.0 0.0'+ " " + str(fll)+ " " + str(fll)+ " " + '0.0',
                        'front_right_leg':'0.0 0.0 0.0'+ " " + str(frl)+ " " + str(-frl)+ " " + '0.0',
                        'back_leg': '0.0 0.0 0.0'+ " " + str(bll)+ " " + str(bll)+ " " + '0.0',
                        'right_back_leg':'0.0 0.0 0.0'+ " " + str(brl)+ " " + str(-brl)+ " " + '0.0'}

        lower_leg_pos_list = {'front_left_leg':'0.0 0.0 0.0'+ " " + str(lfll)+ " " + str(lfll)+ " " + '0.0',
                        'front_right_leg':'0.0 0.0 0.0'+ " " + str(lfrl)+ " " + str(-lfrl)+ " " + '0.0',
                        'back_leg': '0.0 0.0 0.0'+ " " + str(lbll)+ " " + str(lbll)+ " " + '0.0',
                        'right_back_leg':'0.0 0.0 0.0'+ " " + str(lbrl)+ " " + str(-lbrl)+ " " + '0.0'}


        for child_body in torso.findall('body'):
            #bodypart[body.attrib['name']] = body
            child1_geom_pos = child_body[0].attrib
            child1_geom_pos['fromto'] = leg_pos_list[child_body.attrib['name']]
            print("new",child1_geom_pos['fromto'])
            # child2_body_pos = child_body[1].attrib

            nxt_body = child_body[1]
            leg_aux_pos = child_body[1].attrib
            pos = leg_pos_list[child_body.attrib['name']].split(" ")
            leg_aux_pos['pos'] = str(pos[3]+" "+pos[4]+" "+pos[5])
            # print(leg_aux_pos)

            lower_leg_body_geom = child_body[1][2][1].attrib
            # print(lower_leg_body_geom["fromto"])
            lower_leg_body_geom["fromto"]=lower_leg_pos_list[child_body.attrib['name']]
            # print(lower_leg_body_geom)

        tree.write(xml_path)
        # print(xml_path)
        ########### Xml file modification part - end ###############

        print("n_iter", n_iterations)

        #Calling a script run_ppo.py
        args2 = ["python", "policy_transfer/ppo/ant_8d/run_ppo.py", "--fll", fll, "--lfll", lfll, "--frl", frl, "--lfrl", lfrl, "--bll", bll, "--lbll", lbll, "--brl", brl, "--lbrl", lbrl, "--n_iter", n_iterations, "--warmstart", '', "--env", args.env, "--seed", args.seed, "--name", args.name, "--min_timesteps", args.min_timesteps]
        args2 += ["--batch_size", args.batch_size]  
        args2 += ["--clip", args.clip]
        args2 += ["--schedule", args.schedule]
        args2 += ["--train_up", args.train_up]
        args2 += ["--output_interval", args.output_interval]
        args2 += ["--mirror", args.mirror]
        args2 += ["--path", folder_path]
        args2 += ["--xml_path", xml_path]

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
        plt.savefig(os.path.join(folder_path,'./graphs/graph_{}_{}_{}_{}_{}.png'.format(fll, lfll, frl, lfrl, lfll-np.random.random())))                  #location

        return avg_training_rew

    return _train_params

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default=None)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--config_count', help='Max configs in a filter', type=int, default=27)
    parser.add_argument('--eta_val', help='eta', type=int, default=3)
    parser.add_argument('--name', help='name of experiments', type=str, default="")
    parser.add_argument('--min_timesteps', help='mimimum budget size', type=int)
    parser.add_argument('--batch_size', help='batch size', type=int, default=4000)
    parser.add_argument('--clip', help='clip', type=float, default=0.2)
    parser.add_argument('--schedule', help='schedule', default='constant')
    parser.add_argument('--train_up', help='whether train up', default='True')
    parser.add_argument('--output_interval', help='interval of outputting policies', type=int, default=10)
    parser.add_argument('--mirror', help='whether to use mirror, (0: not mirror, 1: hard mirror, 2: soft mirror)', type=int, default=0)
    parser.add_argument('--warmstart', help='path to warmstart policies', type=str, default="")
    parser.add_argument('--xml_path')
    parser.add_argument("--run_path", default="./")
    parser.add_argument("--order")



    args = parser.parse_args()

    set_global_seeds(args.seed)

    print("NAME OF EXPERIMENT", args.name)
    print("Ctrl_cost is made zero - apr10")
    
    # Step 1: Create a new folder
    postfix = int(time.time())
    folder_name = "run_seed{}_{}_{}".format(args.seed, args.min_timesteps, postfix)
    new_folder_path = os.path.join(args.run_path, folder_name)
    os.mkdir(new_folder_path)

    # Step 2: Copy the xml file to this folder
    shutil.copy(args.xml_path, os.path.join(os.path.abspath(new_folder_path), "env.xml"))
    xml_path = os.path.join(os.path.abspath(new_folder_path), "env.xml")

    data_folder = os.path.join(new_folder_path, "data")
    os.mkdir(data_folder)

    graphs_folder = os.path.join(data_folder, "graphs")
    os.mkdir(graphs_folder)

    #logger.configure(new_folder_path)

    global output_interval
    output_interval = args.output_interval
    logger.reset()
    config_name = os.path.join(new_folder_path, 'data/ppo_'+args.env+str(args.seed)+'_'+args.name)



    if len(args.warmstart) > 0:
        config_name += '_warmstart'

    if args.train_up == 'True':
        config_name += '_UP'

    logger.configure(config_name, ['json','stdout'])

    print("ORDER", args.order)
    print("MINIMUM TIMESTEPS", args.min_timesteps)

    hb = Hyperband(get_params, train_params(args), xml_path, data_folder, args.seed, args.config_count, args.eta_val)
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
