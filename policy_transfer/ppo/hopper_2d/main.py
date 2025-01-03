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
Hopper original parameters::
l_b1 = 0.4, l_b2 = 0.45, l_b3 = 0.5, l_b4 = 0.39
t1 = 0.05, t1 = 0.05, t1 = 0.04, t4 = 0.06 
'''
# space = { 
#         'l_b3': hp.uniform('l_b3', 0.35, 0.65), 'l_b4': hp.uniform('l_b4', 0.31, 0.46)
#          }                                  # 30% range

space = { 
        'l_b3': hp.uniform('l_b3', 0.1, 2.0), 'l_b4': hp.uniform('l_b4', 0.1, 1.6)
         }                                  # 1/4 - 4x range

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
    def _train_params(n_iterations, params, xml_path, folder_path, val1, val2):

        #from run_ppo import train
        global UPN_PI

        # l_b1 = round(params.get('l_b1'), 4)
        # l_b2 = round(params.get('l_b2'), 4)
        l_b3 = round(params.get('l_b3'), 4)
        l_b4 = round(params.get('l_b4'), 4)
        # l_b3 = 0.3541
        # l_b4 = 0.3878
        # t1 = round(params.get('t1'), 4)
        # t2 = round(params.get('t2'), 4)
        # t3 = round(params.get('t3'), 4)
        # t4 = round(params.get('t4'), 4)
        t3 = 0.04
        t4 = 0.06
        l_b3 = val1	
        l_b4 =val2
        
        ########### Xml file modification part - start ###############

        tree = ET.parse(xml_path)
        root = tree.getroot()

        def find_val(root, parent_path, child_id):
            parent = root.find(parent_path)
            body= parent.attrib[child_id]
            body_list= body.split()
            y = body_list[2]                      #height or length along vertical axis
            z = body_list[1]                      #width
            x = body_list[0]                      #length along horizonatal
            return y,x,z

        parent_main = "./worldbody/"
        child_id="fromto"
        y_vals = []
        x_vals = []
        z_vals = []

        body_count = 4                                      # Hopper Agent total parts = 4

        for i in range(1,body_count+1):
            current_parent=parent_main+ ("body/" * i) +"/geom[@"
            parent_path = current_parent+child_id+"]"
            y,x,_ = find_val(root, parent_path, child_id)
            y_vals.append(y)
            x_vals.append(x)

        # yl3 = 1.05      
        # yl2 = 1.55     
        yl1 = l_b3      
        xl4 = l_b4

        print(y_vals)
        y0 = float(y_vals[3])
        y1 = yl1 + y0           # y1-y0 gives length from base to y1 -> l_b3 (length of body3)
        # y2 = y1 + (float(y_vals[1]) - float(y_vals[2]))
        # y3 = y2 + (float(y_vals[0]) - float(y_vals[1]))
        y2 = y1 + 0.45
        y3 = y2 + 0.40
                                    # y3 is the highest point in vertical axis   

                                    # 0.1, 10.1, 11.15, 12.60

        x0 = float(x_vals[3])
        print("x", x)
        #setting horizontal part length values
        if xl4!=None:
            xl = xl4 + x0
            x = root.find("./worldbody/body/body/body/body/geom[@fromto]")
            x_elem = x.attrib["fromto"]
            x1_list = x_elem.split()
            print("x1lst", x1_list)
            x1_list[3] = str(xl)
            x_temp = ' '.join(x1_list)
            x.attrib["fromto"] = x_temp

        new_y = [y0, y1, y2, y3]
        new_y.reverse()
        print("new_y", new_y)

        #setting vertical-part length values in xml
        def set_val(root, parent_path, child_id, start_y, end_y=None):
            parent = root.find(parent_path)
            body= parent.attrib[child_id]
            body_list= body.split()
            #print("setting value", start_y)
            if child_id=="fromto":
                body_list[2] = str(start_y)
                if end_y!=None:
                    body_list[5] = str(end_y)

            temp = ' '.join(body_list)
            parent.attrib[child_id] = temp
            
        parent_main = "./worldbody/"
        child_id="fromto"
        y_vals = []

        for i in range(0,len(new_y)):
            j = i + 1
            start = new_y[i] 
            current_parent=parent_main+ ("body/" * j) +"/geom[@"
            parent_path = current_parent+child_id+"]"
            if j < 3:
                end = new_y[i+1]
                set_val(root, parent_path, child_id, start, end)
            else:
                set_val(root, parent_path, child_id, start)     #end is the same as start in this case (i.e. fixed point in groud)


        #Thickness vales from HB t3, t4
        
        # t_b3 = root.find("./worldbody/body/body/body/geom[@size]")
        # t_b3.attrib["size"] = str(t3)
        # t_b4 = root.find("./worldbody/body/body/body/body/geom[@size]")
        # t_b4.attrib["size"] = str(t4)

        tree.write(xml_path)

        ########### Xml file modification part - end ###############

        print("n_iter", n_iterations)

        #Calling a script run_ppo.py
        args2 = ["python", "policy_transfer/ppo/hopper_2d/run_ppo.py", "--l_b4", l_b4, "--l_b3", l_b3, "--t3", t3, "--t4", t4,  "--n_iter", n_iterations, "--warmstart", '', "--env", args.env, "--seed", args.seed, "--name", args.name, "--min_timesteps", args.min_timesteps]
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
        plt.savefig(os.path.join(folder_path,'./graphs/graph_{}_{}_{}.png'.format(l_b4, l_b3, l_b4-np.random.random())))                  #location

        return avg_training_rew

    return _train_params

# def eval_params(args):
#     def _eval_params(all_params, n_iterations):
#         config_name = "./data"
#         with open(os.path.join(config_name, 'params.json'), 'w') as outfile:
#             json.dump(all_params, outfile)
#         # T [{'length': 1.0219238067563368}, {'length': 2.7154676388113264}, {'length': 2.3166319449990627}, {'length': 1.9017891657490518}, {'length': 1.8830407815605827}, {'length': 1.0602292431529876}, {'length': 2.1644915234921376}, {'length': 2.777048912329862}, {'length': 2.9923178802312376}] <class 'list'>
#         _args = ["python", "policy_transfer/ppo/acrobot/evaluate.py", "--n_iter", n_iterations]        
#         run(_args)

#         j = open('./data/return.json')
#         avg_rlreturns = json.load(j)
        
#         return avg_rlreturns

#     return _eval_params


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default=None)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--name', help='name of experiments', type=str, default="")
    parser.add_argument('--min_timesteps', help='mimimum budget size', type=int)
    parser.add_argument('--batch_size', help='batch size', type=int, default=4000)
    parser.add_argument('--clip', help='clip', type=float, default=0.2)
    parser.add_argument('--schedule', help='schedule', default='linear')
    parser.add_argument('--train_up', help='whether train up', default='True')
    parser.add_argument('--output_interval', help='interval of outputting policies', type=int, default=10)
    parser.add_argument('--mirror', help='whether to use mirror, (0: not mirror, 1: hard mirror, 2: soft mirror)', type=int, default=0)
    parser.add_argument('--warmstart', help='path to warmstart policies', type=str, default="")
    parser.add_argument('--xml_path')
    parser.add_argument("--run_path", default="./")
    parser.add_argument('--order')
    parser.add_argument('--val1')    
    parser.add_argument('--val2')

    args = parser.parse_args()

    val1 = args.val1
    val2 = args.val2
    # random.seed(args.seed)
    set_global_seeds(args.seed)
    
    # Step 1: Create a new folder
    postfix = int(time.time())
    folder_name = "run_seed{}_{}".format(args.seed, postfix)
    new_folder_path = os.path.join(args.run_path, folder_name)
    os.mkdir(new_folder_path)

    # Step 2: Copy the xml file to this folder
    shutil.copy(args.xml_path, os.path.join(os.path.abspath(new_folder_path), "env.xml"))
    xml_path = os.path.join(os.path.abspath(new_folder_path), "env.xml")

    data_folder = os.path.join(new_folder_path, "data")
    os.mkdir(data_folder)

    graphs_folder = os.path.join(data_folder, "graphs")
    os.mkdir(graphs_folder)

    global output_interval
    output_interval = args.output_interval
    logger.reset()
    config_name = os.path.join(new_folder_path, 'data/ppo_'+args.env+str(args.seed)+'_'+args.name)

    if len(args.warmstart) > 0:
        config_name += '_warmstart'

    if args.train_up == 'True':
        config_name += '_UP'

    print("ORDER", args.order)
    print("MINTIMESTEPS", args.min_timesteps)

    logger.configure(config_name, ['json','stdout'])

    hb = Hyperband(get_params, train_params(args), xml_path, data_folder, args.seed, float(val1), float(val2))
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