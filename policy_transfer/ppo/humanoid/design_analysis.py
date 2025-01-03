# pip install hyperopt == 0.1.1
from ast import arg
from email import policy
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from sampler import Sampler
import argparse
# from baselines import logger
# from baselines.common import set_global_seeds
import subprocess
import gym
# from env_wrapper import EnvWrapper
import json
import numpy as np
# import baselines.common.tf_util as U
import joblib
from gym import spaces
# from run_ppo import policy_fn
import tensorflow as tf
import os
import shutil
import time
import subprocess
import random
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET


'''
Humanoid original parameters:
l_th = -0.34, l_sh = -0.3, l_f_s = 0.075, l_s_s = 0.049,l_t_s = 0.06 ------> Legs
r_th = -0.34, r_sh = -0.3, r_f_s = 0.075, r_s_s = 0.049, r_t_s = 0.06
l_u_a = .24, l_l_a = .17, l_h = 0.06 ----> Hands 
r_u_a = .24, r_l_a = .17, r_h = 0.06
'''

# space = {'l_th': hp.uniform('l_th', -0.34*0.5, -0.34*1.5), 'l_sh': hp.uniform('l_sh', -0.3*0.5, -0.3*1.5),
#          'l_f_s': hp.uniform('l_f_s', 0.075*0.5, 0.075*1.5), 'l_t_s': hp.uniform('l_t_s', 0.06*0.5, 0.06*1.5),
#          'l_s_s': hp.uniform('l_f_s', 0.049*0.5, 0.049*1.5), 
#          'r_th': hp.uniform('r_th', -0.34*0.5, -0.34*1.5), 'r_sh': hp.uniform('r_sh', -0.3*0.5, -0.3*1.5), 
#          'r_f_s': hp.uniform('r_f_s', 0.075*0.5, 0.075*1.5), 'r_t_s': hp.uniform('l_t_s', 0.06*0.5, 0.06*1.5),
#          'r_s_s': hp.uniform('l_f_s', 0.049*0.5, 0.049*1.5), #legs
#          'l_u_a': hp.uniform('l_u_a', 0.24*0.5, 0.24*1.5), 'l_l_a': hp.uniform('l_l_a', 0.17*0.5, 0.17*1.5),
#          'l_h': hp.uniform('l_h', 0.06*0.5, 0.06*1.5),
#          'r_u_a': hp.uniform('r_u_a', 0.24*0.5, 0.24*1.5), 'r_l_a': hp.uniform('r_l_a', 0.17*0.5, 0.17*1.5),
#          'r_h': hp.uniform('r_h', 0.06*0.5, 0.06*1.5)
#          }

space = {'l_th': hp.uniform('l_th', -0.34*0.5, -0.34*1.5), 'l_sh': hp.uniform('l_sh', -0.3*0.5, -0.3*1.5),
         'l_f_s': hp.uniform('l_f_s', 0.075*0.5, 0.075*1.5), 
         'r_th': hp.uniform('r_th', -0.34*0.5, -0.34*1.5), 'r_sh': hp.uniform('r_sh', -0.3*0.5, -0.3*1.5), 
         'r_f_s': hp.uniform('r_f_s', 0.075*0.5, 0.075*1.5)
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


def train_params(args):
    def _train_params(params, all_returns, xml_path, folder_path):

        #from run_ppo import train
        global UPN_PI

        l_th = round(params.get('l_th'), 4)
        l_sh = round(params.get('l_sh'), 4)
        r_th = round(params.get('r_th'), 4)
        r_sh = round(params.get('r_sh'), 4)
        l_f_s = round(params.get('l_f_s'), 4)
        r_f_s = round(params.get('r_f_s'), 4)
        # l_s_s = round(params.get('l_s_s'), 4)
        # r_s_s = round(params.get('r_s_s'), 4)
        # l_t_s = round(params.get('l_t_s'), 4)
        # r_t_s = round(params.get('r_t_s'), 4)
        # l_u_a = round(params.get('l_u_a'), 4)
        # l_l_a = round(params.get('l_l_a'), 4)
        # l_h = round(params.get('l_h'), 4)
        # r_u_a = round(params.get('r_u_a'), 4)
        # r_l_a = round(params.get('r_l_a'), 4)
        # r_h = round(params.get('r_h'), 4)
         
        # l_th = -0.34 
        # l_sh = -0.3 
        # r_th = -0.34 
        # r_sh = -0.3 
        # r_s_s = 0.049
        # r_t_s = 0.06
        # l_s_s = 0.049
        # l_t_s = 0.06
        # l_f_s = 0.075
        # r_f_s = 0.075
        # l_u_a = .24
        # l_l_a = .17
        # l_h = 0.06 
        # r_u_a = .24 
        # r_l_a = .17
        # r_h = 0.06
    
        # ########### Xml file modification part - start ###############

        import xml.etree.ElementTree as ET

        def compute_pos(value, initial_fromto, initial_pos):
            points_of_value = value.split()
            print("POV", points_of_value)
            points_of_initialfromto = initial_fromto.split()
            x1, y1, z1, x2, y2, z2 = map(float, points_of_value)
            x3, y3, z3, x4, y4, z4 = map(float, points_of_initialfromto)

            x_diff = x2 - x4
            y_diff = y2 - y4
            z_diff = z2 - z4

            points_of_pos = initial_pos.split()
            x5, y5, z5 = map(float, points_of_pos)
            x5 = x5 + x_diff
            y5 = y5 + y_diff
            z5 = z5 + z_diff

            return f"{x5} {y5} {z5}"

        tree = ET.parse(xml_path)
        root = tree.getroot()

        values_dict = {
            'left_thigh': '0 0 0 0 -0.01' + ' ' + str(l_th),
            'left_shin': '0 0 0 0 0' + ' '+ str(l_sh),
            'right_thigh': '0 0 0 0 0.01' + ' ' + str(r_th),
            'right_shin': '0 0 0 0 0' + ' ' + str(r_sh),
            # 'right_upper_arm': '0 0 0' + ' ' + str(r_u_a) + ' '+ str(-r_u_a) + ' '+ str(-r_u_a),
            # 'right_lower_arm': '0 0 0' + ' ' + str(r_l_a) + ' '+ str(r_l_a) + ' '+ str(r_l_a),
            # 'left_upper_arm': '0 0 0' + ' ' + str(l_u_a) + ' '+ str(l_u_a) + ' '+ str(-l_u_a),
            # 'left_lower_arm': '0 0 0' + ' ' + str(l_u_a) + ' '+ str(-l_u_a) + ' '+ str(l_u_a)
        }

        for key, new_fromto in values_dict.items():
            # Find all <body> elements with the specified name
            body_elements = root.findall(f".//body[@name='{key}']")

            geom = body_elements[0].find('geom')
            if geom is not None:
                initial_fromto = geom.get('fromto')
                # Update the "fromto" attribute value
                geom.set('fromto', new_fromto)

            if "thigh" in key:
                shin_body = body_elements[0].find("body")
                initial_pos = shin_body.get('pos')
                pos = compute_pos(new_fromto, initial_fromto, initial_pos)
                shin_body.set('pos', pos)

            if "shin" in key:
                foot_body = body_elements[0].find("body")
                initial_pos = foot_body.get('pos')
                pos = compute_pos(new_fromto, initial_fromto, initial_pos)
                foot_body.set('pos', pos)
            
            # if "upper_arm" in key:
            #     arm_body = body_elements[0].find("body")
            #     initial_arm_pos = arm_body.get('pos')
            #     pos = compute_pos(new_fromto, initial_fromto, initial_arm_pos)
            #     arm_body.set('pos', pos)

            # if "lower_arm" in key:
            #     arm_lower_body = body_elements[0].find("body")
            #     initial_arm_pos = arm_lower_body.get('pos')
            #     pos = compute_pos(new_fromto, initial_fromto, initial_arm_pos)
            #     arm_lower_body.set('pos', pos)

        size_dict = {
            'left_foot': str(l_f_s),
            'right_foot': str(r_f_s),
            # 'left_shin': str(l_s_s),
            # 'right_shin': str(r_s_s),
            # 'left_thigh': str(l_t_s), 
            # 'right_thigh': str(r_t_s),
            # 'left_hand': str(l_h),
            # 'right_hand': str(r_h)
        }
        # Iterate over the dictionary and update the size attribute for the corresponding bodies
        for key, value in size_dict.items():
            body = root.find(".//body[@name='" + key + "']")
            if body is not None:
                geom = body.find('geom')
                if geom is not None:
                    # Update the size attribute value
                    geom.set('size', value)
                    print(f"Updated the 'size' attribute value to '{value}' for <geom> in <body> with name='{key}'.")
                else:
                    print(f"<geom> element not found in the <body> element with name='{key}'.")
            else:
                print(f"No <body> element with name='{key}' found.")

        tree.write(xml_path)

        ########### Xml file modification part - end ###############

        # print("n_iter", n_iterations)

        #Calling a script run_ppo.py
        args2 = ["python", "policy_transfer/ppo/humanoid/design_eval.py"]
        args2 += ["--l_th", l_th, "--l_sh", l_sh, "--l_f_s", l_f_s]
        args2 += ["--r_th", r_th, "--r_sh", r_sh, "--r_f_s", r_f_s]

        
       # # args2 += ["--l_th", l_th, "--l_sh", l_sh, "--l_f_s", l_f_s, "--l_t_s", l_t_s, "--l_s_s", l_s_s]
       # # args2 += ["--r_th", r_th, "--r_sh", r_sh, "--r_f_s", r_f_s, "--r_t_s", r_t_s, "--r_s_s", r_s_s]
       # # args2 += ["--l_u_a", l_u_a, "--l_l_a", l_l_a, "--l_h", l_h]
       # # args2 += ["--r_u_a", r_u_a, "--r_l_a", r_l_a, "--r_h", r_h]
        # # args2 += ["--n_iter", n_iterations, "--warmstart", UPN_PI, "--env", args.env]
        # args2 += ["--seed", args.seed, "--name", args.name, "--min_timesteps", args.min_timesteps]
        # args2 += ["--batch_size", args.batch_size]
        # args2 += ["--clip", args.clip]
        # args2 += ["--schedule", args.schedule]
        # args2 += ["--train_up", args.train_up]
        # args2 += ["--output_interval", args.output_interval]
        # args2 += ["--mirror", args.mirror]
        args2 += ["--path", folder_path]
        args2 += ["--xml_path", xml_path]

        
        run(args2)

        subprocess.run("pwd", shell=True)
        j = open(os.path.join(folder_path, './last_return.json'))
        avg_training_rew = json.load(j)

        all_returns.append(avg_training_rew)

        with open(os.path.join(folder_path, "all_returns.txt"), 'w+') as output_file:
            json.dump(all_returns, output_file)
        
        return avg_training_rew

    return _train_params

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--env', help='environment ID', default=None)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--config_count', help='Max configs in a filter', type=int, default=27)
    # parser.add_argument('--name', help='name of experiments', type=str, default="")
    # parser.add_argument('--min_timesteps', help='mimimum budget size', type=int)
    # parser.add_argument('--batch_size', help='batch size', type=int, default=4000)
    # parser.add_argument('--clip', help='clip', type=float, default=0.2)
    # parser.add_argument('--schedule', help='schedule', default='constant')
    # parser.add_argument('--train_up', help='whether train up', default='True')
    # parser.add_argument('--output_interval', help='interval of outputting policies', type=int, default=10)
    # parser.add_argument('--mirror', help='whether to use mirror, (0: not mirror, 1: hard mirror, 2: soft mirror)', type=int, default=0)
    # parser.add_argument('--warmstart', help='path to warmstart policies', type=str, default="")
    parser.add_argument('--xml_path')
    parser.add_argument("--run_path", default="./")
    # parser.add_argument("--order")

    args = parser.parse_args()

    # set_global_seeds(args.seed)

    # print("NAME OF EXPERIMENT", args.name)
    
    # Step 1: Create a new folder
    postfix = int(time.time())
    folder_name = "run_eval_{}".format(postfix)
    new_folder_path = os.path.join(args.run_path, folder_name)
    os.mkdir(new_folder_path)

    # Step 2: Copy the xml file to this folder
    shutil.copy(args.xml_path, os.path.join(os.path.abspath(new_folder_path), "env.xml"))
    xml_path = os.path.join(os.path.abspath(new_folder_path), "env.xml")

    data_folder = os.path.join(new_folder_path, "data")
    os.mkdir(data_folder)

    graphs_folder = os.path.join(data_folder, "graphs")
    os.mkdir(graphs_folder)

    hb = Sampler(get_params, train_params(args), xml_path, data_folder, args.seed, args.config_count)
    results = hb.run(skip_last=1)


    #writing final output to file
    f = open(os.path.join(data_folder, "out_summary.txt"), 'w+')
    # f.write("Total Resources Used:" + str(total_iterations)+ "\n")
    f.write("Results ="+ str(results) + "\n")
    f.write("-->" + str(len(results)) + " total runs, best:\n")
    final_returns = []
    for r in sorted(results, key=lambda x: x['Return'], reverse=True)[:]:
        f.write("Return:" + str(r['Return']) +" " + str(r['params']) + "\n")
        
    for s in results:    
        final_returns.append(s['Return'])

    np_final_returns = np.array(final_returns)    
    plt.figure()
    plt.plot(np_final_returns)
    plt.savefig(os.path.join(data_folder, './graphs/graph.png'))                  #location
    

if __name__ == '__main__':
    main()