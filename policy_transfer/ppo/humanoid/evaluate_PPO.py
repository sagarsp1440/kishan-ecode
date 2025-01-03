from gflownet.baselines.baselines.ppo2 import ppo2
from gflownet.baselines.baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from gflownet.baselines.baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from gflownet.baselines.baselines.common import set_global_seeds, tf_util as U
# from gflownet.algo.baselines.baselines.logger import configure
import os
import argparse
import gym
import json
import time

'''
This file accepts 1 xml robot file and performs PPO Training for a specified timestep budget. 
'''

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--env_id', help='environment ID', default=None)
    parser.add_argument('--total_timesteps', help='maximum step size', type=int)
    parser.add_argument('--network', help='path for data')
    parser.add_argument('--xml_file_path', help='path for xml')
    parser.add_argument('--perf_log_path', help='path for logging')
    parser.add_argument('--timestamp', help='unique_identifier')
    parser.add_argument('--ctrl_cost_weight', help='ctrl cost weight for gym env', default=0)
    parser.add_argument('--jsonData', help='json file to collect more dictionaries with rews and xmls')

    args = parser.parse_args()


    seed = 111

    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    set_global_seeds(seed)
    
    config_name = args.perf_log_path
    # logger.configure(dir=config_name)

    env = make_vec_env(args.env_id, ctrl_cost_weight = float(args.ctrl_cost_weight), xml_path=args.xml_file_path, env_type=args.env_id, num_env=1, seed=1, reward_scale=1.0, flatten_dict_observations=True) 
    model, eprewmean = ppo2.learn(network=args.network, env=env, total_timesteps=args.total_timesteps, eval_env = None, seed=None, nsteps=4000, ent_coef=0.01, lr=2e-3,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=10, cliprange=0.1,
            save_interval=0, load_path=None, model_fn=None, update_fn=None, init_fn=None, mpi_rank_weight=1, comm=None)
    
    sess.close()
    
    data = {args.xml_file_path: eprewmean}
    with open(os.path.join(config_name, './rews.json'), 'a') as outfile:
        json.dump(data, outfile)
        outfile.write(',')
    print("model", model)


if __name__ == '__main__':
    main()
