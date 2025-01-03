import argparse
import json
import gym, numpy as np
from gym import spaces
from policy_transfer.policies.mlp_policy import MlpPolicy
import tensorflow as tf
import joblib
from env_wrapper import EnvWrapper
from baselines.common import set_global_seeds, tf_util as U
import os


def init_policy(pi, init_policy_params):

    U.initialize()

    cur_scope = pi.get_variables()[0].name[0:pi.get_variables()[0].name.find('/')]
    orig_scope = list(init_policy_params.keys())[0][0:list(init_policy_params.keys())[0].find('/')]
    print(cur_scope, orig_scope)
    for i in range(len(pi.get_variables())):
        if pi.get_variables()[i].name.replace(cur_scope, orig_scope, 1) in init_policy_params:
            assign_op = pi.get_variables()[i].assign(init_policy_params[pi.get_variables()[i].name.replace(cur_scope, orig_scope, 1)])
            tf.get_default_session().run(assign_op)
    
    return pi


def main():
    parser = argparse.ArgumentParser()
    # note type arg, used to load json string
    parser.add_argument('--n_iter', type=float)
    args = parser.parse_args()
    j = open('./data/params.json')
    all_params = json.load(j)

    print("in evaluate.py------------------------")
    

    #Opened evaluate.py
    print("############ all params###########", all_params, type(all_params))
    l1 = []
    n_iterations = args.n_iter

    for i in all_params:
        l1 = i['l1']
        l2 = i['l2']
        ml1 = i['ml1']
        ml2 = i['ml2']
        l1.append(l1)
        
    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    seed = 0
    set_global_seeds(seed)

    #running gym session 

    env = gym.make("Walker2d-v3")   #This is a sample length value
    env.env.obs_dim = 17 + 4  # 4 because self.state size is 4 +1 because only length

    high = np.inf * np.ones(env.env.obs_dim)
    low = -high
    env.env.observation_space = spaces.Box(low, high)
    env.observation_space = spaces.Box(low, high)

    ob_space = env.observation_space
    ac_space = env.action_space

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                             hid_size=64, num_hid_layers=3)


    pi_init = policy_fn('pi', ob_space, ac_space) # Construct network for new policy
    
    # j = open('./data/test.json')
    # data = json.load(j)
    UPN_PI = 
    
    pi = init_policy(pi_init, joblib.load(UPN_PI))
    
    avg_rlreturns = []
    for l in lengths:
        length_HB = l
        print("length_HB", length_HB)

        env = gym.make("CartPole-v1", length = length_HB, render="human")
        env.env.obs_dim = 4 + 1  # 4 because self.state size is 4 +1 because only length

        high = np.inf * np.ones(env.env.obs_dim)
        low = -high
        env.env.observation_space = spaces.Box(low, high)
        env.observation_space = spaces.Box(low, high)  

        extn_env = EnvWrapper(env, up_dim=[length_HB])   #wrapped env

        rl_return = []
        sum_of_rewards = 0
        epi_count = 0
        done_eval = False
        step_count = 0
        num_evals = 10

        obs = extn_env.reset()
        while epi_count < num_evals:
            action, _states = pi.act(0, obs)
            obs, rewards, done_eval, info = extn_env.step(action)
            step_count = step_count + 1
            if step_count >= 500:
                done_eval = True
            # modified to get HB output
            sum_of_rewards = sum_of_rewards + rewards
            rl_return.append(rewards)  # store each of the 1000 values
            extn_env.render()
            if done_eval:
                # print("################ Done received in evaluate.py ###############")
                epi_count += 1
                extn_env.reset()
                step_count = 0
            
        avg_sum_of_rewards = sum_of_rewards/num_evals
        avg_rlreturns.append(avg_sum_of_rewards)
        

    config_name = "./data"
    with open(os.path.join(config_name, 'return.json'), 'w') as outfile:
        json.dump(avg_rlreturns, outfile)

    sess.close()

    # return _eval_params


if __name__ == '__main__':
    main()