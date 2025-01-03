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
from gym.wrappers.time_limit import TimeLimit
from gym.wrappers import Monitor
from PIL import Image 
import PIL
import cv2
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
    
    lengths = []
    n_iterations = args.n_iter

    for i in all_params:
        length = i['length']
        lengths.append(length)
        print("lengths", length) 

    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    seed = 0
    set_global_seeds(seed)

    #running gym session 

    env = gym.make("CartPole-v1", length = length)   #This is a sample length value
    env.env.obs_dim = 4 + 1  # 4 because self.state size is 4 +1 because only length

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
    
    j = open('./data/test.json')
    data = json.load(j)
    UPN_PI = data['pi']
    
    pi = init_policy(pi_init, joblib.load(UPN_PI))
    
    avg_rlreturns = []
    for l in lengths:
        length_HB = l
        print("length_HB", length_HB)

        env = gym.make("CartPole-v1", length = length_HB)
        env.env.obs_dim = 4 + 1  # 4 because self.state size is 4 +1 because only length

        high = np.inf * np.ones(env.env.obs_dim)
        low = -high
        env.env.observation_space = spaces.Box(low, high)
        env.observation_space = spaces.Box(low, high) 
        
        #videodims = (100,100)
        #fourcc = cv2.VideoWriter_fourcc(*'avc1')    
        #video = cv2.VideoWriter("./video/test.mp4",fourcc, 60,videodims)
        
        #extn_env = Monitor(TimeLimit(EnvWrapper(env, up_dim=[length_HB]), 
        #max_episode_steps=500),'.video/tt-'+str(l)+'.mp4',force=True,write_upon_reset=True)  #wrapped env
        extn_env=EnvWrapper(env, up_dim=[length_HB])
        rl_return = []
        sum_of_rewards = 0
        epi_count = 0
        done_eval = False
        step_count = 0
        num_evals = 1

        obs = extn_env.reset()
        images = []
        while epi_count < num_evals:
            action, _states = pi.act(0, obs)
            obs, rewards, done_eval, info = extn_env.step(action)
            step_count = step_count + 1
            #print("INSIDE EVALUATION STEPS")
            if step_count >= 500:
                done_eval = True
            # modified to get HB output
            sum_of_rewards = sum_of_rewards + rewards
            rl_return.append(rewards)  # store each of the 1000 values
            #video.write(cv2.cvtColor(extn_env.render(mode='rgb_array'), cv2.COLOR_RGB2BGR))
            images.append(Image.fromarray(extn_env.render(mode = "rgb_array")))
            if done_eval:
                # print("################ Done received in evaluate.py ###############")
                epi_count += 1
                extn_env.reset()
                #print("TRYIN TO RENDER")    
                #image = extn_env.render(mode = "rgb_array")
                #im = Image.fromarray(image)
                #im.save("./video/test.png")
                step_count = 0
        avg_sum_of_rewards = sum_of_rewards/num_evals
        avg_rlreturns.append(avg_sum_of_rewards)
        images[0].save('./video/tt-'+str(l)+'.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)
    #video.release()
    config_name = "./data"
    with open(os.path.join(config_name, 'return.json'), 'w') as outfile:
        json.dump(avg_rlreturns, outfile)

    sess.close()

    # return _eval_params


if __name__ == '__main__':
    main()