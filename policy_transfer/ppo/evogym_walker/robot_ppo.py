import gym
import numpy as np

import os
import matplotlib.pyplot as plt

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback


from stable_baselines3 import DDPG
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from evogym import sample_robot
import envs
import random

# Create log dir
# log_dir = "tmp/"
# os.makedirs(log_dir, exist_ok=True)

random.seed(10)

def RL_Alg(params, n_iterations):
        dones = False
        rl_return = []
        sum_of_rewards = 0
        epi_count = 0
        avg_rlreturn = 0

        values = list(params.values())
        print("Values", values)
        count = n_iterations * 25000 

        body, connections, flag = sample_robot((4,4), values)

        if flag == "good":

                env = gym.make('SimpleWalkingEnv-v0', body=body)

                model = PPO("MlpPolicy", env, verbose=1)
                model.learn(total_timesteps=count, log_interval=4)
                model.save("d_test")
                env = model.get_env()

                obs = env.reset()
                while not dones:
                        action, _states = model.predict(obs)
                        print("####### states", _states)
                        obs, rewards, dones, info = env.step(action)
                        #modified to get HB output
                        sum_of_rewards = sum_of_rewards + rewards
                        rl_return.append(sum_of_rewards)  # store each of the 1000 values
                        epi_count += 1
                        if epi_count > n_iterations:
                                dones = True
                # plot_results([log_dir], timesteps=count)
                # plt.show()

                #rl_return = np.array(rl_return)
                avg_rlreturn = np.average(rl_return, axis=0)
                print("Avg_rlreturn", avg_rlreturn[0])
        
                # f = open("ListofReturns.txt", mode = 'a') 
                # f.writelines("length:" +str(length_HB)+ "---->")
                # f.writelines(str(avg_rlreturn[0]) + "\n")
                
                return avg_rlreturn[0]

        else:
                avg_rlreturn = -10000
                return avg_rlreturn
        
      

             
#def RL_Alg(params, n_iterations):
        # params = params
        # n_iterations = n_iterations

        # for x in range(1):
        # # create a random robot
        # # {'0': 1, '1': 4, '2': 2, '3': 0, '4': 0, '5': 1, '6': 3, '7': 3, '8': 4} 27.0Iterations ID:2
        # # {'0': 1, '1': 4, '2': 2, '3': 0, '4': 0, '5': 1, '6': 3, '7': 3, '8': 4} 9.0Iterations ID:2
        # # {'0': 1, '1': 3, '2': 2, '3': 3, '4': 1, '5': 0, '6': 2, '7': 2, '8': 2} 27.0Iterations ID:3
        # # {'0': 4, '1': 3, '2': 1, '3': 4, '4': 1, '5': 4, '6': 1, '7': 1, '8': 2} 27.0Iterations ID:1
        #         dict =  {'0': 2, '1': 4, '10': 2, '11': 3, '12': 4, '13': 3, '14': 3, '15': 4, '2': 4, '3': 1, '4': 2, '5': 3, '6': 2, '7': 4, '8': 3, '9': 3}
        #         dict2 = {'0': 2, '1': 1, '10': 3, '11': 4, '12': 0, '13': 1, '14': 1, '15': 0, '2': 4, '3': 0, '4': 1, '5': 1, '6': 3, '7': 4, '8': 1, '9': 3}
        #         dict3 = {'0': 2, '1': 1, '10': 4, '11': 2, '12': 2, '13': 2, '14': 4, '15': 1, '2': 0, '3': 0, '4': 3, '5': 1, '6': 4, '7': 3, '8': 3, '9': 4}
        #         values = list(dict3.values())

        #         body, connections, flag = sample_robot((4,4), values)

        #         # make the SimpleWalkingEnv using gym.make and with the robot information
        #         env = gym.make('SimpleWalkingEnv-v0', body=body)
        #         env.reset()

        #         # step the environment for 500 iterations
        #         for i in range(50000):

        #                 action = env.action_space.sample()
        #                 ob, reward, done, info = env.step(action)
        #                 env.render(verbose=True)

        #                 if done:
        #                         env.reset()

        #         env.close()



