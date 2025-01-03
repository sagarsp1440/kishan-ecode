import numpy as np
from random import random
from math import log, ceil
from time import time, ctime


class Hyperband2:

    def __init__(self, env, get_params_function, train_params_function, xml_path, data_folder_path, seed, config_count):
        self.get_params = get_params_function
        self.train_params = train_params_function
        self.xml_path = xml_path
        self.data_folder_path = data_folder_path
        self.seed = seed
        self.env_name = env
        self.max_episodes = config_count  # maximum episodes per configuration/no. of configurations
        self.eta = 3  # defines configuration down sampling rate (default = 3)
        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.max_episodes))
        #print("smax", self.s_max)
        self.B = (self.s_max + 1) * self.max_episodes  # total budget
        #print("B", self.B)
        self.results = []  # list of dicts
        self.counter = 0
        self.best_return = 0
        self.best_counter = -1

    # can be called multiple times
    def run(self, skip_last=0, dry_run=False):
        total_iterations = 0
        best_filter_return = []
        for s in range(self.s_max + 1):
            # initial number of configurations
            num_config = int(ceil((self.B * self.eta ** s) / (self.max_episodes * (self.s_max + 1))))
            #print("num_config", num_config)
            # initial number of iterations per config/minimum resource allocated to all configs
            r = self.max_episodes * self.eta ** (-s)
            # n random configurations
            rng = np.random.RandomState(seed=s+self.seed)
            #print(self.get_params)
            T = [self.get_params(env=self.env_name, rng=None) for i in range(num_config)]
            #print("T", T)

            for i in range((s + 2) - int(skip_last)):  # changed from s + 1    #changed s+1 to s+2 - Kishan

                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations
                n_configs = num_config * self.eta ** (-i)
                n_iterations = r * self.eta ** i
                print("\n*** {} configurations x {:.1f} iterations each".format(
                    n_configs, n_iterations))

                Returns = []
                early_stops = []
                for t in T:
                    self.counter += 1
                    print("\n Count: {} | {} | highest return so far: {:.4f} (run {})\n", self.counter, ctime(), self.best_return, self.best_counter)
                    start_time = time()
                    result = self.train_params(self.env_name, n_iterations, t, self.xml_path, self.data_folder_path)
                    
                    print("1. Result", result)
                    result = {'Return': result}
                    assert (type(result) == dict)
                    assert ('Return' in result)
                    print("2. Result", result)

                    seconds = int(round(time() - start_time))

                    Return = result['Return']
                    Returns.append(Return)
                    print("Returns", len(Returns), Returns)

                    early_stop = result.get('early_stop', False)
                    early_stops.append(early_stop)

                    # keeping track of the best result so far (for display only)
                    # could do it be checking results each time, but hey
                    if Return > self.best_return:
                        self.best_return = Return
                        self.best_counter = self.counter

                    result['counter'] = self.counter
                    result['seconds'] = seconds
                    result['params'] = t
                    result['iterations'] = n_iterations
                    result['id'] = s

                    total_iterations = total_iterations + n_iterations
                
                    self.results.append(result)

                # select a number of the best configurations for the next loop
                # filter out early stops, if any

                print("Returns", len(Returns), Returns)
                indices = np.argsort(Returns)[::-1][:]
                print("T", T, type(T))

                T = [T[i] for i in indices]
                print("T2", len(T), T)
                T = T[0:int(n_configs / self.eta)]
                print("T3", len(T), T)

            best_filter_return.append(self.best_return)



        return self.results, total_iterations, best_filter_return