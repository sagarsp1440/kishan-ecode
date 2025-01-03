# pip install hyperopt == 0.1.1
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from hyperband import Hyperband
#from QLearning import QL_alg
from robot_ppo import RL_Alg

# handle floats which should be integers
# works with flat params

x = [0,1,2,3,4]
space = {'0': hp.choice('0', x), 
        '1': hp.choice('1', x),
        '2': hp.choice('2', x),
        '3': hp.choice('3', x),
        '4': hp.choice('4', x),
        '5': hp.choice('5', x),
        '6': hp.choice('6', x),
        '7': hp.choice('7', x),
        '8': hp.choice('8', x),
        '9': hp.choice('9', x),
        '10': hp.choice('10', x),
        '11': hp.choice('11', x),
        '12': hp.choice('12', x),
        '13': hp.choice('13', x),
        '14': hp.choice('14', x),
        '15': hp.choice('15', x)
         }


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
    print("a", params)
    return handle_integers(params)


def try_params(n_iterations, t):
    Return = RL_Alg(t, n_iterations)
    return Return


hb = Hyperband(get_params, try_params)
results, total_iterations, best_filter_return = hb.run(skip_last=1)
#print("Best filter return", best_filter_return)

#writing final output to file
f = open("out_summary.txt", 'w+')
f.write("Total Resources Used:" + str(total_iterations)+ "\n")
f.write("Results ="+ str(results) + "\n")
f.write("-->" + str(len(results)) + " total runs, best:\n")
for r in sorted(results, key=lambda x: x['Return'], reverse=True)[:10]:
    f.write("Return:" + str(r['Return']) + " " + " " + str(r['iterations'])+ "Iterations " + "ID:" + str(r['id']) + str(r['params']) + "\n")
#for i in best_filter_return:
#    f.write("Best Filter return " + str(i)+ "\n")
