import numpy as np
import random
import pandas as pd
import pickle

random.seed(2)
np.random.seed(2)
n_f = 10

def simple_function(delay, cost_coefficient):
	return cost_coefficient * delay ** 2

dict_funcs = {'simple_function':simple_function}


arch_func = 'simple_function'
archetype_function = dict_funcs[arch_func]
capacity_drop = 2.
cost_funcs = {}
df = {'eta':[], 'airlineName':[], 'time':[], 'name':[],
	'cost_coefficient':[]}
for i in range(n_f):
	eta = i
	df['eta'].append(eta)
	airlineName = np.random.choice(['A', 'B', 'C'])
	df['airlineName'].append(airlineName)
	df['time'].append(int(eta * capacity_drop))
	name = airlineName + str(i)
	df['name'].append(name)
	cost_coefficient = np.random.uniform(0.5, 2, 1)[0]
	df['cost_coefficient'].append(cost_coefficient)
	# cost_func = lambda delay: archetype_function(delay, cost_coefficient)
	# cost_funcs[name] = cost_func

df = pd.DataFrame(df)
df.to_csv('../test_data/df_test_1.csv')

with open('../test_data/archetype_function_test_1.pic', 'wb') as f:
	pickle.dump(arch_func, f)

