import copy

import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
with open('ModelStructure/Costs/cost_functions_all.pck', 'rb') as dbfile:
    dict_cost_func = pickle.load(dbfile)
dbfile.close()

flights_dict_keys = np.array(list(dict_cost_func.keys()))


def make_random_cost_fun():
    flight_id = np.random.choice(flights_dict_keys, 1)[0]
    f = lambda t: dict_cost_func[flight_id](t, True)
    ff = copy.deepcopy(f)
    print(f.__code__)
    return ff