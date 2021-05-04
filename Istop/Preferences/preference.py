from scipy.optimize import curve_fit, minimize
import scipy
import math
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
import time
import multiprocessing
from multiprocessing import Pool

num_cpu = multiprocessing.cpu_count()

with open('ModelStructure/Costs/cost_functions_all.pck', 'rb') as dbfile:
    # dbfile = open('./cost_functions_1.pck', 'rb')
    dict_cost_func = pickle.load(dbfile)
dbfile.close()


def normalised_costs(flight, delays):
    costs = np.array([dict_cost_func[flight](x, True) for x in delays])
    return 100 * costs / max(costs)


def approx_linear(x, slope):
    return slope * x


def approx_one_margins(x, slope, margin_1, jump_1):
    cost = approx_linear(x, slope)
    cost[margin_1 <= x] = jump_1
    return cost


def approx_two_margins(x, slope, margin_1, jump_1, margin_2, jump_2):
    cost = approx_one_margins(x, slope, margin_1, jump_1)
    cost[margin_2 <= x] = jump_2
    return cost


def approx_three_margins(x, slope, margin_1, jump_1, margin_2, jump_2, margin_3, jump_3):
    cost = approx_two_margins(x, slope, margin_1, jump_1, margin_2, jump_2)
    cost[margin_3 <= x] = jump_3
    return cost


def loss_fun(Y_test, y_train):
    return sum(
        [(y_train[i] - Y_test[i]) ** 2 /10 if y_train[i] - Y_test[i] > 0 else (Y_test[i] - y_train[i]) ** 2 for i in
         range(Y_test.shape[0])])


def obj_approx(params, x, y, approx_fun):

    return loss_fun(y, approx_fun(x, *params))


def fit_curve(vals):
    x, y = vals[:2]
    approx_fun = vals[-1]
    params = vals[2:-1]
    return loss_fun(y, approx_fun(x, *params))


def fit_cost_curve(x, y, max_delay, steps= 6):
    test_values = []
    for slope in np.linspace(0, 1, steps):
        for margin_1 in np.linspace(0, 2*max_delay//3, steps):
            for jump_1 in np.linspace(10, 90, steps//2):
                for margin_2 in np.linspace(margin_1 , max_delay, steps//2):
                    for jump_2 in np.linspace(jump_1, 100, steps//2):
                        for margin_3 in np.linspace(margin_2 , max_delay, steps//2):
                            for jump_3 in np.linspace(jump_2, 100, steps//2):
                                test_values.append(
                                    (x, y, slope, margin_1, jump_1, margin_2, jump_2,
                                     margin_3, jump_3, approx_three_margins))
    pool = Pool(num_cpu)
    guesses = pool.map(fit_curve, test_values)
    best_initial_guess = np.array(test_values[np.argmin(guesses)][2:-1])

    solution = minimize(obj_approx, best_initial_guess, args=(x, y, approx_three_margins),
                        method='Powell', options={'maxiter': 10000, 'xtol': 0.5, 'ftol': 0.01})
    return solution.x


def make_preference_fun(flight_id, delays_vect, max_delay):
    costs = normalised_costs(flight_id, delays_vect)
    result = fit_cost_curve(delays_vect, costs, max_delay)
    slope, margin_1, jump_1, margin_2, jump_2, margin_3, jump_3 = result
    return lambda t: slope*t if t<margin_1 else jump_2 if t<margin_2 else jump_3

