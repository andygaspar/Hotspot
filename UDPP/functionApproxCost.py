from typing import Union, Callable, List
from copy import copy

from ..ModelStructure.Airline.airline import Airline
from ..ModelStructure.modelStructure import ModelStructure
from ..ModelStructure.Slot.slot import Slot
from ..ModelStructure.Flight import flight as fl

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, minimize
import scipy
import math
import numpy as np
import dill as pickle
import time
import multiprocessing
#from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor as Pool

num_cpu = multiprocessing.cpu_count()

# def approx_linear(x, slope):
#     return slope * x

# def approx_one_margins(x, margin_1, jump_1):
#     cost = np.zeros_like(x)
#     cost[margin_1 <= x] = jump_1
#     return cost

# def approx_two_margins(x, margin_1, jump_1, margin_2, jump_2):
#     cost = approx_one_margins(x, margin_1, jump_1)
#     cost[margin_2 <= x] = jump_2
#     return cost

# def approx_slope_one_margin(x, slope, margin_1, jump_1):
#     cost = approx_linear(x, slope)
#     cost[margin_1 <= x] = jump_1
#     return cost

# def approx_slope_two_margins(x, slope, margin_1, jump_1, margin_2, jump_2):
#     cost = approx_slope_one_margin(x, slope, margin_1, jump_1)
#     cost[margin_2 <= x] = jump_2
#     return cost

# def approx_three_margins(x, slope, margin_1, jump_1, margin_2, jump_2, margin_3, jump_3):
#     cost = approx_slope_two_margins(x, slope, margin_1, jump_1, margin_2, jump_2)
#     cost[margin_3 <= x] = jump_3
#     return cost

def loss_fun(Y_test, y_train):
    return sum(
        [(y_train[i] - Y_test[i]) ** 2 /10 if y_train[i] - Y_test[i] > 0 else (Y_test[i] - y_train[i]) ** 2 for i in
         range(Y_test.shape[0])])

def obj_approx(params, fixed_params, x, y, approx_fun):
    return loss_fun(y, approx_fun(x, **fixed_params, **{k:params[i] for i, k in enumerate(approx_fun.get_var_paras())}))

def fit_curve(vals):
    x, y = vals[:2]
    approx_fun = vals[-1]
    fixed_params = vals[2]
    params = vals[3]
    return loss_fun(y, approx_fun(x, **fixed_params, **params))

def compute_test_values(x, y, max_delay, approx_fun, fixed_paras={}, steps=8):
    if approx_fun.nickname=='double_jump':
        test_values = []
        max_val = max(y)
        for slope in np.linspace(0, 1, steps):
            for margin_1 in np.linspace(0, 3*max_delay//4, steps):
                for jump_1 in np.linspace(10, max_val, steps//2):
                    for margin_2 in np.linspace(margin_1, max_delay, steps//2):
                        for jump_2 in np.linspace(jump_1, max_val, steps//2):
                            params = {}
                            params['slope'] = slope
                            params['margin_1'] = margin_1
                            params['jump_1'] = jump_1
                            params['margin_2'] = margin_2
                            params['jump_2'] = jump_2

                            test_values.append(
                                (x, y, fixed_paras, params, approx_fun))

    elif approx_fun.nickname=='jump':
        test_values = []
        max_val = max(y)
        for slope in np.linspace(0, 1, steps):
            for margin in np.linspace(0, 3*max_delay//4, steps):
                for jump in np.linspace(10, max_val, steps//2):
                    params = {}
                    params['slope'] = slope
                    params['margin'] = margin
                    params['jump'] = jump
                    
                    test_values.append(
                        (x, y, fixed_paras, params, approx_fun))
    
    elif approx_fun.nickname=='jump2':
        test_values = []
        max_val = max(y)
        for margin in np.linspace(0, 3*max_delay//4, steps):
            for jump in np.linspace(10, max_val, steps//2):
                params = {}
                params['margin'] = margin
                params['jump'] = jump
                
                test_values.append(
                    (x, y, fixed_paras, params, approx_fun))
    else:
        raise Exception('Cost function approximator not implemented for', approx_fun.nickname)

    return test_values

def fit_cost_curve(x, y, max_delay, fixed_paras={}, steps=8, approx_fun=None):
    test_values = compute_test_values(x, y, max_delay, approx_fun,
                                    fixed_paras=fixed_paras, steps=steps)
    
    #pool = Pool(num_cpu)
    with Pool(max_workers=num_cpu) as pool:
        guesses = pool.map(fit_curve, test_values)
    # best_initial_guess = np.array(test_values[np.argmin(guesses)][2:-1])
    # Here I don't use the values method because I want to make sure that
    # these values are in the same order as they appear in the archetype
    # function
    best_initial_guess = np.array([test_values[np.argmin(guesses)][3][k] for k in approx_fun.get_var_paras()])

    if approx_fun.nickname=='jump':
        bounds = ([0., None], [0., None], [0., None])
    elif approx_fun.nickname=='jump2':
        bounds = ([0., None], [0., None])

    # print ('BEST INITIAL GUESS:', best_initial_guess)

    solution = minimize(obj_approx,
                        best_initial_guess,
                        args=(fixed_paras, x, y, approx_fun),
                        #method='L-BFGS-B',
                        method='Powell',
                        options={'maxiter': 10000,
                                'xtol': 0.5,
                                'ftol': 0.01},
                        bounds=bounds)

    # if best_initial_guess[0]==0.:
    #     print ('SOLUTION:', solution)
    #     import matplotlib.pyplot as plt
    #     plt.plot(x, y)
    #     plt.plot(x, [approx_fun(xx,
    #                             eta=fixed_paras['eta'],
    #                             margin=solution.x[0],
    #                             jump=solution.x[1]) for xx in x])
    #     plt.show()

        # raise Exception()

    return solution.x

def make_preference_fun(max_delay: float, delay_cost_vect: np.array, fixed_paras={}, approx_fun=None):
    delays = np.linspace(fixed_paras['eta'], fixed_paras['eta']+max_delay + 0.1, len(delay_cost_vect))#50)
    result = fit_cost_curve(delays, delay_cost_vect, max_delay,
                            fixed_paras=fixed_paras, approx_fun=approx_fun)
    return result
    # plt.plot(delay_cost_vect)
    # plt.plot(approx_fun(delays, slope, margin_1, jump_1, margin_2, jump_2))
    # plt.show()


class FunctionApproxCost(ModelStructure):
    requirements = ['delayCostVect']

    def __init__(self, slots: List[Slot]=None, flights: List[fl.Flight]=None,
        cost_func_archetype=None, alternative_allocation_rule=False):

        self.cost_func_archetype = cost_func_archetype
        
        if not flights is None:
            super().__init__(slots,
                            flights,
                            alternative_allocation_rule=alternative_allocation_rule,
                            air_ctor=Airline)

    def run(self):
        all_paras = {}
        for flight in self.flights:
            max_delay = self.slots[-1].time - self.slots[0].time
            fixed_parameters = {attr:getattr(flight, attr) for attr in self.cost_func_archetype.fixed_paras}
            paras = make_preference_fun(max_delay,
                                        flight.delayCostVect,
                                        fixed_paras=fixed_parameters,
                                        approx_fun=self.cost_func_archetype)
            
            for i, k in enumerate(self.cost_func_archetype.get_var_paras()):
                setattr(flight, k, paras[i])

            all_paras[flight.name] = {k:paras[i] for i, k in enumerate(self.cost_func_archetype.get_var_paras())}
        return all_paras