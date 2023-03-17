"""
This is the same than functionApprox but the input is the full cost 
function, not only the costVect vector.
"""

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
import pandas as pd
import dill as pickle
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor as Pool

num_cpu = multiprocessing.cpu_count()

def remove_nan_coupled_lists(list1, list2):
    list1 = np.array(list1)
    list2 = np.array(list2)
    mask = ~pd.isnull(list1) & ~pd.isnull(list2)
    list1 = list1[mask]
    list2 = list2[mask]

    return list1, list2

def r_squared(y, y_fit):
    """
    Returns the coefficient of determination of a fit.

    Parameters
    ==========
    y: iterable,
        initial points
    y_fit: iteratble,
        fitted points

    Returns
    =======
    r: float
        coefficient of determination

    Notes
    =====
    Discard nan values.

    """

    y_fit, y = remove_nan_coupled_lists(y_fit, y)

    y_bar = y.mean()
    SS_tot = sum((y-y_bar)**2)
    SS_res = sum((y-y_fit)**2)
    return 1. - SS_res/SS_tot

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

# def compute_test_values(x, y, max_delay, approx_fun, fixed_paras={}, steps=8):
#     if approx_fun.nickname=='double_jump':
#         test_values = []
#         max_val = max(y)
#         for slope in np.linspace(0, 1, steps):
#             for margin_1 in np.linspace(0, 3*max_delay//4, steps):
#                 for jump_1 in np.linspace(10, max_val, steps//2):
#                     for margin_2 in np.linspace(margin_1, max_delay, steps//2):
#                         for jump_2 in np.linspace(jump_1, max_val, steps//2):
#                             params = {}
#                             params['slope'] = slope
#                             params['margin_1'] = margin_1
#                             params['jump_1'] = jump_1
#                             params['margin_2'] = margin_2
#                             params['jump_2'] = jump_2

#                             test_values.append(
#                                 (x, y, fixed_paras, params, approx_fun))

#     elif approx_fun.nickname=='jump':
#         test_values = []
#         max_val = max(y)
#         for slope in np.linspace(0, 1, steps):
#             for margin in np.linspace(0, 3*max_delay//4, steps):
#                 for jump in np.linspace(10, max_val, steps//2):
#                     params = {}
#                     params['slope'] = slope
#                     params['margin'] = margin
#                     params['jump'] = jump
                    
#                     test_values.append(
#                         (x, y, fixed_paras, params, approx_fun))
    
#     elif approx_fun.nickname=='jump2':
#         test_values = []
#         max_val = max(y)
#         for margin in np.linspace(0, 3*max_delay//4, steps):
#             for jump in np.linspace(10, max_val, steps//2):
#                 params = {}
#                 params['margin'] = margin
#                 params['jump'] = jump
                
#                 test_values.append(
#                     (x, y, fixed_paras, params, approx_fun))
#     else:
#         raise Exception('Cost function approximator not implemented for', approx_fun.nickname)

#     return test_values

def fit_cost_curve(x, y, max_delay, fixed_paras={}, steps=8, approx_fun=None,
    default_parameters=None, compute_r_squared=False, algo_fit='L-BFGS-B'):

    """
    Can use 'L-BFGS-B' or 'Powell' for algo_fit, they seem to give the best results.
    """

    if default_parameters is None:
        # test_values = compute_test_values(x, y, max_delay, approx_fun,
        #                                 fixed_paras=fixed_paras, steps=steps)
        
        # #pool = Pool(num_cpu)
        # with Pool(max_workers=num_cpu) as pool:
        #     guesses = pool.map(fit_curve, test_values)
        # # best_initial_guess = np.array(test_values[np.argmin(guesses)][2:-1])
        # # Here I don't use the values method because I want to make sure that
        # # these values are in the same order as they appear in the archetype
        # # function
        # best_initial_guess = np.array([test_values[np.argmin(guesses)][3][k] for k in approx_fun.get_var_paras()])
        best_initial_guess = [10., y.mean()]
    else:
        best_initial_guess = [default_parameters[p] for p in approx_fun.paras if not p in approx_fun.fixed_paras]
    
    # TODO: read that from archetype function...
    if approx_fun.nickname=='jump':
        bounds = ([0., None], [0., None], [0., None])
    elif approx_fun.nickname=='jump2':
        bounds = ([0., None], [0., None])
    elif approx_fun.nickname=='jump3':
        bounds = ([0., None], [0., None], [0., None], [0., None])
    elif approx_fun.nickname=='double_jump':
        bounds = ([0., None], [0., None], [0., None], [0., None], [0., None])
    elif approx_fun.nickname=='double_jump2':
        bounds = ([0., None], [0., None], [0., None], [0., None])
    elif approx_fun.nickname=='double_jump3':
        bounds = ([0., None], [0., None], [0., None], [0., None], [0., None], [0., None])

    solution = minimize(obj_approx,
                        best_initial_guess,
                        args=(fixed_paras, x, y, approx_fun),
                        method=algo_fit,
                        options={'maxiter': 10000,
                                'xtol': 0.5,
                                'ftol': 0.01},
                        bounds=bounds)
    
    # # Keep these for debug
    # print ('BEST INITIAL GUESS:', best_initial_guess)
    # print ('SOLUTION:', solution)
    # print ('ARCHETYPE FUNCTION:', approx_fun.nickname)
    # print ('FIXED PARAS:', fixed_paras)
    # print ('BEST INITIAL GUESS:', best_initial_guess)
    # import matplotlib.pyplot as plt
    # plt.plot(x, y, label='data')
    # f = lambda x: approx_fun(x, **fixed_paras, **{k:solution.x[i] for i, k in enumerate(approx_fun.get_var_paras())})
    # plt.plot(x,
    #         [f(xx) for xx in x],
    #         label='fit')
    # plt.xlabel('Time')
    # plt.ylabel('Cost')
    # plt.legend()
    # plt.savefig('analysis/examples/example_fit.png')
    # plt.show()

    # raise Exception()

    if compute_r_squared:
        f = lambda x: approx_fun(x, **fixed_paras, **{k:solution.x[i] for i, k in enumerate(approx_fun.get_var_paras())})
        r2 = r_squared(y, np.array([f(xx) for xx in x]))
        return solution.x, r2
    else:
        return solution.x, None

def make_preference_fun(max_delay: float, cost_function, fixed_paras={}, approx_fun=None, n_points=1000,
    default_parameters=None, compute_r_squared=False, algo_fit='L-BFGS-B'):
    #delays = np.linspace(fixed_paras['eta'], fixed_paras['eta']+max_delay + 0.1, len(delay_cost_vect))#50)
    x = np.linspace(fixed_paras['eta'], fixed_paras['eta']+max_delay + 0.1, n_points)
    y = np.vectorize(cost_function)(x)
    result = fit_cost_curve(x,
                            y,
                            max_delay,
                            fixed_paras=fixed_paras,
                            approx_fun=approx_fun,
                            default_parameters=default_parameters,
                            compute_r_squared=compute_r_squared,
                            algo_fit=algo_fit)
    return result
    # plt.plot(delay_cost_vect)
    # plt.plot(approx_fun(delays, slope, margin_1, jump_1, margin_2, jump_2))
    # plt.show()


class FunctionApprox(ModelStructure):
    requirements = ['cost_f_true']

    def __init__(self, slots: List[Slot]=None, flights: List[fl.Flight]=None,
        cost_func_archetype=None, alternative_allocation_rule=False,
        default_parameters=None, compute_r_squared=False, algo_fit='L-BFGS-B'):
        self.cost_func_archetype = cost_func_archetype

        self.default_parameters = default_parameters

        self.compute_r_squared = compute_r_squared

        self.algo_fit = algo_fit

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
            paras, r2 = make_preference_fun(max_delay,
                                            flight.cost_f_true,
                                            default_parameters=self.default_parameters,
                                            fixed_paras=fixed_parameters,
                                            approx_fun=self.cost_func_archetype,
                                            compute_r_squared=self.compute_r_squared,
                                            algo_fit=self.algo_fit)
            for i, k in enumerate(self.cost_func_archetype.get_var_paras()):
                setattr(flight, k, paras[i])
            
            all_paras[flight.name] = {k:paras[i] for i, k in enumerate(self.cost_func_archetype.get_var_paras())}
            all_paras[flight.name]['r2'] = r2

        return all_paras