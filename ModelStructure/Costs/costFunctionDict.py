import numpy as np
import pandas as pd
import dill as pickle
from pathlib import Path

dir_path = Path(__file__).resolve().parent

with open(dir_path / 'cost_functions_all.pck', 'rb') as dbfile:
    dict_cost_func = pickle.load(dbfile)
#dbfile.close()

flights_dict_keys = np.array(list(dict_cost_func.keys()))

at_gate = pd.read_csv(dir_path / 'costs_table_gate.csv', sep=" ")
delay_range = list(at_gate.columns[1:].astype(int))

def get_flight_id_keys():
    return flights_dict_keys

def get_interval(time):
    for i in range(len(delay_range) - 1):
        if delay_range[i] <= time < delay_range[i + 1]:
            return i

# def compute_gate_costs(flight, slot):
#     i = get_interval(slot.time)
#     y2 = at_gate[at_gate["flight"] == flight.type][str(delay_range[i + 1])].values[0]
#     y1 = at_gate[at_gate["flight"] == flight.type][str(delay_range[i])].values[0]
#     x2 = delay_range[i + 1]
#     x1 = delay_range[i]
#     return y1 + (slot.time - x1) * (y2 - y1) / (x2 - x1)

def compute_gate_costs(kind, time):
    i = get_interval(time)
    y2 = at_gate[at_gate["flight"] == kind][str(delay_range[i + 1])].values[0]
    y1 = at_gate[at_gate["flight"] == kind][str(delay_range[i])].values[0]
    x2 = delay_range[i + 1]
    x1 = delay_range[i]
    return y1 + (time - x1) * (y2 - y1) / (x2 - x1)


class ArchetypeCostFunction:
    paras = []
    fixed_paras = []
    nickname = 'ArchetypeCostFunction'

    def build_pure_lambda(self, paras):
        def f(time):
            return self(time, **paras)

        return f

    def single_computation(self, x, **paras):
        pass

    def get_var_paras(self):
        return [p for p in self.paras if not p in self.fixed_paras]

    def __call__(self, time, **paras):
        try:
            # vectorial mode
            _ = iter(time)
            results = [self.single_computation(t, **paras) for t in time]
            return results
        except TypeError:
            # scalar mode
            return self.single_computation(time, **paras)            


class LinearCostFunction(ArchetypeCostFunction):
    # BEWARE: the parameters should appear in the same order than in
    # single computation method.
    nickname = 'linear'
    paras = ['slope', 'eta']
    fixed_paras = ['eta']

    def single_computation(self, time, slope=None, eta=None):
        return slope * (time - eta)


class QuadraticCostFunction(ArchetypeCostFunction):
    # BEWARE: the parameters should appear in the same order than in
    # single computation method.
    nickname = 'quadratic'
    paras = ['slope', 'eta']
    fixed_paras = ['eta']

    def single_computation(self, time, slope=None, eta=None):
        return (slope * (time - eta) ** 2) / 2


class StepCostFunction(ArchetypeCostFunction):
    # BEWARE: the parameters should appear in the same order than in
    # single computation method.
    nickname = 'step'
    paras = ['eta', 'slope', 'margin']
    fixed_paras = ['eta']

    def single_computation(self, time, eta=None, slope=None, margin=None):
        return 0 if time - eta < 0 else (time - eta) * slope \
                    if (time - eta) < margin else \
                    ((time - eta) * slope*10 + slope * 30)


class JumpCostFunction(ArchetypeCostFunction):
    nickname = 'jump'
    # BEWARE: the parameters should appear in the same order than in
    # single computation method.
    paras = ['eta', 'slope', 'margin', 'jump']
    fixed_paras = ['eta']

    def single_computation(self, time, eta=None, slope=None, margin=None, jump=None):
        return 0 if time - eta < 0 else (time - eta) * slope \
                    if (time - eta) < margin else \
                    (time - eta) * slope + jump


class Jump2CostFunction(ArchetypeCostFunction):
    nickname = 'jump2'
    # BEWARE: the parameters should appear in the same order than in
    # single computation method.
    paras = ['eta', 'margin', 'jump']
    fixed_paras = ['eta']
    
    def single_computation(self, time, eta=None, margin=None, jump=None):
        slope = 0.1
        return 0 if time - eta < 0 else (time - eta) * slope \
                    if (time - eta) < margin else \
                    (time - eta) * slope + jump


class Jump3CostFunction(ArchetypeCostFunction):
    nickname = 'jump3'
    # BEWARE: the parameters should appear in the same order than in
    # single computation method.
    paras = ['eta', 'slope', 'margin', 'jump', 'offset']
    fixed_paras = ['eta']
    
    def single_computation(self, time, eta=None, slope=None, margin=None, jump=None, offset=None):
        return offset if time - eta < 0 else offset + (time - eta) * slope \
                    if (time - eta) < margin else \
                    offset + (time - eta) * slope + jump


class DoubleJumpCostFunction(ArchetypeCostFunction):
    # BEWARE: the parameters should appear in the same order than in
    # single computation method.
    nickname = 'double_jump'
    paras = ['eta', 'slope', 'margin1', 'jump1', 'margin2', 'jump2']
    fixed_paras = ['eta']

    def single_computation(self, time, eta=None, slope=None, margin1=None,
        jump1=None, margin2=None, jump2=None):
        if time - eta < 0:
            return 0
        elif 0 <= (time - eta) < margin1:
            return (time - eta) * slope
        elif margin1 <= (time - eta) < margin2:
            return (time - eta) * slope + jump1
        elif (time - eta) >= margin2:
            return (time - eta) * slope + jump2


class DoubleJump2CostFunction(ArchetypeCostFunction):
    # BEWARE: the parameters should appear in the same order than in
    # single computation method.
    nickname = 'double_jump2'
    paras = ['eta', 'margin1', 'jump1', 'margin2', 'jump2']
    fixed_paras = ['eta']

    def single_computation(self, time, eta=None, margin1=None,
        jump1=None, margin2=None, jump2=None):
        if time - eta < 0:
            return 0
        elif 0 <= (time - eta) < margin1:
            return 0
        elif margin1 <= (time - eta) < margin2:
            return jump1
        elif (time - eta) >= margin2:
            return jump2

class DoubleJump3CostFunction(ArchetypeCostFunction):
    # BEWARE: the parameters should appear in the same order than in
    # single computation method.
    nickname = 'double_jump3'
    paras = ['eta', 'slope', 'margin1', 'jump1', 'margin2', 'jump2', 'offset']
    fixed_paras = ['eta']

    def single_computation(self, time, eta=None, slope=None, margin1=None,
        jump1=None, margin2=None, jump2=None, offset=None):
        if time - eta < 0:
            return offset
        elif 0 <= (time - eta) < margin1:
            return (time - eta) * slope + offset
        elif margin1 <= (time - eta) < margin2:
            return (time - eta) * slope + jump1 + offset
        elif (time - eta) >= margin2:
            return (time - eta) * slope + jump2 + offset


class GateCostFunction(ArchetypeCostFunction):
    # BEWARE: the parameters should appear in the same order than in
    # single computation method.
    nickname = 'gate'
    paras = ['kind']
    fixed_paras = ['kind']

    def single_computation(self, time, kind=None):
        return compute_gate_costs(kind, time)


class RealisticCostFunction(ArchetypeCostFunction):
    # TODO: fix. Not sure what this is.
    nickname = 'realistic'
    paras = []

    def single_computation(self, time):
        dict(zip(flights_dict_keys,
                    [lambda t: dict_cost_func[flight_id](t, True) for flight_id in flights_dict_keys]))

# Register the functions
archetypes_cost_functions = {LinearCostFunction.nickname:LinearCostFunction,
                            QuadraticCostFunction.nickname:QuadraticCostFunction,
                            StepCostFunction.nickname:StepCostFunction,
                            JumpCostFunction.nickname:JumpCostFunction,
                            Jump2CostFunction.nickname:Jump2CostFunction,
                            Jump3CostFunction.nickname:Jump3CostFunction,
                            DoubleJumpCostFunction.nickname:DoubleJumpCostFunction,
                            DoubleJump2CostFunction.nickname:DoubleJump2CostFunction,
                            DoubleJump3CostFunction.nickname:DoubleJump3CostFunction,
                            GateCostFunction.nickname:GateCostFunction,
                            RealisticCostFunction.nickname:RealisticCostFunction,
                            }

# class CostFuns:

#     def __init__(self):
#         self.costFun = {

#             "linear": lambda flight, slot: flight.cost * (slot.time - flight.eta),

#             "quadratic": lambda flight, slot: (flight.cost * (slot.time - flight.eta) ** 2) / 2,

#             "step": lambda flight, slot: 0 if slot.time - flight.eta < 0 else (slot.time - flight.eta) * flight.cost
#             if (slot.time - flight.eta) < flight.margin else
#             ((slot.time - flight.eta) * flight.cost*10 + flight.cost * 30),

#             "jump": lambda flight, slot: 0 if slot.time - flight.eta < 0 else (slot.time - flight.eta) * flight.slope
#             if (slot.time - flight.eta) < flight.margin1 else
#             (slot.time - flight.eta) * flight.slope + flight.jump1,

#             "gate": lambda flight, slot: compute_gate_costs(flight, slot),
            
#             "realistic": dict(zip(flights_dict_keys,
#                                   [lambda t: dict_cost_func[flight_id](t, True) for flight_id in flights_dict_keys]))

#         }

#     def get_random_cost_vect(self, slot_times, eta):
#         found = False
#         fl_id = None
#         while not found:
#             fl_id = np.random.choice(get_flight_id_keys(), 1)[0]
#             if dict_cost_func[fl_id](0, True) < 1:
#                 found = True
#         # cost_vect = np.array([dict_cost_func[fl_id](t- eta, True)for t in slot_times])
#         min_t = min(slot_times)
#         delay_cost_vect = np.array([dict_cost_func[fl_id](t - min_t, True) for t in slot_times])
#         return delay_cost_vect
