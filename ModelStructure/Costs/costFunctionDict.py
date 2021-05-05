import numpy as np
import pandas as pd
import dill as pickle

with open('ModelStructure/Costs/cost_functions_all.pck', 'rb') as dbfile:
    dict_cost_func = pickle.load(dbfile)
dbfile.close()

flights_dict_keys = np.array(list(dict_cost_func.keys()))

at_gate = pd.read_csv("ModelStructure/Costs/costs_table_gate.csv", sep=" ")
delay_range = list(at_gate.columns[1:].astype(int))


def get_interval(time):
    for i in range(len(delay_range) - 1):
        if delay_range[i] <= time < delay_range[i + 1]:
            return i


def compute_gate_costs(flight, slot):
    i = get_interval(slot.time)
    y2 = at_gate[at_gate["flight"] == flight.type][str(delay_range[i + 1])].values[0]
    y1 = at_gate[at_gate["flight"] == flight.type][str(delay_range[i])].values[0]
    x2 = delay_range[i + 1]
    x1 = delay_range[i]
    return y1 + (slot.time - x1) * (y2 - y1) / (x2 - x1)


class CostFuns:

    def __init__(self):
        self.costFun = {

            "linear": lambda flight, slot: flight.cost * (slot.time - flight.eta),

            "quadratic": lambda flight, slot: (flight.cost * (slot.time - flight.eta) ** 2) / 2,

            "step": lambda flight, slot: 0 if slot.time - flight.eta < 0 else (slot.time - flight.eta) * flight.cost
            if (slot.time - flight.eta) < flight.margin else
            ((slot.time - flight.eta) * flight.cost * 10 + flight.cost * 30),

            "gate": lambda flight, slot: compute_gate_costs(flight, slot),

            "realistic": dict(zip(flights_dict_keys,
                                  [lambda t: dict_cost_func[flight_id](t*2, True) for flight_id in flights_dict_keys]))

        }

    def get_random_real_cost_fun(self):
        flight_id = np.random.choice(flights_dict_keys, 1)[0]
        return self.costFun["realistic"][flight_id]
