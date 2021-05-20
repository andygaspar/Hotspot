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
            ((slot.time - flight.eta) * flight.cost*10 + flight.cost * 30),

            "jump": lambda flight, slot: 0 if slot.time - flight.eta < 0 else (slot.time - flight.eta) * flight.cost
            if (slot.time - flight.eta) < flight.margin else
            (slot.time - flight.eta) * flight.cost + flight.jump,

            "gate": lambda flight, slot: compute_gate_costs(flight, slot),
            
            "realistic": dict(zip(flights_dict_keys,
                                  [lambda t: dict_cost_func[flight_id](t, True) for flight_id in flights_dict_keys]))

        }

    def get_random_cost_vect(self, slot_times, eta):
        found = False
        fl_id = None
        while not found:
            fl_id = np.random.choice(get_flight_id_keys(), 1)[0]
            if dict_cost_func[fl_id](0, True) < 1:
                found = True
        # cost_vect = np.array([dict_cost_func[fl_id](t- eta, True)for t in slot_times])
        min_t = min(slot_times)
        delay_cost_vect = np.array([dict_cost_func[fl_id](t - min_t, True) for t in slot_times])
        return delay_cost_vect
