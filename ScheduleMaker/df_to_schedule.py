from typing import Union, List, Callable

from ModelStructure.Costs.costFunctionDict import CostFuns
from ModelStructure.Slot.slot import Slot
from ModelStructure.Airline import airline as air
from ModelStructure.Flight import flight as fl
from Istop.Preferences import preference
import numpy as np
import pandas as pd

cost_funs = CostFuns()


def make_flight(line, slot_times):
    flight_type = line["type"]
    slot_index = line["slot"]
    num = line["num"]
    flight_name = line["flight"]
    airline_name = line["airline"]
    eta = line["eta"]
    slot_time = line['fpfs']

    slot = Slot(slot_index, slot_time)

    try:
        margin = line["margins"]
    except:
        margin = None

    # ISTOP attributes  *************
    udpp_priority = line["priority"]

    cost_vect, delay_cost_vect = cost_funs.get_random_cost_vect(slot_times, eta)
    # print(delay_cost_vect)
    max_delay = slot_times[-1] - slot_times[1]
    slope, margin_1, jump_2, margin_2, jump_2 = preference.make_preference_fun(max_delay, delay_cost_vect)

    return fl.Flight(flight_type, slot, num, flight_name, airline_name,
                     eta, delay_cost_vect)


def make_flight_list(df: pd.DataFrame):
    flight_list = []
    slot_times = df.time.to_numpy()
    for i in range(df.shape[0]):
        line = df.iloc[i]
        flight_list.append(make_flight(line, slot_times))

    return flight_list
