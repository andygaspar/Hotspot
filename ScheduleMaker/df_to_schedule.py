from typing import Union, List, Callable

from ModelStructure.Costs.costFunctionDict import CostFuns
from ModelStructure.Slot.slot import Slot
from ModelStructure import modelStructure
from ModelStructure.Flight import flight as fl
from Istop.Preferences import preference
import numpy as np
import pandas as pd

cost_funs = CostFuns()


def make_flight(line, slot_times, df_costs):
    slot_index = line["slot"]
    flight_name = line["flight"]
    airline_name = line["airline"]
    eta = line["eta"]
    slot_time = line['fpfs']

    # slot = Slot(slot_index, slot_time)
    if df_costs is None:

        delay_cost_vect = cost_funs.get_random_cost_vect(slot_times, eta)

    else:
        delay_cost_vect = df_costs[flight_name]

    return modelStructure.make_slot_and_flight(slot_time=slot_time, slot_index=slot_index, eta=eta,
                                               flight_name=flight_name, airline_name=airline_name,
                                               delay_cost_vect=delay_cost_vect)


def make_flight_list(df: pd.DataFrame, df_costs: pd.DataFrame = None):
    slot_list = []
    flight_list = []
    slot_times = df.time.to_numpy()
    for i in range(df.shape[0]):
        line = df.iloc[i]
        slot, flight = make_flight(line, slot_times, df_costs)
        slot_list.append(slot)
        flight_list.append(flight)

    return slot_list, flight_list
