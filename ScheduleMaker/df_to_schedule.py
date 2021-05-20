from typing import Union, List, Callable

from Hotspot.ModelStructure.Costs.costFunctionDict import CostFuns
from Hotspot.ModelStructure.Slot.slot import Slot
from Hotspot.ModelStructure import modelStructure
from Hotspot.ModelStructure.Flight import flight as fl
from Hotspot.Istop.Preferences import preference
import numpy as np
import pandas as pd

cost_funs = CostFuns()


def make_flight(line, slot_times):
    slot_index = line["slot"]
    flight_name = line["flight"]
    airline_name = line["airline"]
    eta = line["eta"]
    slot_time = line['fpfs']

    # slot = Slot(slot_index, slot_time)

    delay_cost_vect = cost_funs.get_random_cost_vect(slot_times, eta)

    return modelStructure.make_slot_and_flight(slot_time=slot_time, slot_index=slot_index, eta=eta,
                                               flight_name=flight_name, airline_name=airline_name,
                                               delay_cost_vect=delay_cost_vect)


def make_flight_list(df: pd.DataFrame):
    slot_list = []
    flight_list = []
    slot_times = df.time.to_numpy()
    for i in range(df.shape[0]):
        line = df.iloc[i]
        slot, flight = make_flight(line, slot_times)
        slot_list.append(slot)
        flight_list.append(flight)

    return slot_list, flight_list
