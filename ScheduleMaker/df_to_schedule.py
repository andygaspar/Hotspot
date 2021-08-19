from typing import Union, List, Callable

from ..ModelStructure.Slot.slot import Slot
from ..ModelStructure import modelStructure
from ..ModelStructure.Flight import flight as fl
import numpy as np
import pandas as pd

def make_flight(line, slot_times):
    slot_index = line["slot"]
    flight_name = line["flight"]
    airline_name = line["airline"]
    eta = line["eta"]
    slot_time = line['fpfs']
    margin = line['margins']
    jump = line['jump']
    slope = line['cost']

    class DummyFlight:
        pass

    class DummySlot:
        def __init__(self, slot_time):
            self.time = slot_time

    flight = DummyFlight()

    flight.margin1 = margin
    flight.jump1 = jump
    flight.slope = slope
    flight.eta = eta

    slots = [DummySlot(st) for st in slot_times]

    cost_vect = [CostFuns().costFun['jump'](flight, slot) for slot in slots]

    # slot = Slot(slot_index, slot_time)

    #delay_cost_vect = cost_funs.get_random_cost_vect(slot_times, eta)

    return modelStructure.make_slot_and_flight(slot_time=slot_time,
                                                slot_index=slot_index,
                                                eta=eta,
                                                flight_name=flight_name,
                                                airline_name=airline_name,
                                                cost_vect=cost_vect)


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
