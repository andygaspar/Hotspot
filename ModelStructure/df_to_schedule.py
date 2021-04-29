from typing import Union, List, Callable
from ModelStructure.Slot.slot import Slot
from ModelStructure.Airline import airline as air
from ModelStructure.Flight import flight as fl
import numpy as np
import pandas as pd


def make_flight(line, cost_fun):

    flight_type = line["type"]
    slot_index = line["slot"]
    num = line["num"]
    flight_name = line["flight"]
    airline_name = line["airline"]
    eta = line["eta"]
    fpfs = line['fpfs']
    slot_time = fpfs

    slot = Slot(slot_index, slot_time)

    try:
        margin = line["margins"]
    except:
        margin = None

    # ISTOP attributes  *************
    udpp_priority = line["priority"]

    return fl.Flight(flight_type, slot, num, flight_name, airline_name,
                     eta, fpfs, cost_fun, udpp_priority, margin)


def make_flight_list(df: pd.DataFrame, cost_fun):

    flight_list = []
    for i in range(df.shape[0]):
        line = df.iloc[i]
        flight_list.append(make_flight(line, cost_fun))

    return flight_list