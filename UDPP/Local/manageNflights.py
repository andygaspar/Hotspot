from typing import Union, Callable, List

import numpy as np

from Hotspot.UDPP.AirlineAndFlightAndSlot.udppFlight import UDPPflight
from Hotspot.UDPP.AirlineAndFlightAndSlot.udppAirline import UDPPairline
from Hotspot.UDPP.AirlineAndFlightAndSlot.udppSlot import UDPPslot
from Hotspot.ModelStructure.modelStructure import ModelStructure
from Hotspot.ModelStructure.Solution import solution


def sort_flights_by_priority(flights):
    priorityList = [f.priorityNumber for f in flights]
    sorted_indexes = np.argsort(priorityList)  # np.flip(np.argsort(priorityList))
    return np.array([flights[i] for i in sorted_indexes])


def manage_Nflights(Nflights, localSlots):
    pfSorted = sort_flights_by_priority(Nflights)
    for i in range(len(Nflights)):
        pfSorted[i].newSlot = localSlots[i]
