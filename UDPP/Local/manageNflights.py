from typing import Union, Callable, List

from Programma.UDPP.AirlineAndFlightAndSlot.udppFlight import UDPPflight
from Programma.UDPP.AirlineAndFlightAndSlot.udppAirline import UDPPairline
from Programma.UDPP.AirlineAndFlightAndSlot.udppSlot import UDPPslot
from Programma.ModelStructure.modelStructure import ModelStructure
from Programma.ModelStructure.Solution import solution
import numpy as np


def sort_flights_by_priority(flights):
    priorityList = [f.priorityNumber for f in flights]
    sorted_indexes = np.argsort(priorityList)  # np.flip(np.argsort(priorityList))
    return np.array([flights[i] for i in sorted_indexes])


def manage_Nflights(Nflights, localSlots):
    pfSorted = sort_flights_by_priority(Nflights)
    for i in range(len(Nflights)):
        pfSorted[i].newSlot = localSlots[i]
