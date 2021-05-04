from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Callable
from itertools import combinations
from ModelStructure.Flight.flight import Flight
from ModelStructure.Slot.slot import Slot


class Airline:

    def __init__(self, airline_name: str, airline_index: int, flights: List[Flight]):

        self.name = airline_name

        self.index = airline_index

        self.numFlights = len(flights)

        self.flights = flights

        self.AUslots = np.array([flight.slot for flight in self.flights])

        self.finalCosts = None

        for i in range(len(self.flights)):
            self.flights[i].set_local_num(i)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if type(other) == str:
            return self.name == other
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)
