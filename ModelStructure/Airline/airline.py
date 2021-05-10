from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Callable
from itertools import combinations

from Hotspot.ModelStructure.Flight.flight import Flight
from Hotspot.ModelStructure.Slot.slot import Slot


class Airline:

    def make_airline_flight_list(self, slots: List[Slot], flight_ctor: Callable):
        flight_list = []
        for i in range(self.df.shape[0]):
            line = self.df.iloc[i]
            flight_list.append(flight_ctor(line, self, slots))

        return np.array(flight_list)

    def __init__(self, df_airline: pd.DataFrame, airline_index: int, slots: List[Slot], flight_ctor: Callable = Flight):

        self.df = df_airline

        self.name = self.df["airline"].unique()[0]

        self.index = airline_index

        self.numFlights = df_airline.shape[0]

        self.flights = self.make_airline_flight_list(slots, flight_ctor)

        self.AUslots = np.array([flight.slot for flight in self.flights])

        self.finalCosts = None

        for i in range(len(self.flights)):
            self.flights[i].set_local_num(i)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other: Airline):
        return self.index == other.index

