from typing import List

import numpy as np
import pandas as pd
from itertools import combinations
from Istop.AirlineAndFlight.istopFlight import IstopFlight
from Istop.Preferences import preference
from ModelStructure.Airline import airline as air
import matplotlib.pyplot as plt


class IstopAirline(air.Airline):

    @staticmethod
    def pairs(list_to_comb):
        comb = np.array(list(combinations(list_to_comb, 2)))
        offers = comb
        return offers

    @staticmethod
    def triplet(list_to_comb):
        return np.array(list(combinations(list_to_comb, 3)))

    def __init__(self, airline_name: str, airline_index: int, flights: List[IstopFlight]):

        super().__init__(airline_name, airline_index, flights)

        self.sum_priorities = None #sum(self.df["priority"])

        self.flight_pairs = self.pairs(self.flights)

        self.flight_triplets = self.triplet(self.flights)

        self.flights: List[IstopFlight]

        max_cost = max([cost for flight in self.flights for cost in flight.fitCostVect])

        for flight in self.flights:
            flight.standardisedVector = flight.costVect/max_cost

    # def set_preferences(self, priorityFunction):
    #     flight: IstopFlight
    #     for flight in self.flights:
    #         df_flight = self.df[self.df["flight"] == flight.name]
    #         flight.set_priority(df_flight["priority"].values[0])
    #         flight.set_preference(self.sum_priorities, priorityFunction)


