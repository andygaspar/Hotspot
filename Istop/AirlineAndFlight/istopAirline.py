from typing import List

import numpy as np
import pandas as pd
from itertools import combinations
from Istop.AirlineAndFlight.istopFlight import IstopFlight
from ModelStructure.Airline import airline as air


class IstopAirline(air.Airline):

    @staticmethod
    def pairs(list_to_comb):
        comb = np.array(list(combinations(list_to_comb, 2)))
        offers = comb #[pair for pair in comb if np.abs(pair[0].priority-pair[1].priority) > 0.2]
        return offers

    @staticmethod
    def triplet(list_to_comb):
        return np.array(list(combinations(list_to_comb, 3)))

    def __init__(self, airline_name: str, airline_index: int, flights: List[IstopFlight]):

        super().__init__(airline_name, airline_index, flights)

        self.sum_priorities = None #sum(self.df["priority"])

        self.flight_pairs = self.pairs(self.flights)

        self.flight_triplets = self.triplet(self.flights)

    def set_preferences(self, priorityFunction):
        flight: IstopFlight
        for flight in self.flights:
            df_flight = self.df[self.df["flight"] == flight.name]
            flight.set_priority(df_flight["priority"].values[0])
            flight.set_preference(self.sum_priorities, priorityFunction)
