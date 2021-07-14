from typing import Union, Callable, List

import time

import pandas as pd
import numpy as np

from Hotspot.ModelStructure.Airline.airline import Airline
from Hotspot.ModelStructure.modelStructure import ModelStructure
from Hotspot.ModelStructure.Solution import solution
from Hotspot.ModelStructure.Slot.slot import Slot
from Hotspot.ModelStructure.Flight import flight as fl


def sort_flights_by_time(flights):
    time_list = [f.slot.time for f in flights]
    sorted_indexes = np.argsort(time_list)
    return np.array([flights[i] for i in sorted_indexes])

def get_first_compatible_light(slot, sorted_flights, slots):
    for flight in sorted_flights:
        if flight.eta <= slot.time:
            return flight

def udpp_merge(flights, slots):
    sorted_flights = list(sort_flights_by_time(flights))
    i = 0
    while len(sorted_flights) > 0:
        if sorted_flights[0].eta <= slots[i].time:
            sorted_flights[0].newSlot = slots[i]
            sorted_flights.pop(0)

        else:
            flight = get_first_compatible_light(slots[i], sorted_flights, slots)
            flight.newSlot = slots[i]
            sorted_flights.remove(flight)
        i += 1

def wrap_flight_udpp(flight):
   flight.UDPPLocalSlot = None
   flight.UDPPlocalSolution = None
   flight.test_slots = []


class UDPPMerge(ModelStructure):
    requirements = ['udppPriority', 'udppPriorityNumber', 'tna']
    def __init__(self, slot_list: List[Slot] = None, flights: List[fl.Flight] = None):

        if not flights is None:
            #udpp_flights = [UDPPflight(flight) for flight in flights if flight is not None]
            [wrap_flight_udpp(flight) for flight in flights if flight is not None]
            super().__init__(slot_list, flights, air_ctor=Airline)

    def run(self, optimised=True):
        airline: Airline

        udpp_merge(self.flights, self.slots)
        # print(time.time() - start)
        solution.make_solution(self, performance=hasattr(self, 'initialTotalCosts'))
        for flight in self.flights:
            if flight.eta > flight.newSlot.time:
                print("************** damage, some negative impact has occured****************",
                      flight, flight.eta, flight.newSlot.time)

    #def reset(self, df_init: pd.DataFrame, costFun: Union[Callable, List[Callable]]):
    # def reset(self, slot_list: List[Slot], flights: List[fl.Flight]):

    #     udpp_flights = [UDPPflight(flight) for flight in flights if flight is not None]
    #     super().__init__(slot_list, udpp_flights, air_ctor=Airline)
    #     #super().__init__(df_init=df_init, costFun=costFun, airline_ctor=UDPPairline)
