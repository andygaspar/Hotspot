from typing import Union, Callable, List

import time

import pandas as pd
import numpy as np

from ..ModelStructure.Airline.airline import Airline
from ..ModelStructure.modelStructure import ModelStructure
from ..ModelStructure.Solution import solution
from ..ModelStructure.Slot.slot import Slot
from ..ModelStructure.Flight import flight as fl
from ..UDPP.Local import local


def sort_flights_by_time(flights):
    time_list = [f.newSlot.time for f in flights]
    sorted_indexes = np.argsort(time_list)
    return np.array([flights[i] for i in sorted_indexes])

def get_first_compatible_flight(slot, sorted_flights, slots):
    for flight in sorted_flights:
        if slot in flight.compatibleSlots:
            return flight

def udpp_merge(flights, slots, hfes=5):
    sorted_flights = list(sort_flights_by_time(flights))
    i = 0
    while len(sorted_flights) > 0:
        if slots[i] in sorted_flights[0].compatibleSlots:# or slots[i].time >= sorted_flights[0].eta - hfes:#sorted_flights[0]#slots[i].time >= sorted_flights[0].eta - delta_t :
            sorted_flights[0].newSlot = slots[i]
            sorted_flights.pop(0)

        else:
            flight = get_first_compatible_flight(slots[i], sorted_flights, slots)
            if flight is not None:
                # This is for the case where some slots will be empty.
                flight.newSlot = slots[i]
                sorted_flights.remove(flight)
        i += 1


class UDPPMerge(ModelStructure):
    requirements = ['udppPriority', 'udppPriorityNumber', 'tna']
    def __init__(self, slots: List[Slot] = None, flights: List[fl.Flight] = None,
        alternative_allocation_rule=False):
        if not flights is None:
            super().__init__(slots,
                            flights,
                            alternative_allocation_rule=alternative_allocation_rule,
                            air_ctor=Airline)

    def run(self, optimised=True):
        airline: Airline
 
        for airline in self.airlines:
            if airline.numFlights > 1:
                local.udpp_local(airline, self.slots)
            else:
                airline.flights[0].newSlot = airline.flights[0].slot

        udpp_merge(self.flights, self.slots)#, delta_t=self.delta_t)

        # print(time.time() - start)
        solution.make_solution(self, performance=hasattr(self, 'initialTotalCosts'))
        # for flight in self.flights:
        #     if flight.eta > flight.newSlot.time:
        #         print("************** damage, some negative impact has occured****************",
        #               flight, flight.eta, flight.newSlot.time)

    #def reset(self, df_init: pd.DataFrame, costFun: Union[Callable, List[Callable]]):
    # def reset(self, slot_list: List[Slot], flights: List[fl.Flight]):

    #     udpp_flights = [UDPPflight(flight) for flight in flights if flight is not None]
    #     super().__init__(slot_list, udpp_flights, air_ctor=Airline)
    #     #super().__init__(df_init=df_init, costFun=costFun, airline_ctor=UDPPairline)
