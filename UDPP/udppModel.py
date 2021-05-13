import string
from typing import Union, Callable, List

import pandas as pd

from GlobalFuns.globalFuns import HiddenPrints
from ModelStructure.Airline.airline import Airline
from ModelStructure.modelStructure import ModelStructure
from UDPP.LocalOptimised.udppLocalOpt import UDPPlocalOpt
from UDPP.udppMerge import udpp_merge
from ModelStructure.Solution import solution
from UDPP.UDPPflight.udppFlight import UDPPflight
from ModelStructure.Slot.slot import Slot
from ModelStructure.Flight import flight as fl
import time

import ModelStructure.modelStructure as ms
from UDPP.Local import local

class UDPPmodel(ModelStructure):

    def __init__(self, slot_list: List[Slot], flights: List[fl.Flight]):

        udpp_flights = [UDPPflight(flight) for flight in flights]
        super().__init__(slot_list, udpp_flights, air_ctor=Airline)

    def run(self, optimised=True):
        airline: Airline
        start = time.time()
        for airline in self.airlines:
            if airline.numFlights > 1:
                if optimised:
                    with HiddenPrints():
                        UDPPlocalOpt(airline, self.slots)
                else:
                    local.udpp_local(airline, self.slots)
            else:
                airline.flights[0].newSlot = airline.flights[0].slot

        udpp_merge(self.flights, self.slots)
        # print(time.time() - start)
        solution.make_solution(self)
        for flight in self.flights:
            if flight.eta > flight.newSlot.time:
                print("************** damage, some negative impact has occured****************",
                      flight, flight.eta, flight.newSlot.time)

    def compute_optimal_prioritisation(self):
        airline: Airline
        for airline in self.airlines:
            if airline.numFlights > 1:
                with HiddenPrints():
                    UDPPlocalOpt(airline, self.slots)
            else:
                airline.flights[0].udppPriority = "N"
                airline.flights[0].udppPriorityNumber = 0



