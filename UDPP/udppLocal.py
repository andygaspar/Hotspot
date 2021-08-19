import string
from typing import Union, Callable, List

import time

import pandas as pd

from ..GlobalFuns.globalFuns import HiddenPrints, preferences_from_flights
from ..ModelStructure.Airline.airline import Airline
from ..ModelStructure.modelStructure import ModelStructure
from ..UDPP.LocalOptimised.udppLocalOpt import UDPPlocalOpt
from ..UDPP.udppMerge import udpp_merge
from ..ModelStructure.Solution import solution
from ..ModelStructure.Slot.slot import Slot
from ..ModelStructure.Flight import flight as fl
from ..ModelStructure import modelStructure as ms
from ..UDPP.Local import local

class UDPPLocal(ModelStructure):
    requirements = ['costVect', 'delayCostVect']

    def __init__(self, slots: List[Slot] = None, flights: List[fl.Flight] = None):
        if not flights is None:
            super().__init__(slots, flights, air_ctor=Airline)

    def run(self):
        airline: Airline
        # start = time.time()
        self.compute_optimal_prioritisation()

        return preferences_from_flights(self.flights, paras=['udppPriority', 'udppPriorityNumber', 'tna'])

        # print(time.time() - start)

    def compute_optimal_prioritisation(self):
        airline: Airline
        for airline in self.airlines:
            if airline.numFlights > 1:
                #with HiddenPrints():
                UDPPlocalOpt(airline, self.slots)
            else:
                airline.flights[0].udppPriority = "N"
                airline.flights[0].udppPriorityNumber = 0

    #def reset(self, df_init: pd.DataFrame, costFun: Union[Callable, List[Callable]]):
    # def reset(self, slot_list: List[Slot], flights: List[fl.Flight]):

    #     udpp_flights = [UDPPflight(flight) for flight in flights if flight is not None]
    #     super().__init__(slot_list, udpp_flights, air_ctor=Airline)
    #     #super().__init__(df_init=df_init, costFun=costFun, airline_ctor=UDPPairline)
