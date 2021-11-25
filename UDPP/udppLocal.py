import string
from typing import Union, Callable, List

import time

import pandas as pd

from .LocalOptimised.udppLocalOptGuroby import UDPPlocalOptGurobi
from .LocalOptimised.udppLocalOptMIP import UDPPlocalOptMIP
from ..GlobalFuns.globalFuns import HiddenPrints, preferences_from_flights
from ..ModelStructure.Airline.airline import Airline
from ..ModelStructure.modelStructure import ModelStructure
# from ..UDPP.LocalOptimised.udppLocalOptXP import UDPPlocalOptXP
from ..UDPP.udppMerge import udpp_merge
from ..ModelStructure.Solution import solution
from ..ModelStructure.Slot.slot import Slot
from ..ModelStructure.Flight import flight as fl
from ..ModelStructure import modelStructure as ms
from ..UDPP.Local import local

class UDPPLocal(ModelStructure):
    requirements = ['costVect', 'delayCostVect']

    def __init__(self, slots: List[Slot] = None, flights: List[fl.Flight] = None,
        alternative_allocation_rule=False):

        
        if not flights is None:
            super().__init__(slots,
                            flights,
                            air_ctor=Airline,
                            alternative_allocation_rule=alternative_allocation_rule)

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
                try:
                    UDPPlocalOptGurobi(airline, self.slots)
                    #print ('Using Gurobi')
                    # with HiddenPrints():
                    #     UDPPlocalOptXP(airline, self.slots)
                    #     #print ('Using XPress in UDPP Local')
                except Exception as e:
                    # try:
                    #     UDPPlocalOptGurobi(airline, self.slots)
                    #     #print ('Using Gurobi in UDPP Local')
                    # except:
                    UDPPlocalOptMIP(airline, self.slots)
                    # print ('Using MIP (exception from Gurobi:', e, ')')
            else:
                airline.flights[0].udppPriority = "N"
                airline.flights[0].udppPriorityNumber = 0

    #def reset(self, df_init: pd.DataFrame, costFun: Union[Callable, List[Callable]]):
    # def reset(self, slot_list: List[Slot], flights: List[fl.Flight]):

    #     udpp_flights = [UDPPflight(flight) for flight in flights if flight is not None]
    #     super().__init__(slot_list, udpp_flights, air_ctor=Airline)
    #     #super().__init__(df_init=df_init, costFun=costFun, airline_ctor=UDPPairline)
