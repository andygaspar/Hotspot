from typing import Union, Callable, List

from .LocalOptimised.udppLocalOptGuroby import UDPPlocalOptGurobi
from ..GlobalFuns.globalFuns import HiddenPrints, preferences_from_flights
from ..ModelStructure.Airline.airline import Airline
from ..ModelStructure.modelStructure import ModelStructure
from ..ModelStructure.Slot.slot import Slot
from ..ModelStructure.Flight import flight as fl


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
                UDPPlocalOptGurobi(airline, self.slots)
            else:
                airline.flights[0].udppPriority = "N"
                airline.flights[0].udppPriorityNumber = 0
