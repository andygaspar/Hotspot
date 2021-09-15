from typing import Callable, List, Union

import numpy as np
import pandas as pd

import time

import xpress as xp


from ..GlobalFuns.globalFuns import HiddenPrints
from ..ModelStructure import modelStructure as mS
from ..ModelStructure.Airline import airline as air
from ..ModelStructure.Flight.flight import Flight
from ..ModelStructure.Solution import solution
from ..ModelStructure.Slot.slot import Slot
from ..libs.uow_tool_belt.general_tools import write_on_file as print_to_void
from ..GlobalOptimum.SolversGO.xpress_solver_go import XpressSolverGO
from ..GlobalOptimum.SolversGO.mip_solver_go import MipSolverGO


class GlobalOptimum(mS.ModelStructure):
    requirements = ['delayCostVect', 'costVect']

    def __init__(self, slots: List[Slot] = None, flights: List[Flight] = None, alternative_allocation_rule=0.):
        super().__init__(slots=slots, flights=flights, alternative_allocation_rule=alternative_allocation_rule)




    def run(self, timing=False, update_flights=False, max_time=2000):
        try:

            m = XpressSolverGO(self, max_time)
            solution_vect = m.run(timing, update_flights)

        except:
            print("using MIP")
            m = MipSolverGO(self, max_time)


        self.assign_flights(solution_vect)
        with print_to_void():
            solution.make_solution(self)

            for flight in self.flights:
                if flight.eta > flight.newSlot.time:
                    print("********************** negative impact *********************************",
                          flight, flight.eta, flight.newSlot.time)

        if update_flights:
            self.update_flights()



    def assign_flights(self, solution_vect):
        with print_to_void():
            for flight in self.flights:
                #print ('POUIC flight', flight)
                for slot in self.slots:
                    #print ('POUIC slot', slot, self.m.getSolution(sol[flight.index, slot.index]))
                    if solution_vect[flight.index, slot.index] > 0.5:
                        #print ('POUIC match', flight)
                        flight.newSlot = slot

    # def reset(self, slots, flights):
    #     super().__init__(slots, flights)
