from typing import Callable, List, Union

from ..ModelStructure import modelStructure as mS
import xpress as xp
xp.controls.outputlog = 0
from ..ModelStructure.Airline import airline as air
from ..ModelStructure.Flight.flight import Flight
from ..ModelStructure.Solution import solution
from ..ModelStructure.Slot.slot import Slot
from ..libs.uow_tool_belt.general_tools import write_on_file as print_to_void
from ..NNBound.SolversNNB.xpress_solver_NNB import XpressSolverNNB
from ..NNBound.SolversNNB.mip_solver_NNB import MipSolverNNB

from ..GlobalFuns.globalFuns import HiddenPrints

import numpy as np
import pandas as pd

import time


class NNBoundModel(mS.ModelStructure):
    requirements = ['delayCostVect', 'costVect']

    def __init__(self, slots: List[Slot] = None, flights: List[Flight] = None,
        xp_problem=None, alternative_allocation_rule=False):

        super().__init__(slots,
                        flights,
                        alternative_allocation_rule=alternative_allocation_rule)


    def run(self, timing=False, update_flights=False, max_time=2000):

        try:

            m = XpressSolverNNB(self, max_time)
            solution_vect = m.run(timing, update_flights)

        except:
            print("using MIP")
            m = MipSolverNNB(self, max_time)
            solution_vect = m.run(timing, update_flights)

        self.assign_flights(solution_vect)

        solution.make_solution(self)

        # for flight in self.flights:
        #     if flight.eta > flight.newSlot.time:
        #         print("********************** negative impact *********************************",
        #               flight, flight.eta, flight.newSlot.time)

    def assign_flights(self, sol):
        for flight in self.flights:
            for slot in self.slots:
                if self.m.getSolution(sol[flight.index, slot.index]) > 0.5:
                    flight.newSlot = slot

    def reset(self, slots: List[Slot] = None, flights: List[Flight] = None):

        super().__init__(slots, flights)

        with HiddenPrints():
            self.m.reset()

        self.x = None
