from typing import Callable, List, Union

from ..ModelStructure import modelStructure as mS
import xpress as xp
xp.controls.outputlog = 0
from ..ModelStructure.Airline import airline as air
from ..ModelStructure.Flight.flight import Flight
from ..ModelStructure.Solution import solution
from ..ModelStructure.Slot.slot import Slot
from ..libs.uow_tool_belt.general_tools import write_on_file as print_to_void

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
        if xp_problem is None:
            with print_to_void():
                self.m = xp.problem()
        else:
            self.m = xp_problem
        self.x = None

    def set_variables(self):
        flight: Flight
        airline: air.Airline
        self.x = np.array([[xp.var(vartype=xp.binary) for _ in self.slots] for _ in self.flights])
        self.m.addVariable(self.x)

    def set_constraints(self):
        flight: Flight
        airline: air.Airline
        for flight in self.flights:
            self.m.addConstraint(
                xp.Sum(self.x[flight.index, slot.index] for slot in flight.compatibleSlots) == 1
            )

        for slot in self.slots:
            self.m.addConstraint(
                xp.Sum(self.x[flight.index, slot.index] for flight in self.flights) <= 1
            )

        for airline in self.airlines:
            self.m.addConstraint(
                xp.Sum(flight.cost_fun(flight.slot) for flight in airline.flights) >= \
                xp.Sum(self.x[flight.index, slot.index] * flight.cost_fun(slot)
                       for flight in airline.flights for slot in self.slots)
            )

    def set_objective(self):
        flight: Flight
        self.m.setObjective(
            xp.Sum(self.x[flight.index, slot.index] * flight.cost_fun(slot)
                   for flight in self.flights for slot in self.slots)
        )

    def run(self, timing=False):

        start = time.time()
        self.set_variables()
        self.set_constraints()
        end = time.time() - start
        if timing:
            print("Variables and constraints setting time ", end)

        self.set_objective()

        start = time.time()
        self.m.solve()
        end = time.time() - start
        if timing:
            print("Simplex time ", end)

        self.assign_flights(self.x)

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
