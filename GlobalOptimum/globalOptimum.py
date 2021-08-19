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


class GlobalOptimum(mS.ModelStructure):
    requirements = ['delayCostVect', 'costVect']

    def __init__(self, slots: List[Slot] = None, flights: List[Flight] = None, delta_t=0.):
        super().__init__(slots=slots, flights=flights, delta_t=delta_t)
        with print_to_void():
            self.m = xp.problem()
        self.x = None

    def set_variables(self):
        flight: Flight
        airline: air.Airline
        with print_to_void():
            self.x = np.array([[xp.var(vartype=xp.binary) for _ in self.slots] for _ in self.flights])
            self.m.addVariable(self.x)

    def set_constraints(self):

        flight: Flight
        airline: air.Airline
        with print_to_void():
            for flight in self.flights:
                self.m.addConstraint(
                    xp.Sum(self.x[flight.index, slot.index] for slot in flight.compatibleSlots) == 1
                )

            for slot in self.slots:
                self.m.addConstraint(
                    xp.Sum(self.x[flight.index, slot.index] for flight in self.flights) <= 1
                )

    def set_objective(self):
        flight: Flight
        with print_to_void():
            self.m.setObjective(
                xp.Sum(self.x[flight.index, slot.index] * flight.cost_fun(slot)
                       for flight in self.flights for slot in self.slots)
            )

    def run(self, timing=False, update_flights=False):
        start = time.time()
        with print_to_void():
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
        with print_to_void():
            solution.make_solution(self)

        for flight in self.flights:
            if flight.eta > flight.newSlot.time:
                print("********************** negative impact *********************************",
                      flight, flight.eta, flight.newSlot.time)

        if update_flights:
            self.update_flights()

    def assign_flights(self, sol):
        #with print_to_void():
            for flight in self.flights:
                #print ('POUIC flight', flight)
                for slot in self.slots:
                    #print ('POUIC slot', slot, self.m.getSolution(sol[flight.index, slot.index]))
                    if self.m.getSolution(sol[flight.index, slot.index]) > 0.5:
                        #print ('POUIC match', flight)
                        flight.newSlot = slot

    def reset(self, slots, flights):
        super().__init__(slots, flights)

        with HiddenPrints():
            self.m.reset()

        self.x = None
