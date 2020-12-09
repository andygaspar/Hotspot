from typing import Callable, List, Union

from Programma.ModelStructure import modelStructure as mS
import xpress as xp
from Programma.ModelStructure.Airline import airline as air
from Programma.ModelStructure.Flight import flight as modFl
from Programma.ModelStructure.Solution import solution
from Programma.ModelStructure.Slot.slot import Slot

import numpy as np
import pandas as pd

import time


class MaxBenefitModel(mS.ModelStructure):

    def __init__(self, df_init: pd.DataFrame, costFun: Union[Callable, List[Callable]], model_name="Max Benefit"):

        self.airlineConstructor = air.Airline
        self.flightConstructor = modFl.Flight
        super().__init__(df_init=df_init, costFun=costFun)

        self.m = xp.problem()
        self.x = None

    def set_variables(self):
        flight: modFl.Flight
        airline: air.Airline
        self.x = np.array([[xp.var(vartype=xp.binary) for k in self.slots] for flight in self.flights])
        self.m.addVariable(self.x)

    def set_constraints(self):
        flight: modFl.Flight
        airline: air.Airline
        for flight in self.flights:
            self.m.addConstraint(
                xp.Sum(self.x[flight.slot.index, slot.index] for slot in flight.compatibleSlots) == 1
            )

        for slot in self.slots:
            self.m.addConstraint(
                xp.Sum(self.x[flight.slot.index, slot.index] for flight in self.flights) <= 1
            )

        for airline in self.airlines:
            self.m.addConstraint(
                xp.Sum(flight.costFun(flight, flight.slot) for flight in airline.flights) >= \
                xp.Sum(self.x[flight.slot.index, slot.index] * flight.costFun(flight, slot)
                       for flight in airline.flights for slot in self.slots)
            )

    def set_objective(self):
        flight: modFl.Flight
        self.m.setObjective(
            xp.Sum(self.x[flight.slot.index, slot.index] * flight.costFun(flight, slot)
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

        for flight in self.flights:
            if flight.eta > flight.newSlot.time:
                print("********************** danno *********************************",
                      flight, flight.eta, flight.newSlot.time)

    def assign_flights(self, sol):
        for flight in self.flights:
            for slot in self.slots:
                if self.m.getSolution(sol[flight.slot.index, slot.index]) > 0.5:
                    flight.newSlot = slot
