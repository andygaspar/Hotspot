from typing import Callable, List, Union

import numpy as np
import pandas as pd
import mip

import time



from ...GlobalFuns.globalFuns import HiddenPrints
from ...ModelStructure import modelStructure as mS
from ...ModelStructure.Airline import airline as air
from ...ModelStructure.Flight.flight import Flight
from ...ModelStructure.Solution import solution
from ...ModelStructure.Slot.slot import Slot
from ...libs.uow_tool_belt.general_tools import write_on_file as print_to_void


class MipSolverGO(mS.ModelStructure):

    def __init__(self, model, max_time):
        self.m = mip.Model()
        self.m.verbose = False
        self.m.threads = -1
        self.maxTime = max_time
        self.x = None

        self.flights = model.flights
        self.airlines = model.airlines
        self.slots = model.slots

    def set_variables(self):
        flight: Flight
        airline: air.Airline
        with print_to_void():
            self.x = np.array([[self.m.add_var(var_type=mip.BINARY) for _ in self.slots] for _ in self.slots])

    def set_constraints(self):

        flight: Flight
        airline: air.Airline
        for flight in self.flights:

                self.m += mip.xsum(self.x[flight.index, slot.index] for slot in flight.compatibleSlots) == 1


        for slot in self.slots:
            self.m += mip.xsum(self.x[flight.index, slot.index] for flight in self.flights) <= 1


    def set_objective(self):
        flight: Flight

        self.m.objective = mip.minimize(
            mip.xsum(self.x[flight.index, slot.index] * flight.cost_fun(slot)
                   for flight in self.flights for slot in self.slots)
            )

    def run(self, timing=False, update_flights=False):
        start = time.time()

        self.set_variables()
        self.set_constraints()
        end = time.time() - start
        if timing:
            print("Variables and constraints setting time ", end)

        self.set_objective()

        start = time.time()
        self.m.optimize(max_seconds=self.maxTime)
        end = time.time() - start
        if timing:
            print("Simplex time ", end)

        with print_to_void():
            solution.make_solution(self)

            for flight in self.flights:
                if flight.eta > flight.newSlot.time:
                    print("********************** negative impact *********************************",
                          flight, flight.eta, flight.newSlot.time)

        if update_flights:
            self.update_flights()

        return np.array([[el.x for el in col] for col in self.x])

    def reset(self):
       pass
