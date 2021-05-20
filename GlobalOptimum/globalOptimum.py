from typing import Callable, List, Union

from ModelStructure import modelStructure as mS
import xpress as xp
from ModelStructure.Airline import airline as air
from ModelStructure.Flight import flight as modFl
from ModelStructure.Solution import solution
from ModelStructure.Slot.slot import Slot

import numpy as np
import pandas as pd

import time

import xpress as xp

from Hotspot.ModelStructure import modelStructure as mS
from Hotspot.ModelStructure.Airline import airline as air
from Hotspot.ModelStructure.Flight import flight as modFl
from Hotspot.ModelStructure.Solution import solution
from Hotspot.ModelStructure.Slot.slot import Slot
from Hotspot.libs.uow_tool_belt import write_on_file as print_to_void


class GlobalOptimum(mS.ModelStructure):

    def __init__(self, slot_list: List[Slot], flight_list: List[Flight]):

        super().__init__(slot_list, flight_list)
        with print_to_void():
            self.m = xp.problem()
        self.x = None

    def set_variables(self):
        with print_to_void():
	    flight: Flight
	    airline: air.Airline
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

    def run(self, timing=False):
        with print_to_void():
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

    def assign_flights(self, sol):
	with print_to_void():
		for flight in self.flights:
		    for slot in self.slots:
		        if self.m.getSolution(sol[flight.index, slot.index]) > 0.5:
		            flight.newSlot = slot
