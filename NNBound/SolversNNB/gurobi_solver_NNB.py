from typing import Callable, List, Union

from gurobipy import Model, GRB, quicksum, Env

from ...GlobalFuns.globalFuns import HiddenPrints
from ...ModelStructure import modelStructure as mS
from ...ModelStructure.Airline import airline as air
from ...ModelStructure.Flight.flight import Flight
from ...ModelStructure.Solution import solution
from ...ModelStructure.Slot.slot import Slot
# from ...libs.uow_tool_belt.general_tools import write_on_file as print_to_void

import numpy as np
import pandas as pd

import time


def stop(model, where):
    if where == GRB.Callback.MIP:
        run_time = model.cbGet(GRB.Callback.RUNTIME)

        if run_time > model._time_limit:
            print("stop at", run_time)
            model.terminate()


class NNBoundGurobi:

    def __init__(self, model: mS.ModelStructure):

        self.flights = model.flights
        self.airlines = model.airlines
        self.slots = model.slots
        self.numFlights = model.numFlights

        self.m = Model('CVRP')
        # self.m.setParam('Method', 2) ###################testare == 2 !!!!!!!!!!!!111c
        self.m.modelSense = GRB.MINIMIZE
        self.x = None

    def set_variables(self):
        flight: Flight
        airline: air.Airline
        self.x = self.m.addVars([(i, j) for i in range(self.numFlights) for j in range(len(self.slots))],
                                vtype=GRB.BINARY)

    def set_constraints(self):
        flight: Flight
        airline: air.Airline
        for flight in self.flights:
            self.m.addConstr(
                quicksum(self.x[flight.index, slot.index] for slot in flight.compatibleSlots) == 1
            )

        for slot in self.slots:
            self.m.addConstr(
                quicksum(self.x[flight.index, slot.index] for flight in self.flights) <= 1
            )

        for airline in self.airlines:
            self.m.addConstr(
                quicksum(flight.cost_fun(flight.slot) for flight in airline.flights) >= \
                quicksum(self.x[flight.index, slot.index] * flight.cost_fun(slot)
                         for flight in airline.flights for slot in self.slots)
            )

    def set_objective(self):
        flight: Flight
        self.m.setObjective(
            quicksum(self.x[flight.index, slot.index] * flight.cost_fun(slot)
                     for flight in self.flights for slot in self.slots)
        )

    def run(self, timing=False, verbose=False, time_limit=60):

        self.m._time_limit = time_limit
        if not verbose:
            self.m.setParam('OutputFlag', 0)

        start = time.time()
        self.set_variables()
        self.set_constraints()
        end = time.time() - start
        if timing:
            print("Variables and constraints setting time ", end)

        self.set_objective()

        start = time.time()
        self.m.optimize(stop)
        # self.m.printStats()

        end = time.time() - start
        if timing:
            print("Simplex time ", end)

        return self.get_sol_array()

    def assign_flights(self, sol):
        for flight in self.flights:
            for slot in self.slots:
                if sol[flight.index, slot.index].x > 0.5:
                    flight.newSlot = slot

    def get_sol_array(self):
        solution = np.zeros((len(self.flights), len(self.slots)))
        for flight in self.flights:
            for slot in self.slots:
                if self.x[flight.index, slot.index].x > 0.5:
                    solution[flight.index, slot.index] = 1
        return solution
