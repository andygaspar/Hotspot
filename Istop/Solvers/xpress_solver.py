import sys
import time
from typing import List

import xpress as xp
xp.controls.outputlog = 0
import numpy as np

from Hotspot.ModelStructure.Flight.flight import Flight


class XpressSolver:

    def __init__(self, model, max_time):

        self.m = xp.problem()
        self.m.controls.maxtime = max_time
        self.flights = model.flights
        self.airlines = model.airlines
        self.slots = model.slots

        self.matches = model.matches
        self.emptySlots = model.emptySlots
        self.flights_in_matches = model.flights_in_matches

        self.f_in_matched = model.f_in_matched
        self.get_match_for_flight = model.get_match_for_flight
        self.check_and_set_matches = model.check_and_set_matches

        self.epsilon = sys.float_info.min

        self.x = None
        self.c = None

    def set_variables(self):
        self.x = np.array([[xp.var(vartype=xp.binary) for _ in self.slots] for _ in self.slots], dtype=xp.npvar)

        self.c = np.array([xp.var(vartype=xp.binary) for _ in self.matches], dtype=xp.npvar)

        self.m.addVariable(self.x, self.c)

    def set_constraints(self):

        self.flights: List[Flight]

        for i in self.emptySlots:
            for j in self.slots:
                self.m.addConstraint(self.x[i, j] == 0)

        for flight in self.flights:
            if not self.f_in_matched(flight):
                self.m.addConstraint(self.x[flight.index, flight.index] == 1)
            else:
                self.m.addConstraint(xp.Sum(self.x[flight.index, j.index] for j in flight.compatibleSlots) == 1)

        for j in self.slots:
            self.m.addConstraint(xp.Sum(self.x[i.index, j.index] for i in self.slots) <= 1)

        for flight in self.flights:
            for j in flight.notCompatibleSlots:
                self.m.addConstraint(self.x[flight.index, j.index] == 0)

        for flight in self.flights_in_matches:
            self.m.addConstraint(
                xp.Sum(self.x[flight.index, slot.index]
                       for slot in self.slots if slot != flight.slot) \
                <= xp.Sum([self.c[j] for j in self.get_match_for_flight(flight)]))

            self.m.addConstraint(xp.Sum([self.c[j] for j in self.get_match_for_flight(flight)]) <= 1)

        k = 0
        for match in self.matches:
            flights = [flight for pair in match for flight in pair]
            self.m.addConstraint(xp.Sum(xp.Sum(self.x[i.index, j.index] for i in pair for j in flights)
                                        for pair in match) >= (self.c[k]) * len(flights))

            for pair in match:
                self.m.addConstraint(
                    xp.Sum(self.x[i.index, j.index] * i.fitCostVect[j.index] for i in pair for j in
                           flights) -
                    (1 - self.c[k]) * 10000000 \
                    <= xp.Sum(self.x[i.index, j.index] * i.fitCostVect[i.index] for i in pair for j in
                              flights) - \
                    self.epsilon)

            k += 1

    def set_objective(self):
        self.flights: List[IstopFlight]

        self.m.setObjective(
            xp.Sum(self.x[flight.index, j.index] * flight.fitCostVect[j.index]
                   for flight in self.flights for j in self.slots), sense=xp.minimize)  # s

    def run(self, timing=False):

        self.set_variables()

        start = time.time()
        self.set_constraints()
        end = time.time() - start
        if timing:
            print("Constraints setting time ", end)

        self.set_objective()

        start = time.time()
        self.m.solve()
        end = time.time() - start
        if timing:
            print("Simplex time ", end)

        print("problem status, explained: ", self.m.getProbStatusString(), self.m.getObjVal())
        #print(self.m.getObjVal())




        # for flight in self.flights:
        #     if flight.eta > flight.newSlot.time:
        #         print("********************** danno *********************************",
        #               flight, flight.eta, flight.newSlot.time)


        return self.m.getSolution(self.x), self.m.getSolution(self.c)