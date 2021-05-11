import sys
import time
from typing import List

import mip
import numpy as np

from Istop.AirlineAndFlight.istopFlight import IstopFlight


class MipSolver:

    def __init__(self, model, max_time):

        self.m = mip.Model()
        self.m.verbose = False
        self.m.threads = -1
        self.maxTime = max_time
        self.m.preprocess = 1
        self.m.pump_passes = 1000
        self.m.max_solutions = 1
        self.m.emphasis = 1


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

        self.x = np.array([[self.m.add_var(var_type=mip.BINARY) for _ in self.slots] for _ in self.slots])
        self.c = np.array([self.m.add_var(var_type=mip.BINARY) for _ in self.matches])

    def set_constraints(self):

        self.flights: List[IstopFlight]

        for i in self.emptySlots:
            for j in self.slots:
                self.m += self.x[i, j] == 0

        for flight in self.flights:
            if not self.f_in_matched(flight):
                self.m += self.x[flight.slot.index, flight.slot.index] == 1
            else:
                self.m += mip.xsum(self.x[flight.slot.index, j.index] for j in flight.compatibleSlots) == 1

        for j in self.slots:
            self.m += mip.xsum(self.x[i.index, j.index] for i in self.slots) <= 1

        for flight in self.flights:
            for j in flight.notCompatibleSlots:
                self.m += self.x[flight.slot.index, j.index] == 0

        for flight in self.flights_in_matches:
            self.m += mip.xsum(self.x[flight.slot.index, slot.index]
                               for slot in self.slots if slot != flight.slot) \
                      <= mip.xsum([self.c[j] for j in self.get_match_for_flight(flight)])

            self.m += mip.xsum([self.c[j] for j in self.get_match_for_flight(flight)]) <= 1

        k = 0
        for match in self.matches:
            flights = [flight for pair in match for flight in pair]
            self.m += mip.xsum(mip.xsum(self.x[i.slot.index, j.slot.index] for i in pair for j in flights)
                               for pair in match) >= (self.c[k]) * len(flights)

            for pair in match:
                self.m += mip.xsum(
                    self.x[i.slot.index, j.slot.index] * i.fitCostVect[j.slot.index] for i in pair for j in
                    flights) - (1 - self.c[k]) * 10000000 \
                          <= mip.xsum(
                    self.x[i.slot.index, j.slot.index] * i.fitCostVect[i.slot.index] for i in pair for j in
                    flights) - \
                          self.epsilon

            k += 1

    def set_objective(self):
        self.flights: List[IstopFlight]

        self.m.objective = mip.minimize(mip.xsum(self.x[flight.slot.index, j.index] * flight.fitCostVect[j.index]
                                                 for flight in self.flights for j in self.slots))  # s

    def run(self, timing=False):

        self.set_variables()
        self.m.start = [(self.x[i, i], 1.0) for i in range(len(self.flights))]
        start = time.time()
        self.set_constraints()
        end = time.time() - start
        if timing:
            print("Constraints setting time ", end)

        self.set_objective()

        start = time.time()
        self.m.optimize(max_seconds=self.maxTime)
        end = time.time() - start
        if timing:
            print("Simplex time ", end)

        print("problem status, value: ", self.m.status, self.m.objective_value)

        return np.array([[el.x for el in col] for col in self.x]), np.array([el.x for el in self.c])
