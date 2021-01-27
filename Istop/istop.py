from typing import Callable, Union, List

from ModelStructure import modelStructure as mS
# from mip import *
import sys
from itertools import combinations
from Istop.AirlineAndFlight import istopAirline as air, istopFlight as modFl
from ModelStructure.Solution import solution
from OfferChecker.offerChecker import OfferChecker

import numpy as np
import pandas as pd

import time
import xpress as xp

xp.controls.outputlog = 0


class Istop(mS.ModelStructure):

    @staticmethod
    def index(array, elem):
        for i in range(len(array)):
            if np.array_equiv(array[i], elem):
                return i

    def get_couple(self, couple):
        index = 0
        for c in self.couples:
            if couple[0].num == c[0].num and couple[1].num == c[1].num:
                return index
            index += 1

    @staticmethod
    def get_tuple(flight):
        j = 0
        indexes = []
        for pair in flight.airline.flight_pairs:
            if flight in pair:
                indexes.append(j)
            j += 1
        return indexes

    def get_match_for_flight(self, flight):
        j = 0
        indexes = []
        for match in self.matches:
            for couple in match:
                if flight.num == couple[0].num or flight.num == couple[1].num:
                    indexes.append(j)
            j += 1
        return indexes

    def __init__(self, df_init, costFun: Union[Callable, List[Callable]], alpha=1, triples=False):

        self.preference_function = lambda x, y: x * (y ** alpha)
        self.offers = None
        self.triples = triples
        super().__init__(df_init=df_init, costFun=costFun, airline_ctor=air.IstopAirline)
        airline: air.IstopAirline
        for airline in self.airlines:
            airline.set_preferences(self.preference_function)

        self.airlines_pairs = np.array(list(combinations(self.airlines, 2)))
        self.airlines_triples = np.array(list(combinations(self.airlines, 3)))

        self.epsilon = sys.float_info.min
        self.offerChecker = OfferChecker(self.scheduleMatrix)
        self.m = xp.problem()

        self.x = None
        self.c = None

        self.matches = []
        self.couples = []
        self.flights_in_matches = []

        self.offers_selected = []


    def check_and_set_matches(self):
        start = time.time()
        self.matches = self.offerChecker.all_couples_check(self.airlines_pairs)
        if self.triples:
            self.matches += self.offerChecker.all_triples_check(self.airlines_triples)

        for match in self.matches:
            for couple in match:
                if not self.is_in(couple, self.couples):
                    self.couples.append(couple)
                    if not self.f_in_matched(couple[0]):
                        self.flights_in_matches.append(couple[0])
                    if not self.f_in_matched(couple[1]):
                        self.flights_in_matches.append(couple[1])

        print("preprocess concluded in sec:", time.time()-start, "   Number of possible offers: ", len(self.matches))
        return len(self.matches) > 0

    def set_variables(self):
        self.x = np.array([[xp.var(vartype=xp.binary) for j in self.slots] for i in self.slots])

        self.c = np.array([xp.var(vartype=xp.binary) for i in self.matches])

        self.m.addVariable(self.x, self.c)

    def set_constraints(self):
        # for i in self.emptySlots:
        #     for j in self.slots:
        #         self.m.addConstraint(self.x[i, j] == 0)

        for flight in self.flights:
            if not self.f_in_matched(flight):
                self.m.addConstraint(self.x[flight.slot.index, flight.slot.index] == 1)
            else:
                self.m.addConstraint(xp.Sum(self.x[flight.slot.index, j.index] for j in flight.compatibleSlots) == 1)

        for j in self.slots:
            self.m.addConstraint(xp.Sum(self.x[i.index, j.index] for i in self.slots) <= 1.1)

        for flight in self.flights:
            for j in flight.notCompatibleSlots:
                self.m.addConstraint(self.x[flight.slot.index, j.index] <= 0.1)



        for flight in self.flights_in_matches:
            self.m.addConstraint(xp.Sum([self.c[j] for j in self.get_match_for_flight(flight)]) - 0.1 <=
                                 xp.Sum(self.x[flight.slot.index, slot_to_swap.index] for slot_to_swap in
                                        [s for s in self.slots if s != flight.slot]))
            self.m.addConstraint(
                                 xp.Sum(self.x[flight.slot.index, slot_to_swap.index] for slot_to_swap in
                                        [s for s in self.slots if s != flight.slot]) \
                                 <= xp.Sum([self.c[j] for j in self.get_match_for_flight(flight)]) + 0.1)

            if flight.num == 2:
                self.variabilePerPaola = xp.Sum([self.c[j] for j in self.get_match_for_flight(flight)])

        k = 0
        for match in self.matches:
            flights = [flight for pair in match for flight in pair]

            self.m.addConstraint(xp.Sum(xp.Sum(self.x[i.slot.index, j.slot.index] for i in pair for j in flights)
                                        for pair in match) >= (self.c[k]) * len(match)*2 - 0.1)


            for pair in match:
                self.m.addConstraint(
                    xp.Sum(self.x[i.slot.index, j.slot.index] * i.costFun(i, j.slot) for i in pair for j in flights) -
                    (1 - self.c[k]) * 10000000 \
                    <= xp.Sum(self.x[i.slot.index, j.slot.index] * i.costFun(i, i.slot) for i in pair for j in flights) - \
                    self.epsilon)


            k += 1

    def set_objective(self):

        self.m.setObjective(
            xp.Sum(self.x[flight.slot.index, j.index] * flight.costFun(flight, j)
                   for flight in self.flights for j in self.slots), sense=xp.minimize)

    def run(self, timing=False):
        feasible = self.check_and_set_matches()
        if feasible:
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

            # print("status: ", self.m.getProbStatus())
            print("problem status, explained: ", self.m.getProbStatusString())
            xpSolution = self.x
            # print(self.m.getSolution(self.x))
            self.assign_flights(xpSolution)

        else:
            for flight in self.flights:
                flight.newSlot = flight.slot

        solution.make_solution(self)

        self.offer_solution_maker()

        # for i in self.slots:
        #     if self.x[i, i].x == 0:
        #         for j in self.slots:
        #             if self.x[i, j].x != 0:
        #                 print(i, j)

        for flight in self.flights:
            if flight.eta > flight.newSlot.time:
                print("********************** danno *********************************",
                      flight, flight.eta, flight.newSlot.time)

        offers = 0
        for i in range(len(self.matches)):
            if self.m.getSolution(self.c[i]) > 0.5:
                self.offers_selected.append(self.matches[i])
                offers += 1
        print("Number of offers selected: ", offers)

    def other_airlines_compatible_slots(self, flight):
        others_slots = []
        for airline in self.airlines:
            if airline != flight.airline:
                others_slots.extend(airline.AUslots)
        return np.intersect1d(others_slots, flight.compatibleSlots, assume_unique=True)

    def score(self, flight, slot):
        return (flight.preference * flight.delay(slot) ** 2) / 2

    def offer_solution_maker(self):

        flight: modFl.IstopFlight
        airline_names = ["total"] + [airline.name for airline in self.airlines]
        flights_numbers = [self.numFlights] + [len(airline.flights) for airline in self.airlines]
        offers = [sum([1 for flight in self.flights if flight.slot != flight.newSlot]) / 4]
        for airline in self.airlines:
            offers.append(sum([1 for flight in airline.flights if flight.slot != flight.newSlot]) / 2)

        offers = np.array(offers).astype(int)
        self.offers = pd.DataFrame({"airline": airline_names, "flights": flights_numbers, "offers": offers})
        self.offers.sort_values(by="flights", inplace=True, ascending=False)


    @staticmethod
    def is_in(couple, couples):
        for c in couples:
            if couple[0].num == c[0].num and couple[1].num == c[1].num:
                return True
            if couple[1].num == c[0].num and couple[0].num == c[1].num:
                return True
            return False

    def f_in_matched(self, flight):
        for f in self.flights_in_matches:
            if f.num == flight.num:
                return True
        return False

    def assign_flights(self, xpSolution):
        for flight in self.flights:
            print([self.m.getSolution(self.x[flight.slot.index,j.index]) for j in self.slots])
            for slot in self.slots:
                if self.m.getSolution(xpSolution[flight.slot.index, slot.index]) > 0.5:
                    flight.newSlot = slot
                    # print(flight, flight.slot, flight.newSlot)

    # def find_match(self, i):
    #     for j in self.slotIndexes[self.slotIndexes != i]:
    #         if self.xpSolution[i.slot, j] == 1:
    #             return self.flights[j]
