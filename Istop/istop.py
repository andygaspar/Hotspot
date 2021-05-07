from typing import Callable, Union, List

from GlobalFuns.globalFuns import HiddenPrints
from Istop.AirlineAndFlight.istopFlight import IstopFlight
from ModelStructure import modelStructure as mS
# from mip import *
import sys
from itertools import combinations
from Istop.AirlineAndFlight.istopAirline import IstopAirline
from ModelStructure.Flight.flight import Flight
from ModelStructure.Solution import solution
from OfferChecker.offerChecker import OfferChecker

import numpy as np
import pandas as pd

import time
import xpress as xp

# xp.controls.outputlog = 0


class Istop(mS.ModelStructure):

    @staticmethod
    def index(array, elem):
        for i in range(len(array)):
            if np.array_equiv(array[i], elem):
                return i

    # def get_couple(self, couple):
    #     index = 0
    #     for c in self.couples:
    #         if couple[0].num == c[0].num and couple[1].num == c[1].num:
    #             return index
    #         index += 1

    # @staticmethod
    # def get_tuple(flight):
    #     j = 0
    #     indexes = []
    #     for pair in flight.airline.flight_pairs:
    #         if flight in pair:
    #             indexes.append(j)
    #         j += 1
    #     return indexes

    def get_match_for_flight(self, flight):
        j = 0
        indexes = []
        for match in self.matches:
            for couple in match:
                if flight.slot == couple[0].slot or flight.slot == couple[1].slot:
                    indexes.append(j)
            j += 1
        return indexes

    def __init__(self, flights: List[Flight], triples=False):
        self.offers = None
        self.triples = triples

        istop_flights = [IstopFlight(flight) for flight in flights]

        super().__init__(istop_flights, air_ctor=IstopAirline)
        airline: IstopAirline
        for airline in self.airlines:
            airline.set_preferences(max_delay=self.slots[-1].time - self.slots[0].time)

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
        self.x = np.array([[xp.var(vartype=xp.binary) for _ in self.slots] for _ in self.slots], dtype=xp.npvar)

        self.c = np.array([xp.var(vartype=xp.binary) for _ in self.matches], dtype=xp.npvar)

        self.m.addVariable(self.x, self.c)

    def set_constraints(self):
        for i in self.emptySlots:
            for j in self.slots:
                self.m.addConstraint(self.x[i, j] == 0)

        for flight in self.flights:
            if not self.f_in_matched(flight):
                self.m.addConstraint(self.x[flight.slot.index, flight.slot.index] == 1)
            else:
                self.m.addConstraint(xp.Sum(self.x[flight.slot.index, j.index] for j in flight.compatibleSlots) == 1)

        for j in self.slots:
            self.m.addConstraint(xp.Sum(self.x[i.index, j.index] for i in self.slots) <= 1)

        for flight in self.flights:
            for j in flight.notCompatibleSlots:
                self.m.addConstraint(self.x[flight.slot.index, j.index] == 0)

        for flight in self.flights_in_matches:
            self.m.addConstraint(
                                 xp.Sum(self.x[flight.slot.index, slot.index]
                                        for slot in self.slots if slot != flight.slot) \
                                 <= xp.Sum([self.c[j] for j in self.get_match_for_flight(flight)]))

            self.m.addConstraint(xp.Sum([self.c[j] for j in self.get_match_for_flight(flight)]) <= 1)

        k = 0
        for match in self.matches:
            flights = [flight for pair in match for flight in pair]
            self.m.addConstraint(xp.Sum(xp.Sum(self.x[i.slot.index, j.slot.index] for i in pair for j in flights)
                                        for pair in match) >= (self.c[k]) * len(flights))

            for pair in match:
                self.m.addConstraint(
                    xp.Sum(self.x[i.slot.index, j.slot.index] * i.costVect[j.slot.index] for i in pair for j in flights) -
                    (1 - self.c[k]) * 10000000 \
                    <= xp.Sum(self.x[i.slot.index, j.slot.index] * i.costVect[i.slot.index] for i in pair for j in flights) - \
                    self.epsilon)

            k += 1



    def set_objective(self):

        self.m.setObjective(
            xp.Sum(self.x[flight.slot.index, j.index] * flight.costFun(j)
                   for flight in self.flights for j in self.slots), sense=xp.minimize) #self.scrore instead of cost

    def run(self, timing=False):
        feasible = self.check_and_set_matches()
        if feasible:
            # print("before", self.m.getControl('presolve'))
            # # self.m.setControl({'presolve': 0})
            # self.m.setControl({'maxnode': 100})
            # print("after", self.m.getControl('presolve'))
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
            xpSolution = self.x
            # print(self.m.getSolution(self.x))
            self.assign_flights(xpSolution)

        else:
            for flight in self.flights:
                flight.newSlot = flight.slot

        solution.make_solution(self)
        # 24170.971882985057
        self.offer_solution_maker()

        for flight in self.flights:
            if flight.eta > flight.newSlot.time:
                print("********************** danno *********************************",
                      flight, flight.eta, flight.newSlot.time)

        offers = 0
        for i in range(len(self.matches)):
            # print(self.m.getSolution(self.c[i]))
            if self.m.getSolution(self.c[i]) > 0.9:
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

        flight: IstopFlight
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
            for slot in self.slots:
                if self.m.getSolution(xpSolution[flight.slot.index, slot.index]) > 0.9:
                    flight.newSlot = slot

    # def update_object(self, df_init, costFun: Union[Callable, List[Callable]], alpha=1, triples=False):
    #     self.preference_function = lambda x, y: x * (y ** alpha)
    #     self.offers = None
    #     self.triples = triples
    #     super().__init__(df_init=df_init, costFun=costFun, airline_ctor=air.IstopAirline)
    #     airline: IstopAirline
    #     for airline in self.airlines:
    #         #airline.set_preferences(self.preference_function)
    #         pass
    #
    #     self.airlines_pairs = np.array(list(combinations(self.airlines, 2)))
    #     self.airlines_triples = np.array(list(combinations(self.airlines, 3)))
    #
    #     self.epsilon = sys.float_info.min
    #     self.offerChecker = OfferChecker(self.scheduleMatrix)
    #     with HiddenPrints():
    #         self.m.reset()
    #
    #     self.x = None
    #     self.c = None
    #
    #     self.matches = []
    #     self.couples = []
    #     self.flights_in_matches = []
    #
    #     self.offers_selected = []

"""
0   total           50  299101.666667  298623.000000         0.16
1       B           16  104410.000000  102681.333333         1.66
2       E            1      28.000000      28.000000         0.00
3       C           10   31916.333333   31818.333333         0.31
4       A           19  120191.333333  121539.333333        -1.12
5       D            4   42556.000000   42556.000000         0.00

"""


"""
rows 936
problem status, explained:  mip_optimal 27612.33615924535
rows 936
sets 0
setmembers 0
elems 8140
primalinfeas 0
dualinfeas 0
simplexiter 42033
lpstatus 1
mipstatus 6
cuts 0
nodes 99
nodedepth 1
activenodes 0
mipsolnode 1042
mipsols 14
cols 2545
sparerows 793
sparecols 0
spareelems 2968
sparemipents 315
errorcode 0
mipinfeas 132
presolvestate 1310881
parentnode 0
namelength 1
qelems 0
numiis 0
mipents 2545
branchvar 0
mipthreadid 0
algorithm 2
time 1
originalrows 936
callbackcount_optnode 0
callbackcount_cutmgr 0
systemmemory 1596327
originalqelems 0
maxprobnamelength 1024
stopstatus 0
originalmipents 2545

"""