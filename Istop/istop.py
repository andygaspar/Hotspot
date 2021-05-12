from typing import Callable, Union, List

from GlobalFuns.globalFuns import HiddenPrints
from Istop.AirlineAndFlight.istopFlight import IstopFlight
from Istop.Solvers.mip_solver import MipSolver
from Istop.Solvers.xpress_solver import XpressSolver
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


class Istop(mS.ModelStructure):

    @staticmethod
    def index(array, elem):
        for i in range(len(array)):
            if np.array_equiv(array[i], elem):
                return i

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

        self.airlines: List[IstopAirline]

        for flight in self.flights:
            flight.fitCostVect = flight.costVect

        for airline in self.airlines:
            airline.set_and_standardise_fit_vect()

        self.airlines_pairs = np.array(list(combinations(self.airlines, 2)))
        self.airlines_triples = np.array(list(combinations(self.airlines, 3)))

        self.epsilon = sys.float_info.min
        self.offerChecker = OfferChecker(self.scheduleMatrix)

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

        print("preprocess concluded in sec:", time.time() - start, "   Number of possible offers: ", len(self.matches))
        return len(self.matches) > 0

    def run(self, max_time=120, timing=False):
        feasible = self.check_and_set_matches()
        if feasible:

            try:
                m = XpressSolver(self, max_time)
                solution_vect, offers_vect = m.run(timing=True)

            except:
                print("using MIP")
                p = MipSolver(self, max_time)
                solution_vect, offers_vect = p.run(timing=True)

            self.assign_flights(solution_vect)

            offers = 0
            for i in range(len(self.matches)):
                if offers_vect[i] > 0.9:
                    self.offers_selected.append(self.matches[i])
                    offers += 1
            print("Number of offers selected: ", offers)

        else:
            for flight in self.flights:
                flight.newSlot = flight.slot

        solution.make_solution(self)
        self.offer_solution_maker()

    def other_airlines_compatible_slots(self, flight):
        others_slots = []
        for airline in self.airlines:
            if airline != flight.airline:
                others_slots.extend(airline.AUslots)
        return np.intersect1d(others_slots, flight.compatibleSlots, assume_unique=True)

    def offer_solution_maker(self):
        flight: IstopFlight
        airline_names = ["total"] + [airline.name for airline in self.airlines]
        flights_numbers = [self.numFlights] + [len(airline.flights) for airline in self.airlines]
        # to fix.... it / 4 works only for couples
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

    def assign_flights(self, solution_vect):
        for flight in self.flights:
            for slot in self.slots:
                if solution_vect[flight.slot.index, slot.index] > 0.9:
                    flight.newSlot = slot


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
