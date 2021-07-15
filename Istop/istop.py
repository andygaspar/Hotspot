import sys

from typing import Callable, Union, List
import numpy as np
import pandas as pd
import time

from Hotspot.GlobalFuns.globalFuns import HiddenPrints
from Hotspot.Istop.Solvers.mip_solver import MipSolver
from Hotspot.Istop.Solvers.xpress_solver import XpressSolver
from Hotspot.ModelStructure import modelStructure as mS
from itertools import combinations
from Hotspot.Istop.AirlineAndFlight.istopAirline import IstopAirline
from Hotspot.ModelStructure.Flight.flight import Flight
from Hotspot.ModelStructure.Slot.slot import Slot
from Hotspot.ModelStructure.Solution import solution
from Hotspot.OfferChecker.offerChecker import OfferChecker


def wrap_flight_istop(flight):
    flight.priority = None
    flight.fitCostVect = flight.costVect
    flight.flight_id = None
    flight.standardisedVector = None


class Istop(mS.ModelStructure):
    requirements = ['delayCostVect', 'costVect']

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

    def __init__(self, slot_list: List[Slot]=None, flights: List[Flight]=None, triples=False):
        self.offers = None
        self.triples = triples

        if not flights is None:

            [wrap_flight_istop(flight) for flight in flights]

            super().__init__(slot_list, flights, air_ctor=IstopAirline)

            self.airlines: List[IstopAirline]

            max_delay = self.slots[-1].time - self.slots[0].time
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
            if couple[0].name == c[0].name and couple[1].name == c[1].name:
                return True
            if couple[1].name == c[0].name and couple[0].name == c[1].name:
                return True
            return False

    def f_in_matched(self, flight):
        for f in self.flights_in_matches:
            if f.name == flight.name:
                return True
        return False

    def assign_flights(self, solution_vect):
        for flight in self.flights:
            for slot in self.slots:
                if solution_vect[flight.slot.index, slot.index] > 0.9:
                    flight.newSlot = slot

    # def reset(self, df_init, costFun: Union[Callable, List[Callable]], alpha=1, triples=False):
    #     # To avoid calling xp.problem(), creating a blank line
    #     self.preference_function = lambda x, y: x * (y ** alpha)
    #     self.offers = None
    #     self.triples = triples
    #     super().__init__(df_init=df_init, costFun=costFun, airline_ctor=air.IstopAirline)
    #     airline: air.IstopAirline
    #     for airline in self.airlines:
    #         airline.set_preferences(self.preference_function)

    #     self.airlines_pairs = np.array(list(combinations(self.airlines, 2)))
    #     self.airlines_triples = np.array(list(combinations(self.airlines, 3)))

    #     self.epsilon = sys.float_info.min
    #     self.offerChecker = OfferChecker(self.scheduleMatrix)
    #     with HiddenPrints():
    #         self.m.reset()

    #     self.x = None
    #     self.c = None

    #     self.matches = []
    #     self.couples = []
    #     self.flights_in_matches = []

    #     self.offers_selected = []
