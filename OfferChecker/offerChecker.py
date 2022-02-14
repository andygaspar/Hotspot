import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np
import time
from itertools import permutations
import copy
import os
from pathlib import Path


import ctypes
from multiprocessing import Pool

from numpy.ctypeslib import ndpointer
import numpy as np
import time
from itertools import permutations, product
import copy
import os


def air_couple_check(air_pair):
    fl_pair_a = air_pair[0].flight_pairs_idx
    fl_pair_b = air_pair[1].flight_pairs_idx
    air_coups = list(product(fl_pair_a, fl_pair_b))
    input_vector = np.array(air_coups).flatten()

    return input_vector, air_coups


def air_triple_check(air_triple):

    fl_pair_a = air_triple[0].flight_pairs_idx
    fl_pair_b = air_triple[1].flight_pairs_idx
    fl_pair_c = air_triple[2].flight_pairs_idx

    air_trips = list(product(fl_pair_a, fl_pair_b, fl_pair_c))
    input_vector = np.array(air_trips).flatten()

    return input_vector, air_trips


class OfferChecker(object):

    def __init__(self, schedule_mat, flights, offers_convenience_tol=0.01):
        # os.system('./OfferChecker/install_parallel.sh')
        self.numProcs = os.cpu_count()
        roor_dir = Path(__file__).resolve().parents[1]
        self.lib = ctypes.CDLL(roor_dir / 'OfferChecker/liboffers_parallel.so')
        self.lib.OfferChecker_.argtypes = [ctypes.c_void_p, ctypes.c_short, ctypes.c_short,
                                           ctypes.c_void_p, ctypes.c_short, ctypes.c_short, ctypes.c_void_p,
                                           ctypes.c_short, ctypes.c_short, ctypes.c_short]
        self.lib.air_couple_check_.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
        self.lib.air_triple_check_.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]

        self.compTime = 0

        self.flights = flights
        self.tolerance = offers_convenience_tol

        couples = list(permutations([0, 1, 2, 3]))
        couples_copy = copy.copy(couples)
        for c in couples_copy:
            if (c[0] == 0 and c[1] == 1) or (c[0] == 1 and c[1] == 0):
                couples.remove(c)
        self.couples = np.array(couples).astype(np.short)

        triples = list(permutations([0, 1, 2, 3, 4, 5]))
        triples_copy = copy.copy(triples)
        for t in triples_copy:
            if ((t[0] == 0 and t[1] == 1) or (t[0] == 1 and t[1] == 0)) or \
                    ((t[2] == 2 and t[3] == 3) or (t[2] == 3 and t[3] == 2)) or \
                    ((t[4] == 4 and t[5] == 5) or (t[4] == 5 and t[5] == 4)):
                triples.remove(t)
        self.triples = np.array(triples).astype(np.short)

        self.lib.OfferChecker_.restype = ctypes.c_void_p
        self.obj = self.lib.OfferChecker_(ctypes.c_void_p(schedule_mat.ctypes.data),
                                          ctypes.c_short(schedule_mat.shape[0]),
                                          ctypes.c_short(schedule_mat.shape[1]),
                                          ctypes.c_void_p(self.couples.ctypes.data),
                                          ctypes.c_short(self.couples.shape[0]),
                                          ctypes.c_short(self.couples.shape[1]),
                                          ctypes.c_void_p(self.triples.ctypes.data),
                                          ctypes.c_short(self.triples.shape[0]),
                                          ctypes.c_short(self.triples.shape[1]),
                                          ctypes.c_short(self.numProcs))

    def get_flights(self, fl_tuple):
        return [np.array([self.flights[tup[0]], self.flights[tup[1]]]) for tup in fl_tuple]

    def all_couples_check(self, airlines_pairs):
        airlines_pairs = [pair for pair in airlines_pairs
                          if len(pair[0].flight_pairs) > 0 and len(pair[1].flight_pairs) > 0]

        input_vect = np.array([], dtype=np.short)
        air_pairs = []

        for pair in airlines_pairs:
            res = air_couple_check(pair)
            input_vect = np.append(input_vect, res[0])
            air_pairs += res[1]

        len_array = int(len(input_vect) / 4)
        if len_array > 0:
            self.lib.air_couple_check_.restype = ndpointer(dtype=ctypes.c_double, shape=(len_array,))

            t = time.time()
            reductions = self.lib.air_couple_check_(ctypes.c_void_p(self.obj),
                                                ctypes.c_void_p(input_vect.ctypes.data), ctypes.c_uint(len_array))
            self.compTime += time.time() - t
            idxs = [i for i in range(len_array) if reductions[i] > self.tolerance]

            return [self.get_flights(air_pairs[i]) for i in idxs], reductions[idxs]
        else:
            return [], []

    def all_triples_check(self, airlines_triples):
        airlines_triples = [triple for triple in airlines_triples
                            if len(triple[0].flight_pairs) > 0 and len(triple[1].flight_pairs) > 0
                            and len(triple[2].flight_pairs) > 0]
        input_vect = np.array([], dtype=np.short)
        air_trips = []

        for triple in airlines_triples:
            res = air_triple_check(triple)
            input_vect = np.append(input_vect, res[0])
            air_trips += res[1]
        len_array = int(len(input_vect) / 6)

        if len_array > 0:
            self.lib.air_triple_check_.restype = ndpointer(dtype=ctypes.c_double, shape=(len_array,))

            t = time.time()
            reductions = self.lib.air_triple_check_(ctypes.c_void_p(self.obj),
                                                ctypes.c_void_p(input_vect.ctypes.data), ctypes.c_uint(len_array))
            self.compTime += time.time() - t

            idxs = [i for i in range(len_array) if reductions[i] > self.tolerance]
            return [self.get_flights(air_trips[i]) for i in idxs], reductions[idxs]
        else:
            return [], []


    def check_couple_in_pairs(self, couple, airlines_pairs):
        other_airline = None

        air_pairs = []
        input_vect = []
        for air_pair in airlines_pairs:
            if couple[0].airline.name == air_pair[0].name:
                other_airline = air_pair[1]
            elif couple[0].airline.name == air_pair[1].name:
                other_airline = air_pair[0]

            if other_airline is not None:
                for pair in other_airline.flight_pairs:
                    air_pairs.append([couple, pair])
                    input_vect += [fl.slot.index for fl in couple] + [fl.slot.index for fl in pair]

        len_array = int(len(input_vect) / 4)

        self.lib.air_couple_check_.restype = ndpointer(dtype=ctypes.c_bool, shape=(len_array,))
        input_vect = np.array(input_vect).astype(np.short)
        answer = self.lib.air_couple_check_(ctypes.c_void_p(self.obj),
                                            ctypes.c_void_p(input_vect.ctypes.data), ctypes.c_uint(len_array))

        return [air_pairs[i] for i in range(len_array) if answer[i]]

    def check_couple_in_triples(self, couple, airlines_triples):
        other_airline_A = None
        other_airline_B = None

        air_trips = []
        input_vect = []

        for air_pair in airlines_triples:
            if couple[0].airline.name == air_pair[0].name:
                other_airline_A = air_pair[1]
                other_airline_B = air_pair[2]
            elif couple[0].airline.name == air_pair[1].name:
                other_airline_A = air_pair[0]
                other_airline_B = air_pair[2]
            elif couple[0].airline.name == air_pair[2].name:
                other_airline_A = air_pair[0]
                other_airline_B = air_pair[1]

            if other_airline_A is not None:
                for pairB in other_airline_A.flight_pairs:
                    for pairC in other_airline_B.flight_pairs:
                        air_trips.append([couple, pairB, pairC])
                        input_vect += [fl.slot.index for fl in couple] + [fl.slot.index for fl in pairB] + \
                                      [fl.slot.index for fl in pairC]

        len_array = int(len(input_vect) / 6)
        self.lib.air_triple_check_.restype = ndpointer(dtype=ctypes.c_bool, shape=(len_array,))
        input_vect = np.array(input_vect).astype(np.short)

        answer = self.lib.air_triple_check_(ctypes.c_void_p(self.obj),
                                            ctypes.c_void_p(input_vect.ctypes.data), ctypes.c_uint(len_array))
        return [air_trips[i] for i in range(len_array) if answer[i]]

    def print_mat(self):
        self.lib.print_mat_(self.obj)

    def print_couples(self):
        self.lib.print_couples_(self.obj)

    def print_triples(self):
        self.lib.print_triples_(self.obj)

    def get_solution_assignment(self, matches):

        solution_assignment = []

        for match in matches:
            couple = (len(match) == 2)
            if couple:
                all_fl_in_offer = [fl for tup in match for fl in tup]
                init_cost_a = sum(flight.standardisedVector[flight.slot.index] for flight in match[0])
                init_cost_b = sum(flight.standardisedVector[flight.slot.index] for flight in match[1])
                init_cost = init_cost_a + init_cost_b
                best_offer_reduction = 0
                for perm in self.couples:
                    if np.prod(flight.etaSlot <= all_fl_in_offer[perm[i]].slot for i, flight in
                               enumerate(all_fl_in_offer)):
                        final_cost_a = sum(flight.standardisedVector[[all_fl_in_offer[perm[i]].slot.index]][0]
                                           for i, flight in enumerate(match[0]))
                        final_cost_b = sum(flight.standardisedVector[[all_fl_in_offer[perm[i + 2]].slot.index]][0]
                                           for i, flight in enumerate(match[1]))
                        offer_reduction = init_cost - final_cost_b - final_cost_a
                        if final_cost_a < init_cost_a and final_cost_b < init_cost_b \
                                and offer_reduction > best_offer_reduction:
                            best_offer_reduction = offer_reduction
                            assignment = perm

            else:
                all_fl_in_offer = [fl for tup in match for fl in tup]
                init_cost_a = sum(flight.standardisedVector[flight.slot.index] for flight in match[0])
                init_cost_b = sum(flight.standardisedVector[flight.slot.index] for flight in match[1])
                init_cost_c = sum(flight.standardisedVector[flight.slot.index] for flight in match[2])
                init_cost = init_cost_a + init_cost_b + init_cost_c
                best_offer_reduction = 0
                for perm in self.triples:
                    if np.prod(flight.etaSlot <= all_fl_in_offer[perm[i]].slot for i, flight in
                               enumerate(all_fl_in_offer)):
                        final_cost_a = sum(flight.standardisedVector[[all_fl_in_offer[perm[i]].slot.index]][0]
                                           for i, flight in enumerate(match[0]))
                        final_cost_b = sum(flight.standardisedVector[[all_fl_in_offer[perm[i + 2]].slot.index]][0]
                                           for i, flight in enumerate(match[1]))
                        final_cost_c = sum(flight.standardisedVector[[all_fl_in_offer[perm[i + 4]].slot.index]][0]
                                           for i, flight in enumerate(match[2]))
                        offer_reduction = init_cost - final_cost_b - final_cost_a - final_cost_c

                        if final_cost_a < init_cost_a and final_cost_b < init_cost_b and final_cost_c < init_cost_c \
                                and offer_reduction > best_offer_reduction:
                            best_offer_reduction = offer_reduction
                            assignment = perm

            flights = [fl for m in match for fl in m]
            slots = [fl.slot for fl in flights]

            for i, fl in enumerate(flights):
                solution_assignment.append((fl, slots[assignment[i]]))

        return solution_assignment
