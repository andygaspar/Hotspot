import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np
import time
from itertools import permutations
import copy
import os


class OfferChecker(object):

    def __init__(self, schedule_mat):

        self.numProcs = os.cpu_count()
        self.lib = ctypes.CDLL('./OfferChecker/liboffers_parallel.so')
        self.lib.OfferChecker_.argtypes = [ctypes.c_void_p, ctypes.c_short, ctypes.c_short,
                                           ctypes.c_void_p, ctypes.c_short, ctypes.c_short, ctypes.c_void_p,
                                           ctypes.c_short, ctypes.c_short, ctypes.c_short]
        self.lib.air_couple_check_.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
        self.lib.air_triple_check_.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]

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

    def air_couple_check(self, air_pair):
        fl_pair_a = air_pair[0].flight_pairs
        fl_pair_b = air_pair[1].flight_pairs

        air_pairs = []
        input_vect = []
        for pairA in fl_pair_a:
            for pairB in fl_pair_b:
                air_pairs.append([pairA, pairB])
                input_vect += [fl.slot.index for fl in pairA] + [fl.slot.index for fl in pairB]

        len_array = int(len(input_vect) / 4)

        if len_array > 0:
            self.lib.air_couple_check_.restype = ndpointer(dtype=ctypes.c_bool, shape=(len_array,))
            input_vect = np.array(input_vect).astype(np.short)
            answer = self.lib.air_couple_check_(ctypes.c_void_p(self.obj),
                                                ctypes.c_void_p(input_vect.ctypes.data), ctypes.c_uint(len_array))

            return [air_pairs[i] for i in range(len_array) if answer[i]]
        else:
            return []

    def all_couples_check(self, airlines_pairs):
        matches = []
        for air_pair in airlines_pairs:
            match = self.air_couple_check(air_pair)
            if len(match) > 0:
                matches += match

        return matches

    def air_triple_check(self, air_triple):
        fl_pair_a = air_triple[0].flight_pairs
        fl_pair_b = air_triple[1].flight_pairs
        fl_pair_c = air_triple[2].flight_pairs
        air_trips = []
        input_vect = []
        for pairA in fl_pair_a:
            for pairB in fl_pair_b:
                for pairC in fl_pair_c:
                    air_trips.append([pairA, pairB, pairC])
                    input_vect += [fl.slot.index for fl in pairA] + [fl.slot.index for fl in pairB] + \
                                  [fl.slot.index for fl in pairC]

        len_array = int(len(input_vect) / 6)

        if len_array > 0:
            self.lib.air_triple_check_.restype = ndpointer(dtype=ctypes.c_bool, shape=(len_array,))
            input_vect = np.array(input_vect).astype(np.short)

            answer = self.lib.air_triple_check_(ctypes.c_void_p(self.obj),
                                                ctypes.c_void_p(input_vect.ctypes.data), ctypes.c_uint(len_array))
            return [air_trips[i] for i in range(len_array) if answer[i]]

        else:
            return []

    def all_triples_check(self, airlines_triples):
        matches = []

        for air_triple in airlines_triples:
            match = self.air_triple_check(air_triple)
            if len(match) > 0:
                matches += match
        return matches

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
