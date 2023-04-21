import sys
import time
from typing import List

import numpy as np

from ...ModelStructure.Flight.flight import Flight
from ...GlobalFuns.globalFuns import HiddenPrints
from gurobipy import Model, GRB, quicksum, Env

import time


class Offer:
    def __init__(self, offer, reduction, num):
        self.offer = offer
        self.reduction = reduction
        self.flights = [flight for couple in offer for flight in couple]
        self.num = num

    def __repr__(self):
        return str(self.num) + " " + ' '.join([f.name for f in self.flights])

    def __eq__(self, other):
        return self.num == other.num

    def __lt__(self, other):
        return self.num < other.num

def stop(model, where):

    if where == GRB.Callback.MIP:
        objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        run_time = model.cbGet(GRB.Callback.RUNTIME)

        if run_time > model._time_limit:
            #print("stop at", run_time)
            model.terminate()


class GurobiOfferSolver:

    def __init__(self, model, offers, max_offers, time_limit, reductions, mip_gap):

        self.m = Model('CVRP')
        # self.m.setParam('Method', 2) ###################testare == 2 !!!!!!!!!!!!111c
        self.m.modelSense = GRB.MAXIMIZE
        self.m.setParam('OutputFlag', 0)
        self.m.setParam('LogToConsole', 0)
        with HiddenPrints():
            self.m.setParam('MIPGap', mip_gap)
        self.m._time_limit = time_limit

        self.flights = model.flights
        self.airlines = model.airlines
        self.slots = model.slots

        self.matches = model.matches
        self.emptySlots = model.emptySlots
        self.flights_in_matches = model.flights_in_matches

        self.f_in_matched = model.f_in_matched
        self.get_match_for_flight = model.get_match_for_flight
        self.check_and_set_matches = model.check_and_set_matches

        self.c = None

        order = np.flip(np.argsort(reductions))
        self.offers = [Offer(offers[j], reductions[j], i) for i, j in enumerate(order)]
        self.numOffers = len(self.offers)
        self.reductions = np.array([reductions[i] for i in order])

        if len(self.offers) > max_offers:
            self.offers = self.offers[:max_offers]
            self.reductions = self.reductions[:max_offers]

        self.set_match_for_flight(model.flights)

        self.compatibilityMatrix = None

    def set_variables(self):

        self.c = self.m.addMVar(self.numOffers, vtype=GRB.BINARY)

    def set_priority(self):

        for i, priority in enumerate(self.reductions):
            self.c[i].setAttr("BranchPriority", int(priority))

    def set_constraints(self):

        self.compatibilityMatrix = np.zeros((self.numOffers, self.numOffers), dtype=bool)

        for i, offer in enumerate(self.offers):
            incompatible = np.unique([off for flight in offer.flights for off in flight.offers])
            self.compatibilityMatrix[i, incompatible] = True
            self.compatibilityMatrix[i, i] = False

        for i in range(self.numOffers):
            self.compatibilityMatrix[i, i] = False

        self.m.addConstr(self.compatibilityMatrix@self.c <= (np.ones(self.numOffers)-self.c)*self.numOffers)

    def set_objective(self):
        self.flights: List[Flight]

        self.m.setObjective(self.c@self.reductions)

    def run(self, timing=False, verbose=False, branching=False):


        if not verbose:
            self.m.setParam('OutputFlag', 0)

        self.set_variables()
        if branching:
            self.set_priority()
        start = time.time()
        self.set_constraints()
        end = time.time() - start
        if timing:
            print("Constraints setting time ", end)

        self.set_objective()
        self.m.setParam('Cuts', 3)
        start = time.time()
        # self.m.optimize(stop)
        self.m.optimize()
        end = time.time() - start

        if timing:
            print("Solution time ", end)

        status = None
        if self.m.status == 2:
            status = "optimal"
        if self.m.status == 3:
            status = "infeasible"
        # print(status)

        offer_solution = self.get_solution_offers()

        return offer_solution

    def set_match_for_flight(self, flights: List[Flight]):
        for flight in flights:
            for offer in self.offers:
                match = offer.offer
                for couple in match:
                    if flight.slot == couple[0].slot or flight.slot == couple[1].slot:
                        flight.offers.append(offer.num)

    def get_solution_offers(self):
        solution = []
        for i, offer in enumerate(self.offers):
            if self.c[i].x > 0.5:
                solution.append(offer.offer)
        return solution

