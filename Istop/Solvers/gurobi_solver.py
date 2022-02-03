import sys
import time
from typing import List

# import xpress as xp
# xp.controls.outputlog = 0
import numpy as np

from ...ModelStructure.Flight.flight import Flight
from gurobipy import Model, GRB, quicksum, Env

import time


def stop(model, where):

    if where == GRB.Callback.MIP:
        objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        run_time = model.cbGet(GRB.Callback.RUNTIME)

        if run_time > model._time_limit and abs(objbst - objbnd) < 0.005 * abs(objbst):
            print("stop at", run_time)
            model.terminate()

class GurobiSolver:

    def __init__(self, model):

        self.m = Model('CVRP')
        # self.m.setParam('Method', 2) ###################testare == 2 !!!!!!!!!!!!111c
        self.m.modelSense = GRB.MINIMIZE

        self.flights = model.flights
        self.airlines = model.airlines
        self.slots = model.slots

        self.matches = model.matches
        self.emptySlots = model.emptySlots
        self.flights_in_matches = model.flights_in_matches

        self.f_in_matched = model.f_in_matched
        #self.get_match_for_flight = model.get_match_for_flight
        self.check_and_set_matches = model.check_and_set_matches

        self.epsilon = sys.float_info.min

        # # For debugging
        # stuff0 = [f.name for f in self.flights]
        # stuff = [f.index for f in self.flights]
        # stuff2 = [f.slot for f in self.flights]
        # # print ('slots:', self.slots)
        # # print ('SLOTS IN GurobiSolver for each flight:', stuff2)
        # stuff3 = [f.compatibleSlots for f in self.flights]
        # stuff4 = [f.notCompatibleSlots for f in self.flights]
        # stuff5 = [f.fitCostVect for f in self.flights]
        # #print ('MATCHES:', [[[f.name for f in stuff2] for stuff2 in stuff] for stuff in self.matches])
        # #raise Exception()
        # to_save = (stuff0, stuff, stuff2, stuff3, stuff4, stuff5,
        #             self.slots, #self.matches,
        #             self.emptySlots, [f.index for f in self.flights_in_matches],
        #             [[[f.name for f in stuff2] for stuff2 in stuff] for stuff in self.matches])

        # for flight in self.flights:
        #     if flight.slot.index<flight.compatibleSlots[0].index:
        #         print ('BAAAAAAAAMMMMMM', flight.name, flight.slot, flight.compatibleSlots[:1])
        #         raise Exception()

        # for match in self.matches:
        #     pair1, pair2 = match
        #     #print (pair1, pair2)

        #     # if pair1[0].name==1205:
        #     #     print ('CHECKCHECK', pair1[0].name, pair1[0].slot)
            
        #     try:
        #         assert pair1[0].slot in pair1[1].compatibleSlots
        #     except:
        #         print (pair1, pair1[0].slot, pair1[1].compatibleSlots[0])
        #     try:
        #         assert pair1[1].slot in pair1[0].compatibleSlots
        #     except:
        #         print (pair1, pair1[1].slot, pair1[0].compatibleSlots[0])
        #     try:
        #         assert pair2[0].slot in pair2[1].compatibleSlots
        #     except:
        #         print (pair2, pair2[0].slot, pair2[1].compatibleSlots[0])
        #     try:
        #         assert pair2[1].slot in pair2[0].compatibleSlots
        #     except:
        #         print (pair2, pair2[1].slot, pair2[0].compatibleSlots[0])
                

        # import pickle
        # with open('gurobi_solver_state2.pic', 'wb') as f:
        #     pickle.dump(to_save, f)

        # # FOR TESTING
        # n1 = 14 # 15 optimal
        # n2 = 25# 24 optimal
        # exclude = [15, 16, 17, 18, 21, 22, 23]
        # #self.flights = self.flights[n1:n2]
        # self.flights = [flight for i, flight in enumerate(self.flights) if i>=n1 and i<n2 and (not i in exclude)]
        # #self.slots = [flight.slot for flight in self.flights]
        # f_name = [flight.name for flight in self.flights]
        # self.matches = [[pair1, pair2] for pair1, pair2 in self.matches if (pair1[0].name in f_name) and
        #                                                                     (pair1[1].name in f_name) and
        #                                                                     (pair2[0].name in f_name) and
        #                                                                     (pair2[1].name in f_name)]
        # self.flights_in_matches = [flight for flight in self.flights_in_matches if flight in self.flights]
        # #print ('matches:', self.matches)
        # for i, flight in enumerate(self.flights):
        #     flight.old_index = flight.index
        #     flight.index = i
        # print ('REMAINING FLIGHTS:', self.flights)
        # print ('INDEXES:', [flight.index for flight in self.flights])
        # print ('FLIGHTS IN MATCHES', self.flights_in_matches)

        # if load_debug:
        #     import pickle
        #     with open('gurobi_solver_state.pic', 'rb') as f:
        #         self.flights, self.airlines, self.slots, self.matches,
        #         self.emptySlots, self.flights_in_matches, self.f_in_matched,
        #         self.get_match_for_flight, self.check_and_set_matches = pickle.load(f)


        self.x = None
        self.c = None

    def get_match_for_flight(self, flight):
        j = 0
        indexes = []
        for match in self.matches:
            for couple in match:
                if flight.slot == couple[0].slot or flight.slot == couple[1].slot:
                    indexes.append(j)
            j += 1
        return indexes

    def set_variables(self):

        # self.x = self.m.addVars([(i, j) for i in range(len(self.slots)) for j in range(len(self.slots))],
        self.x = self.m.addVars([(i, j) for i in range(len(self.flights)) for j in range(len(self.slots))],
                       vtype=GRB.BINARY)

        self.c = self.m.addVars([i for i in range(len(self.matches))])

    def set_constraints(self):

        self.flights: List[Flight]

        # for i in self.emptySlots:
        #     for j in self.slots:
        #         self.m.addConstr(self.x[i, j] == 0)

        for flight in self.flights:
            if not self.f_in_matched(flight):
                self.m.addConstr(self.x[flight.index, flight.slot.index] == 1)
            else:
                self.m.addConstr(quicksum(self.x[flight.index, slot.index] for slot in flight.compatibleSlots) == 1)

        for slot in self.slots:
            self.m.addConstr(quicksum(self.x[flight.index, slot.index] for flight in self.flights) <= 1)

        for flight in self.flights:
            for slot in flight.notCompatibleSlots:
                self.m.addConstr(self.x[flight.index, slot.index] == 0)

        for flight in self.flights_in_matches:
            self.m.addConstr(
                quicksum(self.x[flight.index, slot.index]
                       for slot in self.slots if slot != flight.slot) \
                <= quicksum([self.c[j] for j in self.get_match_for_flight(flight)]))

            self.m.addConstr(quicksum([self.c[j] for j in self.get_match_for_flight(flight)]) <= 1)

        k = 0
        for match in self.matches:
            flights = [flight for pair in match for flight in pair]
            self.m.addConstr(quicksum(quicksum(self.x[flightI.index, flightJ.slot.index] for flightI in pair for flightJ in flights)
                                        for pair in match) >= (self.c[k]) * len(flights))

            for pair in match:
                self.m.addConstr(
                    quicksum(self.x[flightI.index, flightJ.slot.index] * flightI.fitCostVect[flightJ.slot.index] for flightI in pair for flightJ in
                           flights) -
                    (1 - self.c[k]) * 10000000 \
                    <= quicksum(self.x[flightI.index, flightJ.slot.index] * flightI.fitCostVect[flightI.slot.index] for flightI in pair for flightJ in
                              flights) - \
                    self.epsilon)

            k += 1

    def set_objective(self):
        self.flights: List[IstopFlight]

        self.m.setObjective(
            quicksum(self.x[flight.index, j.index] * flight.fitCostVect[j.index]
                   for flight in self.flights for j in self.slots), sense=GRB.MINIMIZE)  # s

    def run(self, timing=False, verbose=False, time_limit=60):

        self.m._time_limit = time_limit
        if not verbose:
            self.m.setParam('OutputFlag', 0)

        self.set_variables()
        start = time.time()
        self.set_constraints()
        end = time.time() - start
        if timing:
            print("Constraints setting time ", end)

        self.set_objective()

        start = time.time()
        self.m.optimize(stop)
        end = time.time() - start

        if timing:
            print("Simplex time ", end)

        status = None
        if self.m.status == 2:
            status = "optimal"
        if self.m.status == 3:
            status = "infeasible"
        print('Problem status, value:', status, self.m.getObjective().getValue())

        # print("problem status, explained: ", self.m.getProbStatusString(), self.m.getObjVal())
        #print(self.m.getObjVal())




        # for flight in self.flights:
        #     if flight.eta > flight.newSlot.time:
        #         print("********************** danno *********************************",
        #               flight, flight.eta, flight.newSlot.time)


        return self.get_sol_array(), self.get_solution_offers()


    def get_sol_array(self):
        solution = np.zeros((len(self.flights), len(self.slots)))
        for flight in self.flights:
            for slot in self.slots:
                if self.x[flight.index, slot.index].x > 0.5:
                    solution[flight.index, slot.index] = 1
        return solution

    def get_solution_offers(self):
        solution = np.zeros(len(self.matches))
        for i in range(len(self.matches)):
            #if self.m[i].x > 0.5:
            if self.c[i].x > 0.5:
                solution[i] = 1
        return solution