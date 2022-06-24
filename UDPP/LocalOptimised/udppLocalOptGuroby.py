# from mip import *
from typing import List
import numpy as np

from ...ModelStructure.Airline.airline import Airline
from ...ModelStructure.Flight.flight import Flight
from ...ModelStructure.Slot import slot as sl
from gurobipy import Model, GRB, quicksum, Env
def slot_range(k: int, AUslots: List[sl.Slot]):
    return range(AUslots[k].index + 1, AUslots[k + 1].index)


def eta_limit_slot(flight: Flight, AUslots: List[sl.Slot]):
    i = 0
    for slot in AUslots:
        if slot >= flight.etaSlot:
            return i
        i += 1


def get_num_flights_for_eta(flight: Flight, airline: Airline):
    return sum([1 for fl in airline.flights if fl.etaSlot.time == flight.etaSlot.time])


def UDPPlocalOptGurobi(airline: Airline, slots: List[sl.Slot]):
    m = Model('CVRP')
    m.setParam('OutputFlag', 0)
    x = m.addVars([(i, j) for i in range(airline.numFlights) for j in range(airline.numFlights)], vtype=GRB.BINARY)
    y = m.addVars([(i, j) for i in range(airline.numFlights) for j in range(len(slots))], vtype=GRB.BINARY)
    z = m.addVars([i for i in range(airline.numFlights)], vtype=GRB.BINARY)

    m.modelSense = GRB.MINIMIZE

    flight: Flight

    # slot constraint
    m.addConstr(
        quicksum(x[0, k] for k in range(airline.numFlights)) == 1
    )

    # slot constraint
    for j in slots:
        # one y max for slot
        m.addConstr(
            quicksum(y[flight.localNum, j.index] for flight in airline.flights) <= 1
        )

    for flight in airline.flights:
        # m.addConstr(
        #     [y[flight.localNum, j] == 0 for j in range(flight.etaSlot.index + airline.numFlights - flight.localNum, len(slots))]
        # )

        eta_index = flight.etaSlot.index
        end_index = eta_index + get_num_flights_for_eta(flight, airline)
        m.addConstrs(
            (y[flight.localNum, slot.index] == 0 for slot in slots if slot.index not in range(eta_index, end_index))
        )

    for k in range(airline.numFlights - 1):
        # one x max for slot
        m.addConstr(
            quicksum(x[flight.localNum, k] for flight in airline.flights) <= 1
        )

        m.addConstrs(
            (y[flight.localNum, airline.AUslots[k].index] == 0 for flight in airline.flights)
        )

        m.addConstr(
            quicksum(y[i, j] for i in range(k, airline.numFlights) for j in range(airline.AUslots[k].index)) <= \
            quicksum(x[i, kk] for i in range(k + 1) for kk in range(k, airline.numFlights))
        )

        m.addConstr(
            quicksum(y[flight.localNum, j] for flight in airline.flights for j in slot_range(k, airline.AUslots)) \
            == z[k]
        )

        m.addConstr(
            quicksum(y[flight.localNum, j] for flight in airline.flights for j in range(airline.AUslots[k].index)) <= \
            quicksum(x[i, j] for i in range(k) for j in range(k, airline.numFlights))
        )

        for i in range(k + 1):
            m.addConstr(
                (1 - quicksum(x[flight.localNum, i] for flight in airline.flights)) * 1000 \
                >= z[k] - (k - i)
            )
    # last slot
    m.addConstr(
        quicksum(x[flight.localNum, airline.numFlights - 1] for flight in airline.flights) == 1
    )

    for flight in airline.flights:
        m.addConstrs(
            (y[flight.localNum, j] == 0 for j in range(flight.etaSlot.index))
        )

    for flight in airline.flights[1:]:
        # flight assignment
        m.addConstr(
            quicksum(y[flight.localNum, j] for j in range(flight.etaSlot.index, flight.slot.index)) + \
            quicksum(x[flight.localNum, k] for k in
                   range(eta_limit_slot(flight, airline.AUslots), airline.numFlights)) == 1
        )

    # not earlier than its first flight
    m.addConstrs(
        (y[flight.localNum, j] == 0 for flight in airline.flights for j in range(airline.flights[0].slot.index))
    )

    m.setObjective(
        quicksum(y[flight.localNum, slot.index] * flight.cost_fun(slot)
               for flight in airline.flights for slot in slots) +
        quicksum(x[flight.localNum, k] * flight.cost_fun(airline.AUslots[k])
               for flight in airline.flights for k in range(airline.numFlights))
    )

    m.optimize()
    n_flights = []
    for flight in airline.flights:

        for slot in slots:
            if y[flight.localNum, slot.index].x > 0.5:
                flight.newSlot = slot
                flight.udppPriority = "P"
                flight.tna = slot.time

        for k in range(airline.numFlights):
            if x[flight.localNum, k].x > 0.5:
                flight.newSlot = airline.flights[k].slot
                flight.udppPriority = "N"
                flight.udppPriorityNumber = k
                n_flights.append(flight)

    n_flights.sort(key=lambda f: f.udppPriorityNumber)
    for i in range(len(n_flights)):
        n_flights[i].udppPriorityNumber = i
