# from mip import *
from typing import List
import numpy as np

from ...ModelStructure.Airline.airline import Airline
from ...ModelStructure.Flight.flight import Flight
from ...ModelStructure.Slot import slot as sl
import mip


def slot_range(k: int, AUslots: List[sl.Slot]):
    return range(AUslots[k].index + 1, AUslots[k + 1].index)


def eta_limit_slot(flight: Flight, AUslots: List[sl.Slot]):
    i = 0
    for slot in AUslots:
        if slot >= flight.etaSlot:
            return i
        i += 1


def UDPPlocalOptMIP(airline: Airline, slots: List[sl.Slot]):

    m = mip.Model()
    m.verbose = False
    m.threads = -1

    # Variables corresponding to N priorities (each flight, each slot)
    x = np.array([[m.add_var(var_type=mip.BINARY) for _ in slots] for _ in airline.flights])

    z = np.array([m.add_var(var_type=mip.INTEGER) for _ in airline.flights])

    # Variables corresponding to P priorities (each flight, each slot)
    y = np.array([[m.add_var(var_type=mip.BINARY) for _ in slots] for _ in airline.flights])

    flight: Flight

    # First flight has to have a slot allocated
    m += mip.xsum(x[0, k] for k in range(airline.numFlights)) == 1

    # slot constraint
    for j in slots:
        # one y max for slot (one priority P max per slot)
        m += mip.xsum(y[flight.localNum, j.index] for flight in airline.flights) <= 1

    for k in range(airline.numFlights - 1):
        # one x max for slot (one priority N max per slot)
        m += mip.xsum(x[flight.localNum, k] for flight in airline.flights) <= 1

        m += mip.xsum(y[flight.localNum, airline.AUslots[k].index] for flight in airline.flights) == 0

        m += mip.xsum(y[i, j] for i in range(k, airline.numFlights) for j in range(airline.AUslots[k].index)) <= \
             mip.xsum(x[i, kk] for i in range(k + 1) for kk in range(k, airline.numFlights))

        m += mip.xsum(y[flight.localNum, j] for flight in airline.flights for j in slot_range(k, airline.AUslots)) \
             == z[k]

        m += mip.xsum(
            y[flight.localNum, j] for flight in airline.flights for j in range(airline.AUslots[k].index)) <= \
             mip.xsum(x[i, j] for i in range(k) for j in range(k, airline.numFlights))

        for i in range(k + 1):
            m += (1 - mip.xsum(x[flight.localNum, i] for flight in airline.flights)) * 1000 >= \
                 z[k] - (k - i)

    # last slot
    m += mip.xsum(x[flight.localNum, airline.numFlights - 1] for flight in airline.flights) == 1

    for flight in airline.flights:
        m += mip.xsum(y[flight.localNum, j] for j in range(flight.etaSlot.index)) == 0

    for flight in airline.flights[1:]:
        # flight assignment
        m += mip.xsum(y[flight.localNum, j] for j in range(flight.etaSlot.index, flight.slot.index)) + \
             mip.xsum(x[flight.localNum, k] for k in
                      range(eta_limit_slot(flight, airline.AUslots), airline.numFlights)) == 1

    # not earlier than its first flight
    m += mip.xsum(y[flight.localNum, j] for flight in airline.flights for j in range(airline.flights[0].slot.index)) \
         == 0

    m.objective = mip.minimize(mip.xsum(y[flight.localNum][slot.index] * flight.cost_fun(slot)
                                        for flight in airline.flights for slot in slots)
                               + mip.xsum(x[flight.localNum][k] * flight.cost_fun(airline.AUslots[k])
                                          for flight in airline.flights for k in range(airline.numFlights)))

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



