# from mip import *
from typing import List
import numpy as np
from UDPP.AirlineAndFlightAndSlot import udppAirline as air
from UDPP.AirlineAndFlightAndSlot import udppFlight as fl
from ModelStructure.Slot import slot as sl
import xpress as xp
xp.controls.outputlog = 0

import ModelStructure.modelStructure as ms


def slot_range(k: int, AUslots: List[sl.Slot]):
    return range(AUslots[k].index + 1, AUslots[k + 1].index)


def eta_limit_slot(flight: fl.UDPPflight, AUslots: List[sl.Slot]):
    i = 0
    for slot in AUslots:
        if slot >= flight.etaSlot:
            return i
        i += 1


def UDPPlocalOpt(airline: air.UDPPairline, slots: List[sl.Slot]):

    m = xp.problem()

    x = np.array([[xp.var(vartype=xp.binary) for j in slots] for i in airline.flights])

    z = np.array([xp.var(vartype=xp.integer) for i in airline.flights])

    y = np.array([[xp.var(vartype=xp.binary) for j in slots] for i in airline.flights])

    m.addVariable(x, z, y)

    flight: fl.Flight

    m.addConstraint(
        xp.Sum(x[0, k] for k in range(airline.numFlights)) == 1
    )

    # slot constraint
    for j in slots:
        #one y max for slot
        m.addConstraint(
            xp.Sum(y[flight.localNum, j.index] for flight in airline.flights) <= 1
        )

    for k in range(airline.numFlights - 1):
        #one x max for slot
        m.addConstraint(
            xp.Sum(x[flight.localNum, k] for flight in airline.flights) <= 1
        )


        m.addConstraint(
            xp.Sum(y[flight.localNum, airline.AUslots[k].index] for flight in airline.flights) == 0
        )

        m.addConstraint(
            xp.Sum(y[i, j] for i in range(k, airline.numFlights) for j in range(airline.AUslots[k].index)) <= \
             xp.Sum(x[i, kk] for i in range(k + 1) for kk in range(k, airline.numFlights))
        )

        m.addConstraint(
            xp.Sum(y[flight.localNum, j] for flight in airline.flights for j in slot_range(k, airline.AUslots)) \
             == z[k]
        )

        m.addConstraint(
            xp.Sum(y[flight.localNum, j] for flight in airline.flights for j in range(airline.AUslots[k].index)) <= \
             xp.Sum(x[i, j] for i in range(k) for j in range(k, airline.numFlights))
        )

        for i in range(k + 1):
            m.addConstraint(
                (1 - xp.Sum(x[flight.localNum, i] for flight in airline.flights)) * 1000 \
                 >= z[k] - (k - i)
            )

    # last slot
    m.addConstraint(
        xp.Sum(x[flight.localNum, airline.numFlights - 1] for flight in airline.flights) == 1
    )

    for flight in airline.flights:
        m.addConstraint(
            xp.Sum(y[flight.localNum, j] for j in range(flight.etaSlot.index)) == 0
        )

    for flight in airline.flights[1:]:
        # flight assignment
        m.addConstraint(
            xp.Sum(y[flight.localNum, j] for j in range(flight.etaSlot.index, flight.slot.index)) + \
            xp.Sum(x[flight.localNum, k] for k in
                  range(eta_limit_slot(flight, airline.AUslots), airline.numFlights)) == 1
        )

    # not earlier than its first flight
    m.addConstraint(
        xp.Sum(y[flight.localNum, j] for flight in airline.flights for j in range(airline.flights[0].slot.index)) == 0
    )

    m.setObjective(
            xp.Sum(y[flight.localNum][slot.index] * flight.costFun(flight, slot)
             for flight in airline.flights for slot in slots) +
            xp.Sum(x[flight.localNum][k] * flight.costFun(flight, airline.AUslots[k])
             for flight in airline.flights for k in range(airline.numFlights))
    )

    m.solve()
    print("airline ",airline)
    for flight in airline.flights:

        for k in range(airline.numFlights):
            if m.getSolution(x[flight.localNum, k]) > 0.5:
                flight.newSlot = airline.flights[k].slot
                flight.priorityNumber = k
                flight.priorityValue = "N"
                print(flight.slot, flight.newSlot)



        for slot in slots:
            if m.getSolution(y[flight.localNum, slot.index]) > 0.5:
                flight.newSlot = slot
                flight.priorityNumber = slot.time
                flight.priorityValue = "P"
                print(flight.slot, flight.newSlot, "P")

    print(sum([flight.costFun(flight, flight.slot) for flight in airline.flights]),
          sum([flight.costFun(flight, flight.newSlot) for flight in airline.flights]))
