import copy
import pandas as pd

from ModelStructure.Costs.costFunctionDict import CostFuns
from ScheduleMaker import scheduleMaker
from UDPP.AirlineAndFlightAndSlot.udppAirline import UDPPairline
from UDPP.AirlineAndFlightAndSlot.udppSlot import UDPPslot
from UDPP.AirlineAndFlightAndSlot.udppFlight import UDPPflight
import xpress as xp
import numpy as np
from UDPP.udppModel import UDPPmodel

def temp_costs(airline: UDPPairline):
    return sum([])

def make_prioritisation(airline: UDPPairline):

    slot: UDPPslot
    slots = copy.deepcopy(airline.AUslots)

    #reset indexes
    i = 0
    for slot in slots:
        slot.index = i
        i += 1
    m = xp.problem()

    x = np.array([[xp.var(vartype=xp.binary) for j in slots] for i in airline.flights])

    m.addVariable(x)

    flight: UDPPflight


    for slot in slots:
        m.addConstraint(
            xp.Sum(x[flight.localNum, slot.index] for flight in airline.flights) == 1
        )

    for flight in airline.flights:
        m.addConstraint(
            xp.Sum(x[flight.localNum, slot.index] for slot in slots) <= 1
        )
        m.addConstraint(
            xp.Sum(x[flight.localNum, slot.index] for slot in slots if slot.time < flight.eta) == 0
        )

    # m.setObjective(
    #     xp.Sum(x[flight.localNum][slot.index] * flight.costFun(flight, slot)
    #            for flight in airline.flights for slot in slots)
    # )

    m.setObjective(
        # xp.Sum(x[flight.localNum][slot.index] * (flight.tna - slot.time)
        #        for flight in airline.flights for slot in slots)
        xp.Sum(x[flight.localNum][slot.index] * flight.costFun(flight, slot)
                          for flight in airline.flights for slot in slots)
    )

    m.solve()
    print(m.getObjVal())


    new_slots = []

    for flight in airline.flights:

        for slot in slots:
            if m.getSolution(x[flight.localNum, slot.index]) > 0.5:
                new_slots.append(slot)
                flight.priorityNumber = slot.index
                flight.pre_solution_slot = slot

    # for flight in airline.flights:
    #     if flight.pre_solution_slot.time > flight.tna:
    #
    #         print("**************************  ", flight.slot.time, flight.pre_solution_slot.time, flight.tna)


for i in range(1):
    df_UDPP = scheduleMaker.df_maker(custom=[9,4,5,7,9,4])
    costFun = CostFuns().costFun["step"]

    udppMod = UDPPmodel(df_UDPP, costFun)

    for airline in udppMod.airlines:
        make_prioritisation(airline)

    udppMod.set_priority_value("N")

    udppMod.run(optimised=False)
    udppMod.print_performance()


    udppMod.set_priority_value("M")

    udppMod.run(optimised=False)
    udppMod.print_performance()
    print(udppMod.report["reduction %"].values)

    # airline = [airline for airline in udppMod.airlines if airline.name == "A"][0]
    # for f in airline.flights:
    #     print(f, f.slot.time, f.newSlot.time)

    udppMod.change_CCS(3)
    udppMod.set_priority_value("N")

    udppMod.run(optimised=False)
    udppMod.print_performance()


    udppMod.set_priority_value("M")

    udppMod.run(optimised=False)
    udppMod.print_performance()