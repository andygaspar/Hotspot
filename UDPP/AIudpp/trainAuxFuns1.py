import random
import sys
import os
import numpy as np

from Programma.GlobalFuns.globalFuns import HiddenPrints
from Programma.ModelStructure.Costs.costFunctionDict import CostFuns
from Programma.UDPP import udppModel
from Programma.UDPP.AirlineAndFlightAndSlot.udppAirline import UDPPairline
from Programma.UDPP.Local.udppLocal import udpp_local
from Programma.UDPP.LocalOptimised.udppLocalOpt import UDPPlocalOpt
from data.dfMaker import df_maker


def run_UDPP_local(prior: np.array, air: UDPPairline, slots):
    for i in range(air.numFlights):
        air.flights[i].priorityNumber = prior[i]
    udpp_local(air, slots)


def make_network_input(air: UDPPairline):
    vals = [[f.tna/100, f.eta/100, f.slot.time/100, f.cost] for f in air.flights]
    return [item for val in vals for item in val]


def make_network_output(air: UDPPairline):
    vals = [[0 if f.priorityValue is "P" else 1, f.newSlot.time/100] for f in air.flights]
    return [item for val in vals for item in val]


def make_batch(batchSize):
    inputs = []
    outputs = []
    initialCosts = []
    airlines = []
    UDPPmodels = []

    for run in range(batchSize):
        df = df_maker(custom=[6, 4, 3, 7, 2, 8])
        df["margins"] = [random.choice(range(10, 50)) for i in range(df.shape[0])]
        costFun = CostFuns().costFun["linear"]

        udMod = udppModel.UDPPmodel(df, costFun)

        UDPPmodels.append(udMod)

        airline = [air for air in udMod.airlines if air.name == "A"][0]
        airlines.append(airline)

        initialCosts.append(udMod.compute_costs(airline.flights, "initial"))

        with HiddenPrints():
            UDPPlocalOpt(airline, udMod.slots)

        inputVect = make_network_input(airline)
        outputVect = make_network_output(airline)
        inputs.append(inputVect)
        outputs.append(outputVect)

    return inputs, outputs, airlines, UDPPmodels


def make_prioritisation(inputs: np.array):
    prValues = []
    times = []
    for i in range(len(inputs)):
        if i % 2 == 0:
            prValues.append("P" if inputs[i] < 0.5 else "N")
        else:
            times.append(int(round(inputs[i]*100)))

    return prValues, times
