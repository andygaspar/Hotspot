import random

import numpy as np

from Programma.ModelStructure.Costs.costFunctionDict import CostFuns
from Programma.UDPP import udppModel
from Programma.UDPP.AirlineAndFlightAndSlot.udppAirline import UDPPairline
from Programma.UDPP.Local.udppLocal import udpp_local
from data.dfMaker import df_maker


def run_UDPP_local(prior: np.array, air: UDPPairline, slots):
    for i in range(air.numFlights):
        air.flights[i].priorityNumber = prior[i]
    udpp_local(air, slots)


def make_network_input(air: UDPPairline):
    vals = [[f.tna, f.eta, f.slot.time, f.cost] for f in air.flights]
    return [item for val in vals for item in val]


def make_batch(batchSize):
    inputs = []
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

        inputVect = make_network_input(airline)
        inputs.append(inputVect)

    return inputs, initialCosts, airlines, UDPPmodels
