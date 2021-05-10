import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from IPython import display
import torch.tensor

from Programma.UDPP import udppModel
from Programma.UDPP.AIudpp.trainAuxFuns1 import make_batch, run_UDPP_local
from Programma.UDPP.AirlineAndFlightAndSlot.udppAirline import UDPPairline as Airline
from Programma.UDPP.AirlineAndFlightAndSlot.udppSlot import UDPPslot
from Programma.UDPP.Local.udppLocal import udpp_local
from Programma.UDPP.LocalOptimised.udppLocalOpt import UDPPlocalOpt
from data.dfMaker import df_maker
from Programma.ModelStructure.modelStructure import ModelStructure
from Programma.ModelStructure.Costs.costFunctionDict import CostFuns
from Programma.UDPP.AIudpp import network1 as nn





airline: Airline

batchSize = 100

net = nn.AirNetwork(24, batchSize)

for i in range(50):
    inputs, initialCosts, airlines, UDPPmodels = make_batch(batchSize)
    net.train(6, batchSize, inputs, np.array(initialCosts), airlines, UDPPmodels)
    print(i, net.loss)




for i in range(10):
    inputs, initialCosts, airlines, UDPPmodels = make_batch(1)
    udpp_local(airlines[0],UDPPmodels[0].slots)
    print(UDPPmodels[0].compute_costs(airlines[0].flights, "final"), "  base")

    UDPPlocalOpt(airlines[0], UDPPmodels[0].slots)
    for flight in airlines[0].flights:
        flight.newSlot = flight.UDPPlocalSolution
    print(UDPPmodels[0].compute_costs(airlines[0].flights, "final"), "  opt")

    run_UDPP_local(net.prioritisation(inputs), airlines[0], UDPPmodels[0])
    print(UDPPmodels[0].compute_costs(airlines[0].flights, "final"), "   nn\n\n")


