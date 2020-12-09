import random

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from IPython import display
import torch.tensor

from Programma.UDPP import udppModel
from Programma.UDPP.AIudpp.trainAuxFuns1 import make_batch, make_network_input, make_prioritisation
from Programma.UDPP.AirlineAndFlightAndSlot.udppAirline import UDPPairline
from Programma.UDPP.AirlineAndFlightAndSlot.udppSlot import UDPPslot
from Programma.UDPP.Local.udppLocal import udpp_local
from Programma.UDPP.LocalOptimised.udppLocalOpt import UDPPlocalOpt
from Programma.UDPP.udppModel import UDPPmodel
from data import dfMaker
from data.dfMaker import df_maker
from Programma.ModelStructure.modelStructure import ModelStructure
from Programma.ModelStructure.Costs.costFunctionDict import CostFuns
from Programma.UDPP.AIudpp import network1 as nn



# df = pd.read_csv("../data/data_ruiz.csv")
scheduleType = dfMaker.schedule_types(show=True)
# df = pd.read_csv("dfcrash")
df = df_maker(custom=[6, 4, 3, 7, 2, 8])
df["margins"] = [random.choice(range(10, 50)) for i in range(df.shape[0])]
# df.to_csv("dfcrash")

costFun = CostFuns().costFun["step"]
udMod = UDPPmodel(df, costFun)


airline: UDPPairline
airline = [air for air in udMod.airlines if air.name == "A"][0]
batchSize = 500

net = nn.AirNetwork(24, batchSize)


for i in range(1000):
    inputs, outputs, airlines, UDPPmodels = make_batch(batchSize)
    net.train(6, batchSize, inputs, outputs, airlines, UDPPmodels)
    print(i, net.loss*1000)


net.save_weights()

udMod.run(optimised=True)


output = net.prioritisation(make_network_input(airline))
prValues, times = make_prioritisation(output)
i = 0
for f in airline.flights:
    print(f.priorityValue, f.newSlot.time, prValues[i], times[i])
    i += 1




