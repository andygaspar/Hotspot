import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch.tensor

from UDPP import udppModel
from UDPP.AirlineAndFlightAndSlot.udppAirline import UDPPairline
from UDPP.AirlineAndFlightAndSlot.udppSlot import UDPPslot
from UDPP.Local.udppLocal import udpp_local
from UDPP.LocalOptimised.udppLocalOpt import UDPPlocalOpt
from UDPP.udppModel import UDPPmodel
from ScheduleMaker import scheduleMaker
from ModelStructure.modelStructure import ModelStructure
from ModelStructure.Costs.costFunctionDict import CostFuns
from UDPP.AIudpp import network as nn

costFun = CostFuns().costFun["step"]
net = nn.AirNetwork(24, 5)
net.load_weights("netWeights.pt")
df = pd.read_csv("Data/50_5_increase.csv")
dfA = df[df["airline"] == "A"]

mean_margins = df["margins"].mean()
std_margins = df["margins"].std()
mean_priority = df["priority"].mean()
std_priority = df["priority"].std()
mean_eta = df["eta"].mean()
std_eta = df["eta"].std()
mean_slot = df["slot"].mean()
std_slot = df["slot"].std()


def make_net_input(airline):

    inputs = np.array([[(fl.tna - mean_margins) / std_margins,
               (fl.priority - mean_priority) / std_priority,
                 (fl.eta - mean_eta) / std_eta,
               (fl.slot.time - mean_slot) / std_slot] for fl in airline.flights])

    return inputs.reshape((int(inputs.shape[0] / 6), 24))


for i in range(10):
    selection = dfA[dfA["instance"] == i]
    inputs = selection[["margins", "priority", "eta", "slot"]].values
    outputs = selection["new slot"].values
    inputs = inputs.reshape((int(inputs.shape[0] / 6), 24))
    outputs = outputs.reshape((int(outputs.shape[0]/6), 6))[0]
    predictions = net.prioritisation(inputs)
    for j in range(6):
        print(outputs[j], predictions[j])
    print("\n\n")

for i in range(10):
    print("\n\n\n", "run ", i)
    df = scheduleMaker.df_maker(custom=[6, 4, 3, 7, 2, 8])
    df["margins"] = [np.random.choice(range(10, 50)) for i in range(df.shape[0])]
    udMod = UDPPmodel(df, costFun)
    airline = [air for air in udMod.airlines if air.name == "A"][0]
    inputs = make_net_input(airline)
    predictions = net.prioritisation(inputs)
    j = 0
    for f in airline.flights:
        f.priorityNumber = predictions[j]
        j += 1
    udMod.run(optimised=False)
    udMod.print_performance()
    print(udMod.solution[udMod.solution["airline"] == "A"])
