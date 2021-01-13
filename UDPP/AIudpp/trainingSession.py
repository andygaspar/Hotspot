import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch.tensor

from UDPP import udppModel
from UDPP.AIudpp.trainAuxFuns1 import make_batch, make_network_input, make_prioritisation
from UDPP.AirlineAndFlightAndSlot.udppAirline import UDPPairline
from UDPP.AirlineAndFlightAndSlot.udppSlot import UDPPslot
from UDPP.Local.udppLocal import udpp_local
from UDPP.LocalOptimised.udppLocalOpt import UDPPlocalOpt
from UDPP.udppModel import UDPPmodel
from ScheduleMaker import scheduleMaker
from ModelStructure.modelStructure import ModelStructure
from ModelStructure.Costs.costFunctionDict import CostFuns
from UDPP.AIudpp import network as nn

batchSize = 100
net = nn.AirNetwork(24, batchSize)

df = pd.read_csv("Data/50_5_increase.csv")
dfA = df[df["airline"] == "A"]
for i in range(5000):
    batch_instance = np.random.choice(range(max(dfA["instance"])), size=batchSize, replace=False)
    selection = dfA[dfA["instance"].isin(batch_instance)]
    inputs = selection[["margins", "priority", "eta", "slot"]].values
    outputs = selection["new slot"].values
    inputs = inputs.reshape((int(inputs.shape[0]/6), 24))
    outputs = outputs.reshape((int(outputs.shape[0]/6), 6))
    net.train(inputs, outputs)
    print(i, net.loss)

net.save_weights()








