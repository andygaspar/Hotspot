import random

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

batchSize = 200
max_num_flights = 12
net_input_size = max_num_flights * 4

net = nn.AirNetwork(net_input_size,max_num_flights, batchSize)

df = pd.read_csv("Data/50_5_increase.csv")


for i in range(1500):
    batch_instance = np.random.choice(range(max(df["instance"])), size=batchSize, replace=False)
    instance_selection = df[df["instance"].isin(batch_instance)]
    inputs = np.empty((0,4))
    outputs = np.array([])
    for inst in batch_instance:
        for airline in instance_selection["airline"].unique():
            selection = instance_selection[(instance_selection["airline"] == airline) &
                                           (instance_selection["instance"] == inst)]

            non_zeros_inputs = selection[["margins", "priority", "eta", "slot"]].values
            inputs_to_append = np.append(non_zeros_inputs, np.zeros((max_num_flights-non_zeros_inputs.shape[0], 4)), axis=0)
            inputs = np.append(inputs, inputs_to_append, axis=0)

            non_zeros_outputs = selection["new slot"].values
            outputs_to_append = np.append(non_zeros_outputs, np.zeros(max_num_flights-non_zeros_outputs.shape[0]))
            outputs = np.append(outputs, outputs_to_append)

    inputs = inputs.reshape((int(inputs.shape[0]/max_num_flights), net_input_size))
    outputs = outputs.reshape((int(outputs.shape[0]/max_num_flights), max_num_flights))

    net.train(inputs, outputs)
    print(i, net.loss)

net.save_weights()








