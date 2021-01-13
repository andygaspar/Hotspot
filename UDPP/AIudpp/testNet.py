import numpy as np
import pandas as pd

from UDPP.udppModel import UDPPmodel
from ModelStructure.ScheduleMaker import scheduleMaker
from ModelStructure.Costs.costFunctionDict import CostFuns
from UDPP.AIudpp import network as nn

max_num_flights = 12

costFun = CostFuns().costFun["step"]
net = nn.AirNetwork(20*4, 20, 100)
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
    non_zero_inputs = np.array([[(fl.tna - mean_margins) / std_margins,
               (fl.priority - mean_priority) / std_priority,
                 (fl.eta - mean_eta) / std_eta,
               (fl.slot.time - mean_slot) / std_slot] for fl in airline.flights])
    return np.append(non_zero_inputs, np.zeros((max_num_flights - non_zero_inputs.shape[0], 4)), axis=0)



# for i in range(10):
#     selection = dfA[dfA["instance"] == i]
#     inputs = selection[["margins", "priority", "eta", "slot"]].values
#     outputs = selection["new slot"].values
#     inputs = inputs.reshape((int(inputs.shape[0] / 6), 24))
#     outputs = outputs.reshape((int(outputs.shape[0]/6), 6))[0]
#     predictions = net.prioritisation(inputs)
#     for j in range(6):
#         print(outputs[j], predictions[j])
#     print("\n\n")

for i in range(10):
    print("\n\n\n", "run ", i)
    df = scheduleMaker.df_maker(custom=[6, 4, 3, 7, 2, 8])
    udMod = UDPPmodel(df, costFun)
    for airline in udMod.airlines:
        inputs = make_net_input(airline)
        predictions = net.prioritisation(inputs)
        j = 0
        for f in airline.flights:
            f.priorityNumber = predictions[j]
            print(f.slot, predictions[j])
            j += 1
        print("\n")
    print("\n")
    udMod.run(optimised=False)
    udMod.print_performance()
    print(udMod.solution[udMod.solution["airline"] == "A"])
