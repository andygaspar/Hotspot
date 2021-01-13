import numpy as np

from UDPP.AIudpp.trainAuxFuns1 import make_network_input, make_prioritisation
from UDPP.udppModel import UDPPmodel
from ModelStructure.ScheduleMaker import scheduleMaker
from ModelStructure.Costs.costFunctionDict import CostFuns
from UDPP.AIudpp import network as nn



# df = pd.read_csv("../data/data_ruiz.csv")
scheduleType = scheduleMaker.schedule_types(show=True)
# df = pd.read_csv("dfcrash")
# df = scheduleMaker.df_maker(custom=[6, 4, 3, 7, 2, 8])
# df["margins"] = [random.choice(range(10, 50)) for i in range(df.shape[0])]
# # df.to_csv("dfcrash")
#
costFun = CostFuns().costFun["step"]
# udMod = UDPPmodel(df, costFun)
#
#
# airline: UDPPairline
# airline = [air for air in udMod.airlines if air.name == "A"][0]
batchSize = 200
#
net = nn.AirNetwork(24, batchSize)
#
# for i in range(100):
#     inputs, outputs, airlines, UDPPmodels = make_batch(batchSize)
#     net.train(6, batchSize, inputs, outputs, airlines, UDPPmodels)
#     print(i, net.loss*1000)
#
#
# net.save_weights()

# udMod.run(optimised=True)
# udMod.print_performance()
#
#
# output = net.prioritisation(make_network_input(airline))
# prValues, times = make_prioritisation(output)
# i = 0
# for f in airline.flights:
#     print(f.slot, f.priorityValue, f.newSlot.time, prValues[i], times[i])
#     i += 1

net.load_weights("netWeights.pt")


for i in range(10):
    print("\n\n\n", "run ", i)
    df = scheduleMaker.df_maker(custom=[6, 4, 3, 7, 2, 8])
    df["margins"] = [np.random.choice(range(10, 50)) for i in range(df.shape[0])]
    df1 = df.copy(deep=True)
    udMod = UDPPmodel(df, costFun)
    udMod.run(optimised=True)
    udMod.print_performance()
    print(udMod.solution[udMod.solution["airline"] == "A"])
    airline_opt = [air for air in udMod.airlines if air.name == "A"][0]

    udMod = UDPPmodel(df1, costFun)
    airline = [air for air in udMod.airlines if air.name == "A"][0]
    output = net.prioritisation(make_network_input(airline))
    prValues, times = make_prioritisation(output)
    print(airline_opt.numFlights, airline.numFlights, len(prValues))
    j = 0
    k = 0
    # for f in airline_opt.flights:
    #     print(f.slot, f.priorityValue, f.newSlot.time, prValues[k], times[k])
    #     k += 1
    for f in airline.flights:
        f.priorityNumber = times[j]
        j += 1
    prValues, times = make_prioritisation(output)
    udMod.run(optimised=False)
    udMod.print_performance()
    print(udMod.solution[udMod.solution["airline"] == "A"])

