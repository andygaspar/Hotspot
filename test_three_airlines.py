from Istop import istopThreeAirlines, istop
from ModelStructure.ScheduleMaker import scheduleMaker

from ModelStructure.Costs.costFunctionDict import CostFuns
from UDPP import udppModel
import pandas as pd

# import matplotlib.pyplot as plt
import numpy as np

# df = pd.read_csv("../data/data_ruiz.csv")
np.random.seed(0)
scheduleType = scheduleMaker.schedule_types(show=True)

num_flights = 40
num_airlines = 5

df = scheduleMaker.df_maker(num_flights, num_airlines, distribution=scheduleType[3])
# df.to_csv("three.csv")
# df = pd.read_csv("three.csv")
costFun = CostFuns().costFun["realistic"]
udpp_model_xp = udppModel.UDPPmodel(df, costFun)
udpp_model_xp.run(optimised=True)
udpp_model_xp.print_performance()
print("done")

xpModel = istopThreeAirlines.IstopThree(udpp_model_xp.get_new_df(), costFun)
xpModel.run(True)
xpModel.print_performance()


print(" previous istop")
xpModel_old = istop.Istop(udpp_model_xp.get_new_df(), costFun)
xpModel_old.run(True)
xpModel_old.print_performance()

# for i in range(len(xpModel.flights)):
#     print(xpModel.flights[i], xpModel.flights[i].newSlot, xpModel_old.flights[i].newSlot, xpModel_old.flights[i])

# data.to_csv("50flights.csv")
# print(data)