from Istop import istop
from ModelStructure.ScheduleMaker import scheduleMaker
from ModelStructure.Costs.costFunctionDict import CostFuns
from NNBound import nnBound
from UDPP import udppModel
import pandas as pd

import numpy as np

np.random.seed(0)
scheduleType = scheduleMaker.schedule_types(show=False)

num_flights = 40
num_airlines = 4
distribution = scheduleType[0]
print("schedule type: ", distribution)

df = scheduleMaker.df_maker(num_flights, num_airlines, distribution=distribution)
#df = pd.read_csv("0offers.csv")
costFun = CostFuns().costFun["realistic"]

print("\nnn bound")
max_model = nnBound.NNBoundModel(df, costFun)
max_model.run()
max_model.print_performance()

print("\nudpp")
udpp_model_xp = udppModel.UDPPmodel(df, costFun)
udpp_model_xp.run(optimised=True)
udpp_model_xp.print_performance()


print("\nistop only pairs")
xpModel = istop.Istop(udpp_model_xp.get_new_df(), costFun, triples=False)

xpModel.run(True)
# print(xpModel.matches)
xpModel.print_performance()
# print(xpModel.offers_selected)


print("\nistop with triples")
xpModel = istop.Istop(udpp_model_xp.get_new_df(), costFun, triples=True)
xpModel.run(True)
# print(xpModel.matches)
xpModel.print_performance()
# print(xpModel.offers_selected)

