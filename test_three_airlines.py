from Istop import istopThreeAirlines
from ModelStructure.ScheduleMaker import scheduleMaker

from ModelStructure.Costs.costFunctionDict import CostFuns
from UDPP import udppModel
import pandas as pd

# import matplotlib.pyplot as plt
import numpy as np

# df = pd.read_csv("../data/data_ruiz.csv")
np.random.seed(0)
scheduleType = scheduleMaker.schedule_types(show=True)

num_flights = 50
num_airlines = 5
# df = pd.read_csv("dfcrash")
# df = scheduleMaker.df_maker(50, 4, distribution=scheduleType[3])
df = scheduleMaker.df_maker(num_flights, num_airlines, distribution=scheduleType[3])
df.to_csv("three")
costFun = CostFuns().costFun["realistic"]
udpp_model_xp = udppModel.UDPPmodel(df, costFun)
udpp_model_xp.run(optimised=True)
print("done")

xpModel = istopThreeAirlines.IstopThree(udpp_model_xp.get_new_df(), costFun)
xpModel.run(True)
xpModel.print_performance()

# data.to_csv("50flights.csv")
# print(data)