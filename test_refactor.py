from Istop import istop
from ScheduleMaker import scheduleMaker, df_to_schedule as converter
from ModelStructure.Costs.costFunctionDict import CostFuns
import random

import numpy as np

from UDPP import udppModel

random.seed(0)
np.random.seed(0)
scheduleType = scheduleMaker.schedule_types(show=False)

num_flights = 50
num_airlines = 5


distribution = scheduleType[3]
print("schedule type: ", distribution)

df = scheduleMaker.df_maker(num_flights, num_airlines, distribution=distribution)

costFun = CostFuns().costFun["realistic"]

fl_list = converter.make_flight_list(df, costFun)

# m = ModelStructure(fl_list)
#
# print(m.df)

# print("\nnn bound")
# max_model = nnBound.NNBoundModel(fl_list)
# max_model.run()
# max_model.print_performance()


print("\nudpp")
udpp_model_xp = udppModel.UDPPmodel(fl_list)
udpp_model_xp.run(optimised=True)
udpp_model_xp.print_performance()
print(udpp_model_xp.get_new_df())

new_fl_list = udpp_model_xp.get_new_flight_list()
print(new_fl_list)

print("\nistop only pairs")
xpModel = istop.Istop(new_fl_list, triples=False)
xpModel.run(True)
xpModel.print_performance()
print(xpModel.offers_selected)

"""
TO CONSIDER - TO DO

-slot holes due to multiple hotspot
-cost function definition

"""