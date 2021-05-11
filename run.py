from GlobalOptimum import globalOptimum
from Istop import istop
from NNBound import nnBound
from ScheduleMaker import scheduleMaker, df_to_schedule
from ModelStructure.Costs.costFunctionDict import CostFuns
import numpy as np
import random
from UDPP import udppModel

print()


# ************* init or conversion from other models
random.seed(0)
np.random.seed(0)
scheduleType = scheduleMaker.schedule_types(show=False)

num_flights = 50
num_airlines = 5


distribution = scheduleType[3]
print("schedule type: ", distribution)

df = scheduleMaker.df_maker(num_flights, num_airlines, distribution=distribution)
costFun = CostFuns()

fl_list = df_to_schedule.make_flight_list(df)

# ****************
print("\n global optimum")
global_model = globalOptimum.GlobalOptimum(fl_list)
global_model.run()
global_model.print_performance()


print("\nnn bound")
max_model = nnBound.NNBoundModel(fl_list)
max_model.run()
max_model.print_performance()


print("\nudpp")
udpp_model_xp = udppModel.UDPPmodel(fl_list)
udpp_model_xp.run(optimised=True)
udpp_model_xp.print_performance()
# print(udpp_model_xp.get_new_df())

new_fl_list = udpp_model_xp.get_new_flight_list()
print(new_fl_list)

print("\nistop only pairs")
xpModel = istop.Istop(new_fl_list, triples=False)
xpModel.run(True)
xpModel.print_performance()
print(xpModel.offers_selected)


print("\nistop with triples")
xpModel = istop.Istop(new_fl_list, triples=True)
xpModel.run(True)
xpModel.print_performance()

"""
TO CONSIDER - TO DO

-slot holes due to multiple hotspot
-cost function definition

"""