from GlobalOptimum import globalOptimum
from Istop import istop
from NNBound import nnBound
from ScheduleMaker import scheduleMaker, df_to_schedule
from ModelStructure.Costs.costFunctionDict import CostFuns
import numpy as np
import random
from UDPP import udppModel




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

slot_list, fl_list = df_to_schedule.make_flight_list(df)




# **************** models run
print("\n global optimum")
global_model = globalOptimum.GlobalOptimum(slot_list, fl_list)
global_model.run()
global_model.print_performance()

print("\nnn bound")
max_model = nnBound.NNBoundModel(slot_list, fl_list)
max_model.run()
max_model.print_performance()


print("\nudpp")
udpp_model_xp = udppModel.UDPPmodel(slot_list, fl_list)
udpp_model_xp.run(optimised=True)
udpp_model_xp.print_performance()
# print(udpp_model_xp.get_new_df())

udpp_model_xp.run(optimised=False)
udpp_model_xp.print_performance()

new_fl_list = udpp_model_xp.get_new_flight_list()
# for flight in new_fl_list:
#     if flight.udppPriority == "P":
#         print(flight, flight.tna)

print("\nistop only pairs")
xpModel = istop.Istop(slot_list, new_fl_list, triples=False)
xpModel.run(True)
xpModel.print_performance()
print(xpModel.offers_selected)


# print("\nistop with triples")
# xpModel = istop.Istop(new_fl_list, triples=True)
# xpModel.run(True)
# xpModel.print_performance()

"""
TO CONSIDER - TO DO

-slot holes due to multiple hotspot
-cost function definition

"""