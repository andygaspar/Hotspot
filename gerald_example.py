from Hotspot_package import *
import numpy as np
import random


# in case remember both
random.seed(0)
np.random.seed(0)


# ************* init
scheduleType = schedule_types(show=True)

num_flights = 50
num_airlines = 5


distribution = scheduleType[3]
print("schedule type: ", distribution)

df = df_maker(num_flights, num_airlines, distribution=distribution)

slot_list, fl_list = make_flight_list(df)


# fl_list has flight objects: you can now manually set slot, margin1, jump2, margin2, jump2

# **************** models run
print("\n global optimum")
global_model = GlobalOptimum(slot_list, fl_list)
global_model.run()
global_model.print_performance()

print("how to get attributes")
print(global_model.flights[0].get_attributes())

print("\nnn bound")
max_model = NNBoundModel(slot_list, fl_list)
max_model.run()
max_model.print_performance()


print("\nudpp")
udpp_model_xp = UDPPmodel(slot_list, fl_list)
udpp_model_xp.run(optimised=True)
udpp_model_xp.print_performance()
# print(udpp_model_xp.get_new_df())

udpp_model_xp.run(optimised=False)
udpp_model_xp.print_performance()

new_fl_list = udpp_model_xp.get_new_flight_list()

#remember to run Istop after the UDPP
print("\nistop only pairs")
xpModel = Istop(slot_list, new_fl_list, triples=False)
xpModel.run(True)
xpModel.print_performance()
print(xpModel.offers_selected)


# print("\nistop with triples")
# xpModel = istop.Istop(new_fl_list, triples=True)
# xpModel.run(True)
# xpModel.print_performance()

