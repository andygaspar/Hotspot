from Hotspot_package import *

import numpy as np

from typing import Callable

import pandas as pd

np.random.seed(0)


class DavidFlight:

    def __init__(self, name: str, airline_name: str, time: float, eta: float):
        self.eta = eta
        self.airlineName = airline_name
        self.time = time
        self.name = name
        self.cost_coefficient = np.random.uniform(0.5, 2, 1)[0]
        self.cost_fun = lambda delay: self.cost_coefficient * delay ** 2


# ************* SETTING UP, IGNORE THIS PART

df = pd.read_csv("david_test.csv")
david_flights = []
for i in range(df.shape[0]):
    line = df.iloc[i]
    david_flights.append(DavidFlight(line["flight"], line["airline"], line["time"], line["eta"]))

# WE START FROM HERE

# from your model we get
slot_times = range(0, 98, 2)  # or an np array or list or whatever
empty_slots_times = [34, 60, 86]  # if needed, in this case yes

# make the slot list and the flight list as follow (without empty slots is much easier)
slot_list = []
flight_list = []

flight_list_index = 0

hotspot_start = slot_times[0]

for i in range(len(slot_times)):
    if slot_times[i] in empty_slots_times:
        slot, flight = make_slot_and_flight(slot_index=i, slot_time=slot_times[i], empty_slot=True)

    else:
        f = david_flights[flight_list_index]
        slot, flight = make_slot_and_flight(slot_index=i, slot_time=slot_times[i],
                                            flight_name=f.name, airline_name=f.airlineName, eta=f.eta,
                                            delay_cost_vect=
                                            np.array([f.cost_fun(time - hotspot_start)
                                                      for time in slot_times])
                                            )
        flight_list_index += 1
    slot_list.append(slot)
    flight_list.append(flight)

# **************** models run
print("\n global optimum")
global_model = GlobalOptimum(slot_list, flight_list)
global_model.run()
global_model.print_performance()

print("how to get attributes")
print(global_model.flights[0].get_attributes())

print("\nnn bound")
max_model = NNBoundModel(slot_list, flight_list)
max_model.run()
max_model.print_performance()

# this can be used also for an agent only, initialising one AU's flight only, but all slots as needed for the protection
print("\nudpp")
udpp_model_xp = UDPPmodel(slot_list, flight_list)
udpp_model_xp.run(optimised=True)
udpp_model_xp.print_performance()

print("how to get attributes, note now the priority and the priority number")
print(udpp_model_xp.flights[0].get_attributes())

udpp_model_xp.run(optimised=False)
udpp_model_xp.print_performance()

# remember to run Istop after the UDPP
new_fl_list = udpp_model_xp.get_new_flight_list()

print("\nistop only pairs")
xpModel = Istop(slot_list, new_fl_list, triples=False)
xpModel.run(True)
xpModel.print_performance()
print(xpModel.offers_selected)

# print("\nistop with triples")
# xpModel = istop.Istop(new_fl_list, triples=True)
# xpModel.run(True)
# xpModel.print_performance()
