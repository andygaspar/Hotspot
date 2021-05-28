#from Hotspot_package import *
import sys
sys.path.insert(1, '../..')

import numpy as np
import random
from collections import OrderedDict
from pprint import pprint
from copy import deepcopy


from Hotspot import *


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

fl_list_init = deepcopy(fl_list)


# fl_list has flight objects: you can now manually set slot, margin1, jump2, margin2, jump2
print ()
print ('Slots attached to flights (initial):', {flight:flight.slot.index for flight in fl_list})
print ('newSlots attached to flights (initial):', {flight:getattr(flight.newSlot, 'index', None) for flight in fl_list})

# **************** models run
print("\n global optimum")
global_model = GlobalOptimum(slot_list, fl_list)
global_model.run()
global_model.print_performance()

print ('Slots attached to flights (after globaloptimum):', {flight:flight.slot.index for flight in fl_list})
print ('newSlots attached to flights (after globaloptimum):', {flight:getattr(flight.newSlot, 'index', None) for flight in fl_list})
# print("how to get attributes")
# print(global_model.flights[0].get_attributes())

# print("\nnn bound")
# max_model = NNBoundModel(slot_list, fl_list)
# max_model.run()
# max_model.print_performance()

# print ('COIN2 Slots attached to flights (after best):', OrderedDict([(flight, flight.slot.index) for flight in fl_list]))
# print ('newSlots attached to flights (after best):', [getattr(flight.newSlot, 'index', None) for flight in fl_list])
print ('\nReset slots in flights...')
[flight.reset_slot() for flight in fl_list]
print ()

print("\nudpp")
print ('Slots attached to flights (before UDPP):', {flight:flight.slot.index for flight in fl_list})
print ('newSlots attached to flights (before UDPP):', {flight:getattr(flight.newSlot, 'index', None) for flight in fl_list})

udpp_model_xp = UDPPmodel(slot_list, fl_list)
udpp_model_xp.run(optimised=True)
udpp_model_xp.print_performance()
# print(udpp_model_xp.get_new_df())

print ('Slots attached to flights (after UDPP, without slot update):', {flight:flight.slot.index for flight in fl_list})
print ('newSlots attached to flights (after UDPP, without slot update):', {flight:getattr(flight.newSlot, 'index', None) for flight in fl_list})

print ('\nReset slots in flights...')
[flight.reset_slot() for flight in fl_list]

print ('\nSlots attached to flights (before UDPP):', {flight:flight.slot.index for flight in fl_list})
print ('newSlots attached to flights (before UDPP):', {flight:getattr(flight.newSlot, 'index', None) for flight in fl_list})

udpp_model_xp.run(optimised=False, update_flights=True)
udpp_model_xp.print_performance()

print ('Slots attached to flights (after UDPP, with slot update):', {flight:flight.slot.index for flight in fl_list})
print ('newSlots attached to flights (after UDPP, with slot update):', {flight:getattr(flight.newSlot, 'index', None) for flight in fl_list})

print ()
print ('\nSlots attached to flights (before UDPP):', {flight:flight.slot.index for flight in fl_list})
print ('newSlots attached to flights (before UDPP):', {flight:getattr(flight.newSlot, 'index', None) for flight in fl_list})

fl_list = deepcopy(fl_list_init)
udpp_model_xp = UDPPmodel(slot_list, fl_list)
udpp_model_xp.run(optimised=False, update_flights=False)
udpp_model_xp.print_performance()
new_fl_list = udpp_model_xp.get_new_flight_list()

print ('Slots attached to flights (after UDPP, with new flight list):', {flight:flight.slot.index for flight in new_fl_list})
print ('newSlots attached to flights (after UDPP, with new flight list):', {flight:getattr(flight.newSlot, 'index', None) for flight in new_fl_list})
print ()



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

