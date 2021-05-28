#from Hotspot_package import *
import sys
sys.path.insert(1, '../..')

import numpy as np
import random
from collections import OrderedDict
from pprint import pprint
from copy import deepcopy


from Hotspot import *

class Test:
	def __init__(self):
		# in case remember both
		random.seed(0)
		np.random.seed(0)


		# ************* init
		scheduleType = schedule_types(show=True)

		num_flights = 10
		num_airlines = 3


		distribution = scheduleType[0]#3]
		print("schedule type: ", distribution)

		self.df = df_maker(num_flights, num_airlines, distribution=distribution)

		print ('Initial DataFrame:')
		print (self.df)

	def test_GlobalOptimum(self):
		# fl_list has flight objects: you can now manually set slot, margin1, jump2, margin2, jump2
		# **************** models run
		print("\n Global Optimum")
		slot_list, fl_list = make_flight_list(self.df)
		print ('Slots attached to flights (before GlobalOptimum):', {flight:flight.slot.index for flight in fl_list})
		print ('newSlots attached to flights (before GlobalOptimum):', {flight:getattr(flight.newSlot, 'index', None) for flight in fl_list})
		global_model = GlobalOptimum(slot_list, fl_list)
		global_model.run()
		global_model.print_performance()

		print ('Slots attached to flights (after GlobalOptimum):', {flight:flight.slot.index for flight in fl_list})
		print ('newSlots attached to flights (after GlobalOptimum):', {flight:getattr(flight.newSlot, 'index', None) for flight in fl_list})
		print ()

		slot_list, fl_list = make_flight_list(self.df)
		print ('Slots attached to flights (before GlobalOptimum):', {flight:flight.slot.index for flight in fl_list})
		print ('newSlots attached to flights (before GlobalOptimum):', {flight:getattr(flight.newSlot, 'index', None) for flight in fl_list})
		global_model = GlobalOptimum(slot_list, fl_list)
		global_model.run()
		global_model.print_performance()
		global_model.update_flights()
		print ('Slots attached to flights (after GlobalOptimum, with slot update):', {flight:flight.slot.index for flight in fl_list})
		print ('newSlots attached to flights (after GlobalOptimum), with slot update:', {flight:getattr(flight.newSlot, 'index', None) for flight in fl_list})
		print ()		

	def test_nnbound(self):
		print("\nnnBound")
		
		slot_list, fl_list = make_flight_list(self.df)
		print ('Slots attached to flights (before nnBound):', {flight:flight.slot.index for flight in fl_list})
		print ('newSlots attached to flights (before nnBound):', {flight:getattr(flight.newSlot, 'index', None) for flight in fl_list})
		max_model = NNBoundModel(slot_list, fl_list)
		max_model.run()
		max_model.print_performance()
		print ('Slots attached to flights (after nnBound, without slot update):', {flight:flight.slot.index for flight in fl_list})
		print ('newSlots attached to flights (after nnBound, without slot update):', {flight:getattr(flight.newSlot, 'index', None) for flight in fl_list})
		print ()

		slot_list, fl_list = make_flight_list(self.df)
		print ('Slots attached to flights (before nnBound):', {flight:flight.slot.index for flight in fl_list})
		print ('newSlots attached to flights (before nnBound):', {flight:getattr(flight.newSlot, 'index', None) for flight in fl_list})
		max_model = NNBoundModel(slot_list, fl_list)
		max_model.run()
		max_model.print_performance()
		max_model.update_flights()
		print ('Slots attached to flights (after nnBound, with slot update):', {flight:flight.slot.index for flight in fl_list})
		print ('newSlots attached to flights (after nnBound, with slot update):', {flight:getattr(flight.newSlot, 'index', None) for flight in fl_list})

	def test_udpp(self):
		print("\nUPPP")
		slot_list, fl_list = make_flight_list(self.df)
		print ('Slots attached to flights (before UDPP):', {flight:flight.slot.index for flight in fl_list})
		print ('newSlots attached to flights (before UDPP):', {flight:getattr(flight.newSlot, 'index', None) for flight in fl_list})
		udpp_model_xp = UDPPmodel(slot_list, fl_list)
		udpp_model_xp.run(optimised=True)
		udpp_model_xp.print_performance()
		
		print ('Slots attached to flights (after UDPP, without slot update):', {flight:flight.slot.index for flight in fl_list})
		print ('newSlots attached to flights (after UDPP, without slot update):', {flight:getattr(flight.newSlot, 'index', None) for flight in fl_list})

		# Non-optimised version does not work
		# print("\nUPPP")
		# slot_list, fl_list = make_flight_list(self.df)
		# print ('Slots attached to flights (before UDPP):', {flight:flight.slot.index for flight in fl_list})
		# print ('newSlots attached to flights (before UDPP):', {flight:getattr(flight.newSlot, 'index', None) for flight in fl_list})
		# udpp_model_xp = UDPPmodel(slot_list, fl_list)
		# udpp_model_xp.run(optimised=False, update_flights=False)
		# udpp_model_xp.print_performance()
		
		# print ('Slots attached to flights (after UDPP, without slot update):', {flight:flight.slot.index for flight in fl_list})
		# print ('newSlots attached to flights (after UDPP, without slot update):', {flight:getattr(flight.newSlot, 'index', None) for flight in fl_list})

		slot_list, fl_list = make_flight_list(self.df)
		print ('\nSlots attached to flights (before UDPP):', {flight:flight.slot.index for flight in fl_list})
		print ('newSlots attached to flights (before UDPP):', {flight:getattr(flight.newSlot, 'index', None) for flight in fl_list})
		udpp_model_xp = UDPPmodel(slot_list, fl_list)
		udpp_model_xp.run(optimised=True)
		udpp_model_xp.print_performance()
		udpp_model_xp.update_flights()

		print ('Slots attached to flights (after UDPP, with slot update):', {flight:flight.slot.index for flight in fl_list})
		print ('newSlots attached to flights (after UDPP, with slot update):', {flight:getattr(flight.newSlot, 'index', None) for flight in fl_list})

		slot_list, fl_list = make_flight_list(self.df)
		print ('\nSlots attached to flights (before UDPP):', {flight:flight.slot.index for flight in fl_list})
		print ('newSlots attached to flights (before UDPP):', {flight:getattr(flight.newSlot, 'index', None) for flight in fl_list})
		udpp_model_xp = UDPPmodel(slot_list, fl_list)
		udpp_model_xp.run(optimised=True)
		udpp_model_xp.print_performance()
		
		new_fl_list = udpp_model_xp.get_new_flight_list()

		print ('Slots attached to flights (after UDPP, with new flight list):', {flight:flight.slot.index for flight in new_fl_list})
		print ('newSlots attached to flights (after UDPP, with new flight list):', {flight:getattr(flight.newSlot, 'index', None) for flight in new_fl_list})
		print ()

	def test_istop(self):

		slot_list, fl_list = make_flight_list(self.df)

		#remember to run Istop after the UDPP
		print("\nISTOP only pairs")
		slot_list, fl_list = make_flight_list(self.df)
		print ('\nSlots attached to flights (before ISTOP):', {flight:flight.slot.index for flight in fl_list})
		print ('newSlots attached to flights (before ISTOP):', {flight:getattr(flight.newSlot, 'index', None) for flight in fl_list})
		xpModel = Istop(slot_list, fl_list, triples=False)
		xpModel.run(True)
		xpModel.print_performance()
		print(xpModel.offers_selected)
		print ('Slots attached to flights (after ISTOP, without slot update):', {flight:flight.slot.index for flight in fl_list})
		print ('newSlots attached to flights (after ISTOP, without slot update):', {flight:getattr(flight.newSlot, 'index', None) for flight in fl_list})


		# print("\nISTOP with triples")
		# xpModel = istop.Istop(new_fl_list, triples=True)
		# xpModel.run(True)
		# xpModel.print_performance()


if __name__=='__main__':
	test = Test()
	test.test_GlobalOptimum()
	test.test_udpp()
	test.test_nnbound()
	test.test_istop()

