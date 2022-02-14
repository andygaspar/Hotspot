import sys
sys.path.insert(1, '../..')

from numpy import array

from Hotspot import models, models_correspondence_cost_vect, models_correspondence_approx
from Hotspot import Engine, LocalEngine, HotspotHandler, print_allocation
from Hotspot.libs.uow_tool_belt.general_tools import sort_lists
from Hotspot.libs.other_tools import compute_cost, generate_comparison

class Flight:
	pass

flight_names = ['A6001', 'A4003', 'A4002', 'A1000', 'A6000', 'A2000', 'A4001', 'A5000', 'A3000', 'A4000']
etas = [1.3284454334785711, 5.725348906862964, 10.011098323697942, 11.998242687553478, 14.614522199090489, 15.168806882178083, 16.230722099866547, 16.40837052722892, 17.945282936014696, 19.482770641946615]
slot_times = [ 1,  6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86,
 91, 96]
 
costVec_dict = {'A6001': array([  0.  ,   0.  ,  90.83,  90.83,  90.83, 179.12, 179.12, 179.12,
       179.12, 179.12, 179.12, 179.12, 179.12, 179.12, 179.12, 179.12,
       179.12, 179.12, 179.12, 179.12]), 'A4003': array([  0.  ,   0.  ,   0.  , 103.42, 103.42, 103.42, 103.42, 103.42,
       196.41, 196.41, 196.41, 196.41, 196.41, 196.41, 196.41, 196.41,
       196.41, 196.41, 196.41, 196.41]), 'A4002': array([  0.  ,   0.  ,   0.  , 116.06, 116.06, 116.06, 116.06, 225.62,
       225.62, 225.62, 225.62, 225.62, 225.62, 225.62, 225.62, 225.62,
       225.62, 225.62, 225.62, 225.62]), 'A1000': array([  0.  ,   0.  ,   0.  ,   0.  , 111.32, 111.32, 111.32, 216.86,
       216.86, 216.86, 216.86, 216.86, 216.86, 216.86, 216.86, 216.86,
       216.86, 216.86, 216.86, 216.86]), 'A6000': array([  0.  ,   0.  ,   0.  ,   0.  , 111.4 , 111.4 , 111.4 , 214.83,
       214.83, 214.83, 214.83, 214.83, 214.83, 214.83, 214.83, 214.83,
       214.83, 214.83, 214.83, 214.83]), 'A2000': array([  0.  ,   0.  ,   0.  ,   0.  , 104.71, 104.71, 104.71, 104.71,
       104.71, 188.47, 188.47, 188.47, 188.47, 188.47, 188.47, 188.47,
       188.47, 188.47, 188.47, 188.47]), 'A4001': array([  0.  ,   0.  ,   0.  ,  97.23,  97.23,  97.23,  97.23,  97.23,
       212.37, 212.37, 212.37, 212.37, 212.37, 212.37, 212.37, 212.37,
       212.37, 212.37, 212.37, 212.37]), 'A5000': array([  0.  ,   0.  ,   0.  ,  82.01,  82.01,  82.01,  82.01,  82.01,
       187.97, 187.97, 187.97, 187.97, 187.97, 187.97, 187.97, 187.97,
       187.97, 187.97, 187.97, 187.97]), 'A3000': array([  0.  ,   0.  ,   0.  ,   0.  , 111.44, 111.44, 111.44, 111.44,
       111.44, 214.11, 214.11, 214.11, 214.11, 214.11, 214.11, 214.11,
       214.11, 214.11, 214.11, 214.11]), 'A4000': array([  0.  ,   0.  ,   0.  ,   0.  ,  88.14,  88.14,  88.14,  88.14,
        88.14, 203.41, 203.41, 203.41, 203.41, 203.41, 203.41, 203.41,
       203.41, 203.41, 203.41, 203.41])}
airlines = {'A6': ['A6001', 'A6000'], 'A4': ['A4003', 'A4002', 'A4001', 'A4000'], 'A1': ['A1000'], 'A2': ['A2000'], 'A5': ['A5000'], 'A3': ['A3000']}

def test1():
	print ('TEST NUMBER 1: GLOBAL OPTIMUM.')
	algo = 'globaloptimum'

	external_flights = [] 
	for i, name in enumerate(flight_names):
		flight = Flight()
		flight.name = name
		flight.eta = etas[i]
		flight.airlineName = name[:2]
		external_flights.append(flight)

	engine = Engine(algo=algo)

	#cost_func_archetype = 'jump2'
	hotspot_handler = HotspotHandler(engine=engine,
									#cost_func_archetype=cost_func_archetype,
									alternative_allocation_rule=True)

	slots, flights = hotspot_handler.prepare_hotspot_from_flights_ext(flights_ext=external_flights,
																		slot_times=slot_times,
																		attr_map={'name':'flight_name',
																				'airlineName':'airline_name',
																				'eta':'eta'
																				},
																		# set_cost_function_with={'cost_function':'cost_func',
																		# 						'kind':'lambda',
																		# 						'absolute':False,
																		# 						'eta':'eta'}
																		)
	for flight in flights.values():
		flight.costVect = costVec_dict[flight.name]

	# This computes the cost vectors if needed.
	hotspot_handler.prepare_all_flights()

	print ('FPFS allocation:')
	fpfs_allocation = hotspot_handler.get_allocation()
	print_allocation(fpfs_allocation)
	print ()

	# Run model
	allocation = engine.compute_optimal_allocation(hotspot_handler=hotspot_handler,
												kwargs_init={}
												)

	# Allocation is an ordered dict linking flight -> slot
	print ('Final allocation:')
	print_allocation(allocation)
	print ()
	print ('Reduction in cost (last stage of model, approximated costs:')
	engine.print_optimisation_performance()

	print ()
	print ('Reduction in cost (overall, real costs:')
	results = generate_comparison(fpfs_allocation, allocation, airlines, costVec_dict)
	print (results)

	# Put slots in the original flight object as attribute "attr_slot_ext"
	hotspot_handler.assign_slots_to_flights_ext_from_allocation(allocation, attr_slot_ext='slot')
	print ('Slot assigned to first flight:', external_flights[0].slot)

	# One can also use this method to use internal flights object to get the slot, instead of the
	# allocation dict.
	hotspot_handler.update_flight_attributes_int_to_ext({'newSlot':'slot'})
	print ('Slot assigned to last flight:', external_flights[-1].slot)

	# You can also get another shiny list of flights
	new_flights = hotspot_handler.get_new_flight_list()

	# you can see the results of each individual engine by accessing this attribute
	# print ('Merged results:', engine.model.merge_results)

	print ()
	print ()
	print ()

if __name__=='__main__':
	test1()