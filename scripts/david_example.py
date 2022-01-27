import sys
sys.path.insert(1, '../..')

from numpy import array

from Hotspot import models, models_correspondence_cost_vect, models_correspondence_approx
from Hotspot import Engine, LocalEngine, HotspotHandler, print_allocation
from Hotspot.libs.uow_tool_belt.general_tools import sort_lists

class Flight:
	pass

def test1():
	algo = 'udpp_istop_approx'

	flight_names = ['A5002', 'A3000', 'A1004', 'A3003', 'A3004', 'A3005', 'A3002',
					'A1006', 'A1010', 'A3006', 'A2010', 'A2009', 'A5005']
	#etas = [586, 589, 592, 622, 624, 627, 639, 649, 652, 667, 667, 667, 678]
	etas = [586, 589, 592, 622, 624, 627, 639, 649, 652, 664, 667, 667, 678]
	costVec_dict = {'A5002': array([49162.3824, 51154.76640000001, 53147.150400000006, 53811.2784, 54475.40640000001, 55139.534400000004, 55803.6624, 56467.790400000005, 57131.9184, 57796.04640000001, 58460.1744, 59124.30240000001, 59788.430400000005]),
					'A3000': array([47255.08, 49042.600000000006, 50830.12, 51425.96, 52021.8, 52617.64, 53213.479999999996, 53809.32, 54405.16, 55001.0, 58083.32400000001, 58734.648, 59385.972]),
					'A1004': array([47916.0832, 49844.4472, 53952.15232, 54655.11912000001, 55358.085920000005, 56061.05272000001, 56764.01952000001, 57466.98632, 58169.95312, 58872.91992000001, 59575.88672000001, 60278.85352000001, 60981.820320000006]),
					'A3003': array([0.0, 0.0, 47916.0832, 48558.8712, 49201.6592, 49844.4472, 50487.2352, 51130.0232, 51772.811200000004, 54655.11912000001, 55358.085920000005, 56061.05272000001, 56764.01952000001]),
					'A3004': array([0.0, 0.0, 46677.224, 47307.208, 47937.192, 48567.17600000001, 49197.16, 49827.144, 50457.128, 51087.11200000001, 53944.30560000001, 54633.18800000001, 55322.070400000004]),
					'A3005': array([0.0, 0.0, 14076.088000000002, 14664.168000000001, 15252.248000000001, 15840.328000000001, 16428.408000000003, 17016.488, 17604.568, 40792.648, 41380.727999999996, 41968.808, 42556.888]),
					'A3002': array([0.0, 0.0, 0.0, 0.0, 14937.408, 15516.615999999998, 16096.936, 16677.256, 17257.576, 17837.895999999997, 18418.216, 18998.536, 19578.856]),
					'A1006': array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 45439.8, 46027.88, 46615.96, 47204.04000000001, 47792.12, 48380.2, 48968.28]),
					'A1010': array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3327.2416000000003, 9444.0664, 10041.6224, 10639.178399999999, 11236.7344, 11834.2904]),
					'A3006': array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 44734.976, 45311.416000000005, 45887.85600000001]),
					'A2010': array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16536.936, 17158.384000000005, 17779.832000000002]),
					'A2009': array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 28504.232, 29511.352000000003, 70318.47200000001]),
					'A5005': array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 17407.16])} 
	# costVec_dict = {'A5002': array([49162.3824, 51154.7664, 53147.1504, 53811.2784, 54475.4064, 55139.5344, 55803.6624, 56467.7904, 57131.9184, 57796.0464, 58460.1744, 59124.3024, 59788.4304]),
	# 				'A3000': array([47255.08 , 49042.6  , 50830.12 , 51425.96 , 52021.8  , 52617.64 , 53213.48 , 53809.32 , 54405.16 , 55001.   , 58083.324, 58734.648, 59385.972]),
	# 				'A1004': array([47916.0832 , 49844.4472 , 53952.15232, 54655.11912, 55358.08592, 56061.05272, 56764.01952, 57466.98632, 58169.95312, 58872.91992, 59575.88672, 60278.85352, 60981.82032]),
	# 				'A3003': array([    0.     ,     0.     , 47916.0832 , 48558.8712 , 49201.6592 , 49844.4472 , 50487.2352 , 51130.0232 , 51772.8112 , 54655.11912, 55358.08592, 56061.05272, 56764.01952]),
	# 				'A3004': array([    0.    ,     0.    , 46677.224 , 47307.208 , 47937.192 , 48567.176 , 49197.16  , 49827.144 , 50457.128 , 51087.112 , 53944.3056, 54633.188 , 55322.0704]),
	# 				'A3005': array([    0.   ,     0.   , 14076.088, 14664.168, 15252.248, 15840.328, 16428.408, 17016.488, 17604.568, 40792.648, 41380.728, 41968.808, 42556.888]),
	# 				'A3002': array([    0.   ,     0.   ,     0.   ,     0.   , 14937.408, 15516.616, 16096.936, 16677.256, 17257.576, 17837.896, 18418.216, 18998.536, 19578.856]),
	# 				'A1006': array([    0.  ,     0.  ,     0.  ,     0.  ,     0.  ,     0.  , 45439.8 , 46027.88, 46615.96, 47204.04, 47792.12, 48380.2 , 48968.28]),
	# 				'A3006': array([    0.  ,     0.  ,     0.  ,     0.  ,     0.  ,     0.  , 45439.8 , 46027.88, 46615.96, 47204.04, 47792.12, 48380.2 , 48968.28]),
	# 				'A1010': array([    0.    ,     0.    ,     0.    ,     0.    ,     0.    , 0.    ,     0.    ,  3327.2416,  9444.0664, 10041.6224, 12000., 14000., 18000.]),
	# 				'A2010': array([    0.   ,     0.   ,     0.   ,     0.   ,     0.   ,     0.   , 0.   ,     0.   ,     0.   ,     0.   , 16536.936, 17158.384, 17779.832]),
	# 				'A2009': array([    0.   ,     0.   ,     0.   ,     0.   ,     0.   ,     0.   , 0.   ,     0.   ,     0.   ,     0.   , 28504.232, 29511.352, 70318.472]),
	# 				'A5005': array([    0.  ,     0.  ,     0.  ,     0.  ,     0.  ,     0.  , 0.  ,     0.  ,     0.  ,     0.  ,     0.  ,     0.  , 17407.16])} 

	external_flights = [] 
	for i, name in enumerate(flight_names):
		flight = Flight()
		flight.name = name
		flight.eta = etas[i]
		flight.airlineName = name[:2]
		external_flights.append(flight)

	slot_times = [600, 615, 630, 635, 640, 645, 650,
				  655, 660, 665, 670, 675, 680]

	engine = Engine(algo=algo)

	cost_func_archetype = 'jump2'
	hotspot_handler = HotspotHandler(engine=engine,
									cost_func_archetype=cost_func_archetype)

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
	print_allocation(hotspot_handler.get_allocation())
	print ()

	# Run model
	allocation = engine.compute_optimal_allocation(hotspot_handler=hotspot_handler,
												)

	# Allocation is an ordered dict linking flight -> slot
	print ('Final allocation:')
	print_allocation(allocation)
	print ()
	engine.print_optimisation_performance()

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
	print ('Merged results:', engine.model.merge_results)

	print ()

def test2():
	algo = 'udpp_approx'

	flight_names = ['A5002', 'A3000', 'A1004', 'A3003', 'A3004', 'A3005', 'A3002',
					'A1006', 'A1010', 'A3006', 'A2010', 'A2009', 'A5005']
	#etas = [586, 589, 592, 622, 624, 627, 639, 649, 652, 667, 667, 667, 678]
	etas = [586, 589, 592, 622, 624, 627, 639, 649, 652, 664, 667, 667, 678]
	costVec_dict = {'A5002': array([49162.3824, 51154.7664, 53147.1504, 53811.2784, 54475.4064, 55139.5344, 55803.6624, 56467.7904, 57131.9184, 57796.0464, 58460.1744, 59124.3024, 59788.4304]),
					'A3000': array([47255.08 , 49042.6  , 50830.12 , 51425.96 , 52021.8  , 52617.64 , 53213.48 , 53809.32 , 54405.16 , 55001.   , 58083.324, 58734.648, 59385.972]),
					'A1004': array([47916.0832 , 49844.4472 , 53952.15232, 54655.11912, 55358.08592, 56061.05272, 56764.01952, 57466.98632, 58169.95312, 58872.91992, 59575.88672, 60278.85352, 60981.82032]),
					'A3003': array([    0.     ,     0.     , 47916.0832 , 48558.8712 , 49201.6592 , 49844.4472 , 50487.2352 , 51130.0232 , 51772.8112 , 54655.11912, 55358.08592, 56061.05272, 56764.01952]),
					'A3004': array([    0.    ,     0.    , 46677.224 , 47307.208 , 47937.192 , 48567.176 , 49197.16  , 49827.144 , 50457.128 , 51087.112 , 53944.3056, 54633.188 , 55322.0704]),
					'A3005': array([    0.   ,     0.   , 14076.088, 14664.168, 15252.248, 15840.328, 16428.408, 17016.488, 17604.568, 40792.648, 41380.728, 41968.808, 42556.888]),
					'A3002': array([    0.   ,     0.   ,     0.   ,     0.   , 14937.408, 15516.616, 16096.936, 16677.256, 17257.576, 17837.896, 18418.216, 18998.536, 19578.856]),
					'A1006': array([    0.  ,     0.  ,     0.  ,     0.  ,     0.  ,     0.  , 45439.8 , 46027.88, 46615.96, 47204.04, 47792.12, 48380.2 , 48968.28]),
					'A3006': array([    0.  ,     0.  ,     0.  ,     0.  ,     0.  ,     0.  , 45439.8 , 46027.88, 46615.96, 47204.04, 47792.12, 48380.2 , 48968.28]),
					'A1010': array([    0.    ,     0.    ,     0.    ,     0.    ,     0.    , 0.    ,     0.    ,  3327.2416,  9444.0664, 10041.6224, 12000., 14000., 18000.]),
					'A2010': array([    0.   ,     0.   ,     0.   ,     0.   ,     0.   ,     0.   , 0.   ,     0.   ,     0.   ,     0.   , 16536.936, 17158.384, 17779.832]),
					'A2009': array([    0.   ,     0.   ,     0.   ,     0.   ,     0.   ,     0.   , 0.   ,     0.   ,     0.   ,     0.   , 28504.232, 29511.352, 70318.472]),
					'A5005': array([    0.  ,     0.  ,     0.  ,     0.  ,     0.  ,     0.  , 0.  ,     0.  ,     0.  ,     0.  ,     0.  ,     0.  , 17407.16])} 

	external_flights = [] 
	for i, name in enumerate(flight_names):
		flight = Flight()
		flight.name = name
		flight.eta = etas[i]
		flight.airlineName = name[:2]
		external_flights.append(flight)

	slot_times = [600, 615, 630, 635, 640, 645, 650,
				  655, 660, 665, 670, 675, 680]

	engine = Engine(algo=algo)

	cost_func_archetype = 'jump2'
	hotspot_handler = HotspotHandler(engine=engine,
									cost_func_archetype=cost_func_archetype)

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
	print_allocation(hotspot_handler.get_allocation())
	print ()

	# Run model
	allocation = engine.compute_optimal_allocation(hotspot_handler=hotspot_handler,
												)

	# Allocation is an ordered dict linking flight -> slot
	print ('Final allocation:')
	print_allocation(allocation)
	print ()
	engine.print_optimisation_performance()

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
	print ('Merged results:', engine.model.merge_results)

	print ()

if __name__=='__main__':
	test1()

	test2()