"""
Version using the wrapper
"""

import sys
sys.path.insert(1, '../..')
sys.path.insert(1, '..')

import numpy as np
from typing import Callable
import pandas as pd
import pickle
import random

from Hotspot import models, print_allocation
from Hotspot import HotspotHandler, LocalEngine, OptimalAllocationEngine as Engine
#from Hotspot.ModelStructure.Costs.costFunctionDict import CostFuns

from create_test_data import dict_funcs

# In case remember both
random.seed(0)
np.random.seed(0)

def simple_function(delay, cost_coefficient):
	return cost_coefficient * delay ** 2

def load_test(test_number):
	df = pd.read_csv('../test_data/df_test_{}.csv'.format(test_number))

	# with open('../test_data/archetype_function_test_{}.pic'.format(test_number), 'rb') as f:
	# 	archetype_function = dict_funcs[pickle.load(f)]

	flights = []
	for i, row in df.iterrows():
		flight = ModelFlight(name=row['name'],
							airline_name=row['airlineName'],
							time=row['time'],
							eta=row['eta'],
							cost_coefficient=row['cost_coefficient'],
							function=simple_function)
		# print ('AKKO', flight.name, row['cost_coefficient'])
		# print ('feon', flight.name, flight.cost_func(7.))
		flights.append(flight)

	return flights

# First bit to simulate things coming from another model, e.g. 
# flight objects

class ModelFlight:
	def __init__(self, name: str, airline_name: str, time: float, eta: float,
		cost_coefficient, function):
		self.eta = eta
		self.airlineName = airline_name
		self.time = time
		self.name = name
		self.coef = cost_coefficient
		self.cost_func = lambda delay: function(delay, cost_coefficient)
		self.cost_func2 = simple_function

tests_to_run = [1]

## These tests use costVect and delayCostVect, so they should always be passed ###

for test_number in tests_to_run:
	print (pd.read_csv('../test_data/df_test_{}.csv'.format(test_number)))
	for algo in ['udpp_merge', 'istop', 'nnbound', 'globaloptimum']:
		print ()
		print ()

		print ("########### {} TRUE COST VECTORS (Mercury) ############".format(algo))
		mercury_flights = load_test(test_number)
		slot_times = list(range(0, 2*len(mercury_flights), 2))

		mercury_flights_per_airline = {}
		for flight in mercury_flights:
			mercury_flights_per_airline[flight.airlineName] = mercury_flights_per_airline.get(flight.airlineName, []) + [flight]

		# ------- Network Manager agent starts here ----- # 
		engine = Engine(algo=algo)
		hh_NM = HotspotHandler(engine=engine)
		mercury_flights_dict = [{'flight_name':mf.name,
									'airline_name':mf.airlineName,
									'eta':mf.eta,
									} for mf in mercury_flights]
		hh_NM.prepare_hotspot_from_dict(attr_list=mercury_flights_dict,
										slot_times=slot_times) 
		all_allocated_slots = hh_NM.get_assigned_slots()
		to_be_sent_to_airlines = {}
		for airline, flights_in_airline in mercury_flights_per_airline.items():
			message_to_airline = {}
			for flight in flights_in_airline:
				message_to_airline[flight] = {'slot':all_allocated_slots[flight.name]}
			message_to_airline['slots'] = list(all_allocated_slots.values())
			to_be_sent_to_airlines[airline] = message_to_airline
		# ------- Network Manager agents ends here ----- # 
		
		all_messages = []
		for airline, mercury_flights_airline in mercury_flights_per_airline.items():
			message_from_NM = to_be_sent_to_airlines[airline]
			if algo=='udpp_merge':
				algo_local = 'udpp_local'
			else:
				algo_local ='get_cost_vectors'

			engine_local = LocalEngine(algo=algo_local)

			hh = HotspotHandler(engine=engine_local)

			mercury_flights_dict = [{'flight_name':mf.name,
									'airline_name':mf.airlineName,
									'eta':mf.eta,
									'cost_function':mf.cost_func, # pass real cost function here
									'slot':message_from_NM[mf]['slot']
									} for mf in mercury_flights_airline]

			_, flights_airline = hh.prepare_hotspot_from_dict(attr_list=mercury_flights_dict,
																slots=message_from_NM['slots'],
																set_cost_function_with={'cost_function':'cost_function',
																						'kind':'lambda',
																						'absolute':False,
																						'eta':'eta'},
																)

			hh.prepare_all_flights()
			preferences = engine_local.compute_optimal_parameters(hotspot_handler=hh)
			to_be_sent_to_NM = {}
			for i, (name, pref) in enumerate(preferences.items()):
				to_be_sent_to_NM[name] = pref
				# ------- Flight agent ends here ------ #
			all_messages.append(to_be_sent_to_NM)
			
		# ------- Network Manager agent starts here again ----- # 
		for message in all_messages:
			hh_NM.update_flight_attributes_int_from_dict(attr_list=message) 

		hh_NM.prepare_all_flights()

		allocation = engine.compute_optimal_allocation(hotspot_handler=hh_NM)
		
		print_allocation(allocation)
		print (engine.model.report)
		print (engine.model.solution)

		if algo=='udpp_merge':
			df_sol = pd.read_csv('../test_data/df_solution_udpp_merge_{}.csv'.format(test_number),
								index_col=0,
										)
			ar_sol_base = df_sol.to_numpy()
			ar_sol =engine.model.solution.to_numpy()
			np.array_equal(ar_sol, ar_sol_base)
			print ('Test {} passed!'.format(test_number))


### These tests use approximation for costVect and delayCostVect, so they may fail ####

for test_number in tests_to_run:
	print (pd.read_csv('../test_data/df_test_{}.csv'.format(test_number)))
	for algo in ['istop', 'udpp_merge_istop', 'nnbound', 'globaloptimum']:
		print ()
		print ()

		print ("########### {} APPROX (Mercury) ############".format(algo))
		mercury_flights = load_test(test_number)
		slot_times = list(range(0, 2*len(mercury_flights), 2))

		mercury_flights_per_airline = {}
		for flight in mercury_flights:
			mercury_flights_per_airline[flight.airlineName] = mercury_flights_per_airline.get(flight.airlineName, []) + [flight]

		# ------- Network Manager agent starts here ----- # 
		engine = Engine(algo=algo)
		archetype_cost_function = 'jump'
		hh_NM = HotspotHandler(engine=engine,
								archetype_cost_function=archetype_cost_function
								)
		mercury_flights_dict = [{'flight_name':mf.name,
									'airline_name':mf.airlineName,
									'eta':mf.eta,
									} for mf in mercury_flights]
		hh_NM.prepare_hotspot_from_dict(attr_list=mercury_flights_dict,
										slot_times=slot_times) 
		all_allocated_slots = hh_NM.get_assigned_slots()
		to_be_sent_to_airlines = {}
		for airline, flights_in_airline in mercury_flights_per_airline.items():
			message_to_airline = {}
			for flight in flights_in_airline:
				message_to_airline[flight] = {'slot':all_allocated_slots[flight.name]}
			message_to_airline['archetype_cost_function'] = archetype_cost_function
			message_to_airline['slots'] = list(all_allocated_slots.values())
			to_be_sent_to_airlines[airline] = message_to_airline
		# ------- Network Manager agents ends here ----- # 
		
		all_messages = []
		for airline, mercury_flights_airline in mercury_flights_per_airline.items():
			message_from_NM = to_be_sent_to_airlines[airline]
			if algo=='udpp_merge':
				algo_local = 'udpp_local'
			elif algo=='udpp_merge_istop':
				algo_local = 'udpp_local_function_approx'
			else:
				algo_local = 'function_approx'
			engine_local = LocalEngine(algo=algo_local)

			hh = HotspotHandler(engine=engine_local,
								archetype_cost_function=message_from_NM['archetype_cost_function']
								)

			mercury_flights_dict = [{'flight_name':mf.name,
									'airline_name':mf.airlineName,
									'eta':mf.eta,
									'cost_function':mf.cost_func, # pass real cost function here
									'slot':message_from_NM[mf]['slot']
									} for mf in mercury_flights_airline]

			_, flights_airline = hh.prepare_hotspot_from_dict(attr_list=mercury_flights_dict,
																slots=message_from_NM['slots'],
																set_cost_function_with={'cost_function':'cost_function',
																						'kind':'lambda',
																						'absolute':False,
																						'eta':'eta'},
																)

			hh.prepare_all_flights()
			preferences = engine_local.compute_optimal_parameters(hotspot_handler=hh)
			to_be_sent_to_NM = {}
			for i, (name, pref) in enumerate(preferences.items()):
				to_be_sent_to_NM[name] = pref
				# ------- Flight agent ends here ------ #
			all_messages.append(to_be_sent_to_NM)
									
		# ------- Network Manager agent starts here again ----- # 
		if algo=='udpp_merge':
			set_cost_function_with = None
		else:
			set_cost_function_with = 'default_cf_paras'
		for message in all_messages:
			hh_NM.update_flight_attributes_int_from_dict(attr_list=message,
														set_cost_function_with=set_cost_function_with
														) 
		hh_NM.prepare_all_flights()

		allocation = engine.compute_optimal_allocation(hotspot_handler=hh_NM)
		print_allocation(allocation)
		print (engine.model.report)
		print (engine.model.solution)
