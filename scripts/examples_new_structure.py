"""
Version using the wrapper
"""

import sys
sys.path.insert(1, '../..')

import numpy as np
from typing import Callable
import pandas as pd

from Hotspot import models, print_allocation
from Hotspot import HotspotHandler, LocalEngine, OptimalAllocationEngine as Engine
#from Hotspot.ModelStructure.Costs.costFunctionDict import CostFuns

np.random.seed(0)

# First bit to simulate things coming from another model, e.g. 
# flight objects

class ModelFlight:
	def __init__(self, name: str, airline_name: str, time: float, eta: float):
		self.eta = eta
		self.airlineName = airline_name
		self.time = time
		self.name = name
		self.cost_coefficient = np.random.uniform(0.5, 2, 1)[0]
		self.cost_func = lambda delay: self.cost_coefficient * delay ** 2

def create_original_flights(n_f=50):
	df = pd.read_csv("david_test.csv")
	david_flights = []
	for i in list(range(df.shape[0]))[:n_f]:
		line = df.iloc[i]
		david_flights.append(ModelFlight(line["flight"], line["airline"], line["time"], line["eta"]))
	
	return david_flights

n_f = 10

print ('Available models:', models.keys())

for algo in ['udpp_merge', 'istop', 'nnbound', 'globaloptimum']:
	print ()
	print ()
	print ()

	# Most of the code in the same for all algorithms.
	print ("########### {} merging (Mercury) ############".format(algo))
	# Use case for Mercury, strong agent paradigm
	mercury_flights = create_original_flights(n_f=10)
	slot_times = list(range(0, 2*n_f, 2))  # or an np array or list or whatever

	# Bundle flights for each airline
	mercury_flights_per_airline = {}
	for flight in mercury_flights:
		mercury_flights_per_airline[flight.airlineName] = mercury_flights_per_airline.get(flight.airlineName, []) + [flight]

	# ------- Network Manager agent starts here ----- # 
	engine = Engine(algo=algo)

	# Cost function approximation to use
	# Not used for UDPP
	archetype_cost_function = 'jump'

	# Create Hotspot handler
	if algo == 'udpp_merge':
		archetype_cost_function = None
	
	hh_NM = HotspotHandler(engine=engine,
							archetype_cost_function=archetype_cost_function
							)

	# Create flights in hotspot handler and compute FPFS
	# This is composed of information only accessible to the NM.
	mercury_flights_dict = [{'flight_name':mf.name,
								'airline_name':mf.airlineName,
								'eta':mf.eta,
								} for mf in mercury_flights]

	hh_NM.prepare_hotspot_from_dict(attr_list=mercury_flights_dict,
									slot_times=slot_times) 

	# For each airline, get slot for each of their flight and send that to the airlines,
	# together with the archetype_cost_function
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

	# The following can be run async (even with shared slots? not sure...).
	all_messages = []
	for airline, mercury_flights_airline in mercury_flights_per_airline.items():
		# ------ Flight agent begins here ------ #
		# Receive the message from NM
		message_from_NM = to_be_sent_to_airlines[airline]

		# Prepare the engine function_approx that will compute the parameters 
		# to be given to the ISTOP engine
		if algo=='udpp_merge':
			algo_local = 'udpp_local'
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

		# Here we pass directly the real cost function, because we want to ask for an approximation
		# computed by the udpp_local engine.
		_, flights_airline = hh.prepare_hotspot_from_dict(attr_list=mercury_flights_dict,
															slots=message_from_NM['slots'],
															set_cost_function_with={'cost_function':'cost_function',
																					'kind':'lambda',
																					'absolute':False,
																					'eta':'eta'},
															)
		# Prepare flights for engine
		hh.prepare_all_flights()

		# Compute preferences (parameters approximating the cost function)
		preferences = engine_local.compute_optimal_parameters(hotspot_handler=hh)

		# Preferences are then sent out to the NM, together with other information (like airline name etc)
		to_be_sent_to_NM = {}
		for i, (name, pref) in enumerate(preferences.items()):
			to_be_sent_to_NM[name] = pref

			# ------- Flight agent ends here ------ #

		# Send to NM here
		all_messages.append(to_be_sent_to_NM)

	#print ('all_messages', all_messages)									
	# ------- Network Manager agent starts here again ----- # 
	# NM can now prepare the flights with the preferences sent by the airline
	# no need to specificy cost function here, preference parameters will be used instead,
	# with the archetype function already set above.

	if algo=='udpp_merge':
		set_cost_function_with = None
	else:
		set_cost_function_with = 'default_cf_paras'
	
	for message in all_messages:
		hh_NM.update_flight_attributes_int_from_dict(attr_list=message,
													set_cost_function_with=set_cost_function_with
													) 

	# Prepare the flights (compute cost vectors)
	hh_NM.prepare_all_flights()

	# Merge UDPP preferences
	allocation = engine.compute_optimal_allocation(hotspot_handler=hh_NM)
	print_allocation(allocation)

print ()
print ()
print ()

# Another example with weaker agent paradigm (pass direclty the cost vectors)
# This would work the same for ISTOP, NNBound, and GlobalOptimum.
print ("########### ISTOP merging (Mercury) (Weak agent paradigm) ############".format(algo))
mercury_flights = create_original_flights(n_f=10)
slot_times = list(range(0, 2*n_f, 2))  # or an np array or list or whatever
mercury_flights_per_airline = {}
for flight in mercury_flights:
	mercury_flights_per_airline[flight.airlineName] = mercury_flights_per_airline.get(flight.airlineName, []) + [flight]

# ------- Network Manager agent starts here ----- # 
engine = Engine(algo='istop')
# Here we don't need an archetype function
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
	# ------ Flight agent begins here ------ #
	message_from_NM = to_be_sent_to_airlines[airline]
	engine_local = LocalEngine(algo='function_approx')

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
	# Prepare flights for engine. This is where the cost vectors are computed
	hh.prepare_all_flights()
	preferences = hh.get_cost_vectors()

	# Preferences (cost vectors here) are then sent out to the NM, together with other information (like airline name etc)
	to_be_sent_to_NM = {}
	for i, (name, pref) in enumerate(preferences.items()):
		to_be_sent_to_NM[name] = pref

		# ------- Flight agent ends here ------ #

	all_messages.append(to_be_sent_to_NM)
									
# ------- Network Manager agent starts here again ----- # 

# We don't need to set a cost function here, the cost vectors are passed
# as attributes in the message.
for message in all_messages:
	hh_NM.update_flight_attributes_int_from_dict(attr_list=message) 

# Prepare the flights: does not do anything here (but does not hurt).
hh_NM.prepare_all_flights()

allocation = engine.compute_optimal_allocation(hotspot_handler=hh_NM)
print_allocation(allocation)

print ()