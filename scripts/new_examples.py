#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Version using the wrapper
"""

import sys
sys.path.insert(1, '../..')

import numpy as np
from typing import Callable
import pandas as pd

from Hotspot import models, models_correspondence_cost_vect, models_correspondence_approx
from Hotspot import Engine, LocalEngine, HotspotHandler, print_allocation

np.random.seed(0)

# First bit to simulate things coming from another model, e.g. 
# flight objects
class ExternalFlight:
	def __init__(self, name: str, airline_name: str, time: float, eta: float):
		self.eta = eta
		self.airlineName = airline_name
		self.time = time
		self.name = name
		self.cost_coefficient = np.random.uniform(0.5, 2, 1)[0]
		self.cost_func = lambda delay: self.cost_coefficient * delay ** 2

def create_original_flights(n_f=50):
	df = pd.read_csv("david_test.csv")
	external_flights = []
	for i in list(range(df.shape[0]))[:n_f]:
		line = df.iloc[i]
		external_flights.append(ExternalFlight(line["flight"], line["airline"], line["time"], line["eta"]))
	
	return external_flights

n_f = 10


print ('Available models:', models.keys())
print ()

def examples_direct_cost_vector(algo='istop'):
	"""
	These examples feature more direct computaion, to be used
	for instance for RL.

	In this case, one can use external flight object directly to build
	the internal object (instead of a dictionary) and then update the 
	external flight object with the new slots.

	These examples use cost vectors.
	"""

	print ("\n################### First direct example ({}) ####################".format(algo))
	external_flights = create_original_flights(n_f=10)
	slot_times = list(range(0, 2*n_f, 2))  # or an np array or list or whatever

	# Build slots and flights for model. 
	# external_flights is unmodified by this operation.
	# paras is dic to get the attributes needed inside the flights
	# in the external_flights list. The keys are the names of the 
	# attributes in the original flight object, the values are
	# the correponding names to be passed to the internal flight object
	# A cost function and cost vector can automatically be built wiht the 
	# dictionary set_cost_function_with. The key "cost_function"
	# should point to an attribute in the original object. The "kind"
	# parameter should be set to "lambda" if in the original object
	# the function is a simple lambda function time -> cost or 
	# delay -> cost. 'absolute' refers to whether the original cost function
	# takes the delay as argument or the absolute time of the slot.
	# finally, if the former case, one needs to specificy the eta. The field
	# is designated with the value corresponding to the key 'eta'
	# Initialise model
	engine = Engine(algo=algo)

	hotspot_handler = HotspotHandler(engine=engine)
	slots, flights = hotspot_handler.prepare_hotspot_from_flights_ext(flights_ext=external_flights,
																		slot_times=slot_times,
																		attr_map={'name':'flight_name',
																				'airlineName':'airline_name',
																				'eta':'eta'
																				},
																		set_cost_function_with={'cost_function':'cost_func',
																								'kind':'lambda',
																								'absolute':False,
																								'eta':'eta'})
	# This computes the cost vectors
	hotspot_handler.prepare_all_flights()

	# Run model
	allocation = engine.compute_optimal_allocation(hotspot_handler=hotspot_handler)

	# Allocation is an ordered dict linking flight -> slot
	print_allocation(allocation)
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
	print ()

	print ("\n################### Second direct example, with detached functions ({}) ####################".format(algo))

	# Another example where the cost function is not attached to the flight
	external_flights = create_original_flights(n_f=10)
	slot_times = list(range(0, 2*n_f, 2))  # or an np array or list or whatever

	def build_random_cost_function(margin=25, jump=100.):
		def f(delay):
			# Relative cost function (input is delay)
			if delay<margin*np.random.uniform(0.5, 1.5):
				return 0.
			else:
				return jump*np.random.uniform(0.5, 1.5)
		return f

	cost_functions = [build_random_cost_function() for flight in external_flights]
	
	# Initialise model
	engine = Engine(algo=algo)

	hotspot_handler = HotspotHandler(engine=engine)
	slots, flights = hotspot_handler.prepare_hotspot_from_flights_ext(flights_ext=external_flights,
																slot_times=slot_times,
																attr_map={'name':'flight_name',
																		'airlineName':'airline_name',
																		'eta':'eta'
																		},
																set_cost_function_with={'cost_function':cost_functions,
																						'kind':'lambda',
																						'absolute':False,
																						'eta':'eta'})
	hotspot_handler.prepare_all_flights()

	# Run model
	allocation = engine.compute_optimal_allocation(hotspot_handler=hotspot_handler)
	
	# Allocation is an ordered dict linking flight -> slot
	print_allocation(allocation)
	print ()

	print ("\n################### Third direct Example, from dataframe ({}) ####################".format(algo))

	# Another example where we use a dataframe to build the flights.
	# In this case, a parametrisable cost function can be passed.
	# One can also pass a list of cost functions in this case, just like
	# above.
	df = pd.read_csv("david_test.csv").iloc[:n_f]
	# Add margins and jumps to df
	df['margin'] = np.random.uniform(5., 30., size=len(df))
	df['jump'] = np.random.uniform(20., 100., size=len(df))
	print (df)

	def custom_cost_function(delay, margin=None, jump=None):
		# Relative cost function (input is delay)
		if delay<margin:
			return 0.
		else:
			return jump

	engine = Engine(algo=algo)
	hotspot_handler = HotspotHandler(engine=engine)
	slots, flights = hotspot_handler.prepare_hotspot_from_dataframe(df=df,
																slot_times=slot_times,
																attr_map={'flight':'flight_name',
																		'airline':'airline_name',
																		'eta':'eta',
																		'margin':'margin',
																		'jump':'jump'
																		},
																set_cost_function_with={'cost_function':custom_cost_function,
																						'kind':'lambda',
																						'absolute':False,
																						'eta':'eta',
																						'kwargs':['margin', 'jump']
																						})
	hotspot_handler.prepare_all_flights()

	allocation = engine.compute_optimal_allocation(hotspot_handler=hotspot_handler)
	engine.print_optimisation_performance()

def examples_direct_approx(algo='istop_total'):
	"""
	These examples are similar to the previous ones, but they use approximated functions
	with an archetype.

	Works for instance with istop_total, nn_bound_total etc.
	"""
	print ("\n################### Direct example ({}) with archetype function ####################".format(algo))
	external_flights = create_original_flights(n_f=10)
	slot_times = list(range(0, 2*n_f, 2))  # or an np array or list or whatever

	engine = Engine(algo=algo)

	cost_func_archetype = 'jump'
	hotspot_handler = HotspotHandler(engine=engine,
									cost_func_archetype=cost_func_archetype)
	
	slots, flights = hotspot_handler.prepare_hotspot_from_flights_ext(flights_ext=external_flights,
																		slot_times=slot_times,
																		attr_map={'name':'flight_name',
																				'airlineName':'airline_name',
																				'eta':'eta'
																				},
																		set_cost_function_with={'cost_function':'cost_func',
																								'kind':'lambda',
																								'absolute':False,
																								'eta':'eta'})
	# This computes the cost vectors if needed.
	hotspot_handler.prepare_all_flights()

	# Run model
	allocation = engine.compute_optimal_allocation(hotspot_handler=hotspot_handler,
												)

	# Allocation is an ordered dict linking flight -> slot
	print_allocation(allocation)
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
	print ()

def example_agent_paradigm_vect(algo='udpp_merge'):
	"""
	This example follows a strong agent-based paradigm. The NM starts by initialising 
	the allocation engine, then sends to each airline the slots available. The airline
	computes the cost vector for each slot. The airline sends this back to the
	NM, which computes the best allocation.

	Note that the way flights are built in the local engine (in the airline) and the global engine
	(in the NM) is through dictionaries. This is the preferred way of communication in this case.
	"""

	print ()
	print ("########### Agent-based {} with cost vectors ############".format(algo))
	# mercury_flights = load_test(test_number)
	# slot_times = list(range(0, 2*len(mercury_flights), 2))

	mercury_flights = create_original_flights(n_f=10)
	slot_times = list(range(0, 2*n_f, 2))  # or an np array or list or whatever

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
	all_allocated_slots = hh_NM.get_allocation()
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
		# ------ Flight agent starts here ------- #
		message_from_NM = to_be_sent_to_airlines[airline]
		
		algo_local = models_correspondence_cost_vect[algo]

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

	allocation = engine.compute_optimal_allocation(hotspot_handler=hh_NM,
													kwargs_init={} # due to a weird bug, this line is required
													)
	
	print_allocation(allocation)
	# print (engine.model.report)
	# print (engine.model.solution)

def example_agent_paradigm_approx(algo='istop'):
	"""
	This is equivalent to the previous one, except the airlines provide some
	parameters that are used by the global engine, and not directly the cost 
	vectors. For instance, the airline returns the UDPPpriority parameter if
	udpp_merge is used.

	Note: this does not make sense with UDPP merge, but one can 
	"""

	print ()
	print ("########### Agent-based {} with approximated functions ############".format(algo))
	# mercury_flights = load_test(test_number)
	# slot_times = list(range(0, 2*len(mercury_flights), 2))

	mercury_flights = create_original_flights(n_f=10)
	slot_times = list(range(0, 2*n_f, 2))  # or an np array or list or whatever

	mercury_flights_per_airline = {}
	for flight in mercury_flights:
		mercury_flights_per_airline[flight.airlineName] = mercury_flights_per_airline.get(flight.airlineName, []) + [flight]

	# ------- Network Manager agent starts here ----- # 
	engine = Engine(algo=algo)
	cost_func_archetype = 'jump'
	hh_NM = HotspotHandler(engine=engine,
							cost_func_archetype=cost_func_archetype
							)
	mercury_flights_dict = [{'flight_name':mf.name,
								'airline_name':mf.airlineName,
								'eta':mf.eta,
								} for mf in mercury_flights]
	hh_NM.prepare_hotspot_from_dict(attr_list=mercury_flights_dict,
									slot_times=slot_times) 
	all_allocated_slots = hh_NM.get_allocation()
	to_be_sent_to_airlines = {}
	for airline, flights_in_airline in mercury_flights_per_airline.items():
		message_to_airline = {}
		for flight in flights_in_airline:
			message_to_airline[flight] = {'slot':all_allocated_slots[flight.name]}
		message_to_airline['cost_func_archetype'] = cost_func_archetype
		message_to_airline['slots'] = list(all_allocated_slots.values())
		to_be_sent_to_airlines[airline] = message_to_airline
	# ------- Network Manager agents ends here ----- # 
	
	all_messages = []
	for airline, mercury_flights_airline in mercury_flights_per_airline.items():
		# ------ Flight agent starts here ------ #
		message_from_NM = to_be_sent_to_airlines[airline]

		algo_local = models_correspondence_approx[algo]

		engine_local = LocalEngine(algo=algo_local)

		hh = HotspotHandler(engine=engine_local,
							cost_func_archetype=message_from_NM['cost_func_archetype']
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

	allocation = engine.compute_optimal_allocation(hotspot_handler=hh_NM,
													kwargs_init={} # due to a weird bug, this line is required
													)
	print_allocation(allocation)
	# print (engine.model.report)
	# print (engine.model.solution)

def example_agent_paradigm_approx_alternate(algo='istop'):
	"""
	Idem but with alternative rule for allocation
	"""

	print ()
	print ("########### Agent-based {} with approximated functions and alternate allocation rule ############".format(algo))
	mercury_flights = create_original_flights(n_f=10)
	slot_times = list(range(0, 2*n_f, 2))  # or an np array or list or whatever

	mercury_flights_per_airline = {}
	for flight in mercury_flights:
		mercury_flights_per_airline[flight.airlineName] = mercury_flights_per_airline.get(flight.airlineName, []) + [flight]

	# ------- Network Manager agent starts here ----- # 
	engine = Engine(algo=algo)
	cost_func_archetype = 'jump'
	hh_NM = HotspotHandler(engine=engine,
							cost_func_archetype=cost_func_archetype,
							alternative_slot_allocation_rule=True)
	mercury_flights_dict = [{'flight_name':mf.name,
								'airline_name':mf.airlineName,
								'eta':mf.eta,
								} for mf in mercury_flights]
	hh_NM.prepare_hotspot_from_dict(attr_list=mercury_flights_dict,
									slot_times=slot_times) 
	all_allocated_slots = hh_NM.get_allocation()
	to_be_sent_to_airlines = {}
	for airline, flights_in_airline in mercury_flights_per_airline.items():
		message_to_airline = {}
		for flight in flights_in_airline:
			message_to_airline[flight] = {'slot':all_allocated_slots[flight.name]}
		message_to_airline['cost_func_archetype'] = cost_func_archetype
		message_to_airline['slots'] = list(all_allocated_slots.values())
		to_be_sent_to_airlines[airline] = message_to_airline
	# ------- Network Manager agents ends here ----- # 
	
	all_messages = []
	for airline, mercury_flights_airline in mercury_flights_per_airline.items():
		# ------ Flight agent starts here ------ #
		message_from_NM = to_be_sent_to_airlines[airline]

		algo_local = models_correspondence_approx[algo]

		engine_local = LocalEngine(algo=algo_local)

		hh = HotspotHandler(engine=engine_local,
							cost_func_archetype=message_from_NM['cost_func_archetype'],
							alternative_slot_allocation_rule=True)

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

	allocation = engine.compute_optimal_allocation(hotspot_handler=hh_NM,
													kwargs_init={} # due to a weird bug, this line is required
													)
	print_allocation(allocation)
	# print (engine.model.report)
	# print (engine.model.solution)

def example_mercury_test_case(algo='udpp_merge'):
	print ()
	print ("########### Mercury test case with {} ############".format(algo))
	mercury_flights = create_original_flights(n_f=10)
	slot_times = list(range(0, 2*n_f, 2))  # or an np array or list or whatever

	mercury_flights_per_airline = {}
	for flight in mercury_flights:
		mercury_flights_per_airline[flight.airlineName] = mercury_flights_per_airline.get(flight.airlineName, []) + [flight]

	# ------- Network Manager agent starts here ----- # 
	engine = Engine(algo=algo)
	if not algo == 'udpp_merge':
		cost_func_archetype = 'jump'
	else:
		cost_func_archetype = None
	hh_NM = HotspotHandler(engine=engine,
							cost_func_archetype=cost_func_archetype,
							alternative_slot_allocation_rule=True)
	mercury_flights_dict = [{'flight_name':mf.name,
								'airline_name':mf.airlineName,
								'eta':mf.eta,
								} for mf in mercury_flights]
	hh_NM.prepare_hotspot_from_dict(attr_list=mercury_flights_dict,
									slot_times=slot_times) 
	all_allocated_slots = hh_NM.get_allocation()
	to_be_sent_to_airlines = {}
	for airline, flights_in_airline in mercury_flights_per_airline.items():
		message_to_airline = {}
		for flight in flights_in_airline:
			message_to_airline[flight] = {'slot':all_allocated_slots[flight.name]}
		message_to_airline['cost_func_archetype'] = cost_func_archetype
		message_to_airline['slots'] = list(all_allocated_slots.values())
		to_be_sent_to_airlines[airline] = message_to_airline
	# ------- Network Manager agents ends here ----- # 
	
	all_messages = []
	for airline, mercury_flights_airline in mercury_flights_per_airline.items():
		# ------ Flight agent starts here ------ #
		message_from_NM = to_be_sent_to_airlines[airline]

		if algo=='udpp_merge':
			algo_local = models_correspondence_cost_vect[algo]
		else:
			algo_local = models_correspondence_approx[algo]

		engine_local = LocalEngine(algo=algo_local)

		hh = HotspotHandler(engine=engine_local,
							cost_func_archetype=message_from_NM['cost_func_archetype'],
							alternative_slot_allocation_rule=True)

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
		preferences = engine_local.compute_optimal_parameters(hotspot_handler=hh,
																kwargs_init={})
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

	allocation = engine.compute_optimal_allocation(hotspot_handler=hh_NM,
													kwargs_init={} # due to a weird bug, this line is required
													)
	print_allocation(allocation)
	# print (engine.model.report)
	# print (engine.model.solution)

def other_examples(algo='udpp_merge'):
	print ()
	print ("########### Only one flight {} ############".format(algo))
	n_f = 1
	mercury_flights = create_original_flights(n_f=n_f)
	mercury_flights[0].eta = 1
	slot_times = list(range(0, 2*n_f, 2))  # or an np array or list or whatever

	mercury_flights_per_airline = {}
	for flight in mercury_flights:
		mercury_flights_per_airline[flight.airlineName] = mercury_flights_per_airline.get(flight.airlineName, []) + [flight]

	# ------- Network Manager agent starts here ----- # 
	engine = Engine(algo=algo)
	if not algo == 'udpp_merge':
		cost_func_archetype = 'jump'
	else:
		cost_func_archetype = None
	hh_NM = HotspotHandler(engine=engine,
							cost_func_archetype=cost_func_archetype,
							alternative_slot_allocation_rule=True)
	mercury_flights_dict = [{'flight_name':mf.name,
								'airline_name':mf.airlineName,
								'eta':mf.eta,
								} for mf in mercury_flights]
	hh_NM.prepare_hotspot_from_dict(attr_list=mercury_flights_dict,
									slot_times=slot_times) 
	all_allocated_slots = hh_NM.get_allocation()
	to_be_sent_to_airlines = {}
	for airline, flights_in_airline in mercury_flights_per_airline.items():
		message_to_airline = {}
		for flight in flights_in_airline:
			message_to_airline[flight] = {'slot':all_allocated_slots[flight.name]}
		message_to_airline['cost_func_archetype'] = cost_func_archetype
		message_to_airline['slots'] = hh_NM.slots
		to_be_sent_to_airlines[airline] = message_to_airline
	# ------- Network Manager agents ends here ----- # 
	
	all_messages = []
	for airline, mercury_flights_airline in mercury_flights_per_airline.items():
		# ------ Flight agent starts here ------ #
		message_from_NM = to_be_sent_to_airlines[airline]

		if algo=='udpp_merge':
			algo_local = models_correspondence_cost_vect[algo]
		else:
			algo_local = models_correspondence_approx[algo]

		engine_local = LocalEngine(algo=algo_local)

		hh = HotspotHandler(engine=engine_local,
							cost_func_archetype=message_from_NM['cost_func_archetype'],
							alternative_slot_allocation_rule=True)

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
		preferences = engine_local.compute_optimal_parameters(hotspot_handler=hh,
																kwargs_init={})
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

	allocation = engine.compute_optimal_allocation(hotspot_handler=hh_NM,
													kwargs_init={} # due to a weird bug, this line is required
													)
	print_allocation(allocation)
	# print (engine.model.report)
	# print (engine.model.solution)

def other_examples2(algo='udpp_merge'):
	#Slots = [0:1191, 1:1215, 2:1233, 3:1242, 4:1248, 5:1260, 6:1263, 7:1308, 8:1320, 9:1335, 10:1338, 11:1341, 12:1344, 13:1362, 14:1365, 15:1368, 16:1371, 17:1398, 18:1467]
	f_eta = [(5354, 1320), (5283, 1310), (5505, 1191), (5626, 1244), (5712, 1371), (5913, 1399), (6199, 1338), (6292, 1369), (6475, 1343), (6633, 1336), (6635, 1362), (6668, 1363), (6682, 1342)]
	
	print ()
	print ("########### Custom example for Mercury {} ############".format(algo))
	n_f = len(f_eta)
	mercury_flights = create_original_flights(n_f=n_f)
	for i, flight in enumerate(mercury_flights):
		flight.eta = f_eta[i][1]

	#print ('Flights/ETA:', [(flight.name, flight.eta) for flight in mercury_flights])
	slot_times = [1191, 1215, 1233, 1242, 1248, 1260, 1263, 1308, 1320, 1335, 1338, 1341, 1344, 1362, 1365, 1368, 1371, 1398, 1467]#list(range(0, 2*n_f, 2))  # or an np array or list or whatever

	mercury_flights_per_airline = {}
	for flight in mercury_flights:
		mercury_flights_per_airline[flight.airlineName] = mercury_flights_per_airline.get(flight.airlineName, []) + [flight]

	# ------- Network Manager agent starts here ----- # 
	engine = Engine(algo=algo)
	if not algo == 'udpp_merge':
		cost_func_archetype = 'jump'
	else:
		cost_func_archetype = None
	hh_NM = HotspotHandler(engine=engine,
							cost_func_archetype=cost_func_archetype,
							alternative_slot_allocation_rule=True)
	mercury_flights_dict = [{'flight_name':mf.name,
								'airline_name':mf.airlineName,
								'eta':mf.eta,
								} for mf in mercury_flights]
	hh_NM.prepare_hotspot_from_dict(attr_list=mercury_flights_dict,
									slot_times=slot_times) 
	# FPFS allocation
	all_allocated_slots = hh_NM.get_allocation()
	to_be_sent_to_airlines = {}
	for airline, flights_in_airline in mercury_flights_per_airline.items():
		message_to_airline = {}
		for flight in flights_in_airline:
			message_to_airline[flight] = {'slot':all_allocated_slots[flight.name]}
		message_to_airline['cost_func_archetype'] = cost_func_archetype
		message_to_airline['slots'] = hh_NM.slots #list(all_allocated_slots.values())
		to_be_sent_to_airlines[airline] = message_to_airline
	# ------- Network Manager agents ends here ----- # 
	
	all_messages = []
	for airline, mercury_flights_airline in mercury_flights_per_airline.items():
		# ------ Flight agent starts here ------ #
		message_from_NM = to_be_sent_to_airlines[airline]

		if algo=='udpp_merge':
			algo_local = models_correspondence_cost_vect[algo]
		else:
			algo_local = models_correspondence_approx[algo]

		engine_local = LocalEngine(algo=algo_local)

		#print ('Creating hotspot handler for airline', airline)
		hh = HotspotHandler(engine=engine_local,
							cost_func_archetype=message_from_NM['cost_func_archetype'],
							alternative_slot_allocation_rule=True)

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
		preferences = engine_local.compute_optimal_parameters(hotspot_handler=hh,
																kwargs_init={})
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

	allocation = engine.compute_optimal_allocation(hotspot_handler=hh_NM,
													kwargs_init={} # due to a weird bug, this line is required
													)
	print_allocation(allocation)
	# print (engine.model.report)
	# print (engine.model.solution)

if __name__=='__main__':
	for algo in ['udpp', 'istop', 'nnbound', 'globaloptimum']:
		examples_direct_cost_vector(algo=algo)

	#for algo in ['istop_approx', 'nnbound_approx', 'globaloptimum_approx']:
	for algo in ['istop_approx']:
		examples_direct_approx(algo=algo)

	for algo in ['udpp_merge', 'istop', 'nnbound', 'globaloptimum']:
		example_agent_paradigm_vect(algo=algo)

	for algo in ['istop', 'nnbound', 'globaloptimum']:
		example_agent_paradigm_approx(algo=algo)

	for algo in ['nnbound', 'globaloptimum']: # istop not working yet
		example_agent_paradigm_approx_alternate(algo=algo)

	for algo in ['udpp_merge', 'nnbound', 'globaloptimum']: # istop not working yet
		example_mercury_test_case(algo=algo)

	for algo in ['udpp_merge', 'nnbound', 'globaloptimum']: # istop not working yet
		other_examples(algo=algo)

	for algo in ['udpp_merge', 'nnbound', 'globaloptimum']: # istop not working yet
		other_examples2(algo=algo)
