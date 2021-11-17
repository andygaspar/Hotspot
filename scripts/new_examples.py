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
import pickle

from Hotspot import models, models_correspondence_cost_vect, models_correspondence_approx
from Hotspot import Engine, LocalEngine, HotspotHandler, print_allocation
from Hotspot.libs.uow_tool_belt.general_tools import sort_lists

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

class ExternalFlight2:
	def __init__(self, name: str, airline_name: str, eta: float):
		self.eta = eta
		self.airlineName = airline_name
		self.name = name
		self.cost_coefficient = np.random.uniform(0.5, 2, 1)[0]
		self.cost_func = lambda delay: self.cost_coefficient * delay ** 2

def create_original_flights(n_f=50):
	assert n_f<=300
	if n_f>50:
		print ("You are using more than 50 flights from david_test.csv, be careful!!!")
	df = pd.read_csv("../test_data/david_test.csv")
	external_flights = []
	for i in list(range(df.shape[0]))[:n_f]:
		line = df.iloc[i]
		external_flights.append(ExternalFlight(line["flight"], line["airline"], line["time"], line["eta"]))
	
	return external_flights

def create_original_flights2(etas=[], flight_names=[], airline_names=[]):
	external_flights = []
	for i in range(len(etas)):
		eta = etas[i]
		flight_name = flight_names[i]
		airline_name = airline_names[i]
		flight = ExternalFlight2(eta=eta,
						name=flight_name,
						airline_name=airline_name)
		external_flights.append(flight)
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
	df = pd.read_csv("../test_data/david_test.csv").iloc[:n_f]
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
							alternative_allocation_rule=True)
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
							alternative_allocation_rule=True)

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
							alternative_allocation_rule=True)
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
							alternative_allocation_rule=True)

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
							alternative_allocation_rule=True)
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
							alternative_allocation_rule=True)

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
	#f_airline = 
	print ()
	print ("########### Custom example for Mercury {} ############".format(algo))
	n_f = len(f_eta)
	mercury_flights = create_original_flights(n_f=n_f)
	for i, flight in enumerate(mercury_flights):
		flight.eta = f_eta[i][1]
		#flight.airlineName = f_airline[i][1]


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
							alternative_allocation_rule=True)
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
							alternative_allocation_rule=True)

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

def other_examples3(algo='udpp_merge'):
	#Slots = [0:1191, 1:1215, 2:1233, 3:1242, 4:1248, 5:1260, 6:1263, 7:1308, 8:1320, 9:1335, 10:1338, 11:1341, 12:1344, 13:1362, 14:1365, 15:1368, 16:1371, 17:1398, 18:1467]
	f_eta = [1390.969, 1383.842, 1427.3139999999999, 1405.6979999999999, 1386.454, 1391.232, 1386.885, 1408.041, 1422.5639999999999, 1382.09, 1389.154, 1392.293, 1400.475, 1396.252, 1397.09, 1409.3899999999999, 1408.885, 1397.494, 1404.1879999999999]
	f_airline = [(8546, 2575), (8708, 2575), (9039, 2679), (9147, 2679), (9579, 2244), (9590, 2575), (9601, 2641), (9893, 2679), (9899, 2679), (10031, 2244), (10034, 2244), (11809, 2244), (10118, 2575), (10772, 2323), (10722, 2244), (10989, 2244), (11126, 2244), (11176, 2244), (11397, 2244)]
	slot_times = [1382, 1386, 1388, 1390, 1392, 1394, 1396, 1398, 1400, 1402, 1404, 1406, 1408, 1410, 1412, 1414, 1416, 1422, 1426]

	print ()
	print ("########### Custom example for Mercury {} ############".format(algo))
	n_f = len(f_eta)
	mercury_flights = create_original_flights(n_f=n_f)
	for i, flight in enumerate(mercury_flights):
		flight.eta = f_eta[i]
		flight.airlineName = f_airline[i][1]
	
	#print ('Flights/ETA:', [(flight.name, flight.eta) for flight in mercury_flights])
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
							alternative_allocation_rule=True)
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
							alternative_allocation_rule=True)

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

def other_examples4(algo='udpp_merge'):
	#Slots = [0:1191, 1:1215, 2:1233, 3:1242, 4:1248, 5:1260, 6:1263, 7:1308, 8:1320, 9:1335, 10:1338, 11:1341, 12:1344, 13:1362, 14:1365, 15:1368, 16:1371, 17:1398, 18:1467]
	f_eta = [1441.45, 1442.713, 1444.09, 1444.649, 1450.262, 1440.464, 1442.072, 1445.329]
	f_airline = [(7880, 2576), (8640, 2395), (9374, 2287), (9548, 2582), (10434, 2586), (11701, 2368), (12625, 2287), (12245, 2298)]
	slot_times = [1440, 1444, 1448, 1451, 1451, 1451, 1451, 1451]

	print ()
	print ("########### Custom example for Mercury {} ############".format(algo))
	n_f = len(f_eta)
	mercury_flights = create_original_flights(n_f=n_f)
	for i, flight in enumerate(mercury_flights):
		flight.eta = f_eta[i]
		flight.airlineName = f_airline[i][1]
	
	#print ('Flights/ETA:', [(flight.name, flight.eta) for flight in mercury_flights])
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
							alternative_allocation_rule=True)
	mercury_flights_dict = [{'flight_name':mf.name,
								'airline_name':mf.airlineName,
								'eta':mf.eta,
								} for mf in mercury_flights]
	hh_NM.prepare_hotspot_from_dict(attr_list=mercury_flights_dict,
									slot_times=slot_times) 
	#hh_NM.compute_FPFS()
	# Current allocation (FPFS)
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
							alternative_allocation_rule=True)

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

def other_examples5(algo='udpp_merge'):
	#Slots = [0:1191, 1:1215, 2:1233, 3:1242, 4:1248, 5:1260, 6:1263, 7:1308, 8:1320, 9:1335, 10:1338, 11:1341, 12:1344, 13:1362, 14:1365, 15:1368, 16:1371, 17:1398, 18:1467]
	f_eta = [1433.607, 1436.374, 1449.0430000000001, 1382.373, 1452.648, 1450.502, 1430.118, 1438.944, 1422.1689999999999, 1452.1, 1471.12, 1426.853, 1419.761, 1381.9199999999998, 1404.422, 1437.242, 1413.529, 1388.746, 1436.3449999999998, 1409.276, 1416.666, 1430.577, 1432.655, 1411.968, 1450.143, 1432.591, 1430.848, 1417.57, 1398.094, 1396.8010000000002, 1422.221, 1404.229, 1407.156, 1393.224, 1444.818, 1418.024, 1391.154, 1394.263, 1390.2259999999999, 1417.608, 1408.3229999999999, 1430.503, 1424.942, 1466.5797777722853, 1431.6709999999998, 1400.22, 1427.306, 1447.553, 1442.2839999999999, 1402.672, 1410.064, 1416.8899999999999, 1431.526, 1443.714, 1427.7630000000001, 1417.389, 1427.653, 1443.489, 1451.069, 1416.731, 1436.154, 1444.8509999999999, 1447.176, 1462.176, 1469.9166176733054, 1399.5210070045725, 1450.8182734907136]
	f_airline = [(7028, 2540), (7313, 2540), (7812, 2540), (8025, 2845), (8278, 2540), (8381, 2809), (8882, 2540), (8906, 2678), (8980, 2540), (8986, 2551), (9092, 2702), (9344, 2540), (9374, 2540), (9409, 2715), (11020, 2540), (9370, 2540), (9827, 2540), (9928, 2575), (9987, 2540), (9991, 2540), (9606, 2540), (10052, 2540), (10343, 3012), (10486, 2540), (10779, 2540), (10574, 2533), (10672, 2566), (10746, 2540), (10768, 2540), (10783, 2586), (10795, 2540), (10978, 2540), (10980, 2540), (10984, 2540), (10988, 2540), (10991, 2646), (9752, 2778), (11081, 2658), (11130, 2658), (11204, 2540), (11218, 2540), (11227, 2540), (11251, 2540), (7365, 2725), (11015, 2540), (11563, 2586), (11636, 2540), (11245, 3012), (11643, 2540), (11880, 2540), (11900, 2540), (12010, 2937), (12073, 2540), (12089, 2540), (12101, 2540), (12248, 2540), (12427, 2540), (13081, 2586), (13437, 2673), (13403, 2795), (13005, 2778), (13359, 2540), (13362, 2540), (13792, 2578), (9302, 2565), (9240, 2658), (9879, 2715)]
	slot_times = [1381, 1384, 1388, 1390, 1391, 1392, 1394, 1395, 1397, 1398, 1400, 1401, 1402, 1404, 1407, 1408, 1410, 1411, 1412, 1414, 1415, 1417, 1418, 1420, 1421, 1422, 1424, 1425, 1427, 1428, 1430, 1431, 1432, 1434, 1435, 1437, 1438, 1440, 1441, 1442, 1445, 1447, 1448, 1450, 1451, 1452, 1454, 1455, 1457, 1458, 1460, 1461, 1462, 1464, 1465, 1467, 1468, 1470, 1471, 1472, 1474, 1475, 1477, 1478, 1480, 1480, 1480]
	print ()
	print ("########### Custom example for Mercury {} ############".format(algo))
	n_f = len(f_eta)
	mercury_flights = create_original_flights(n_f=n_f)
	for i, flight in enumerate(mercury_flights):
		flight.eta = f_eta[i]
		flight.airlineName = f_airline[i][1]
	
	#print ('Flights/ETA:', [(flight.name, flight.eta) for flight in mercury_flights])
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
							alternative_allocation_rule=True)
	mercury_flights_dict = [{'flight_name':mf.name,
								'airline_name':mf.airlineName,
								'eta':mf.eta,
								} for mf in mercury_flights]
	hh_NM.prepare_hotspot_from_dict(attr_list=mercury_flights_dict,
									slot_times=slot_times) 
	#hh_NM.compute_FPFS()
	# Current allocation (FPFS)
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
							alternative_allocation_rule=True)

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

def other_examples6(algo='udpp_merge'):
	f_eta = [1617.817, 1591.55, 1622.315, 1603.712, 1621.759, 1620.7469999999998, 1600.946, 1584.556, 1598.742, 1612.628, 1626.974, 1586.415, 1612.435, 1612.148, 1629.5210000000002, 1599.492, 1608.1155168866874, 1600.1703964621527, 1660.8967263927457, 1648.1796221298284, 1597.5595093881843]
	f_airline = [(14592, 2269), (15361, 2613), (15646, 2613), (15662, 2367), (15676, 2613), (17087, 2280), (16923, 2268), (17135, 2613), (17280, 2613), (17483, 2531), (17576, 2613), (17604, 2613), (17687, 2613), (17704, 2408), (17849, 2613), (17894, 2613), (15645, 2613), (15659, 2613), (16178, 2291), (16254, 2613), (15937, 2613)]
	slot_times = [1586, 1588, 1590, 1596, 1598, 1600, 1601, 1603, 1605, 1606, 1611, 1613, 1615, 1616, 1620, 1621, 1623, 1628, 1630, 1646, 1660]
	print ()
	print ("########### Custom example for Mercury {} ############".format(algo))
	n_f = len(f_eta)
	mercury_flights = create_original_flights(n_f=n_f)
	for i, flight in enumerate(mercury_flights):
		flight.eta = f_eta[i]
		flight.airlineName = f_airline[i][1]
	
	#print ('Flights/ETA:', [(flight.name, flight.eta) for flight in mercury_flights])
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
							alternative_allocation_rule=True)
	mercury_flights_dict = [{'flight_name':mf.name,
								'airline_name':mf.airlineName,
								'eta':mf.eta,
								} for mf in mercury_flights]
	hh_NM.prepare_hotspot_from_dict(attr_list=mercury_flights_dict,
									slot_times=slot_times) 
	#hh_NM.compute_FPFS()
	# Current allocation (FPFS)
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
							alternative_allocation_rule=True)

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

def other_examples7(algo='udpp_merge'):
	slot_times = [1367.7272727272727, 1369.090909090909, 1371.8181818181818, 1373.1818181818182, 1375.909090909091, 1381.3636363636363, 1382.7272727272727, 1384.090909090909, 1385.4545454545455, 1386.8181818181818, 1388.1818181818182, 1389.5454545454545, 1392.2727272727273, 1393.6363636363637, 1395.0, 1396.3636363636363, 1397.7272727272727, 1399.090909090909, 1400.4545454545455, 1401.8181818181818, 1403.1818181818182, 1404.5454545454545, 1405.909090909091, 1407.2727272727273, 1408.6363636363637, 1410.0, 1411.3636363636363, 1412.7272727272727, 1414.090909090909, 1415.4545454545455, 1416.8181818181818, 1418.1818181818182, 1419.5454545454545, 1420.909090909091, 1422.2727272727273, 1423.6363636363637, 1425.0, 1426.3636363636363, 1427.7272727272727, 1430.4545454545455, 1431.8181818181818, 1433.1818181818182, 1435.909090909091, 1437.2727272727273, 1438.6363636363635, 1440.0, 1441.3636363636363, 1442.7272727272727, 1444.090909090909, 1445.4545454545455, 1446.8181818181818, 1448.1818181818182, 1449.5454545454545, 1450.909090909091, 1452.2727272727273, 1453.6363636363635, 1455.0, 1456.3636363636363, 1457.7272727272727, 1459.090909090909, 1460.4545454545455, 1461.8181818181818, 1463.1818181818182, 1464.5454545454545, 1465.0, 1466.111111111111, 1467.2222222222222, 1468.3333333333333, 1469.4444444444443, 1470.5555555555557, 1471.6666666666667, 1472.7777777777778, 1473.888888888889, 1475.0, 1476.111111111111, 1481.6666666666667, 1482.7777777777778, 1486.111111111111, 1490.5555555555557, 1491.6666666666667, 1492.7777777777778, 1496.111111111111, 1497.2222222222222, 1498.3333333333333, 1499.4444444444443, 1507.2222222222222, 1512.7777777777778]
	f_airline = [(1080, 702), (1062, 661), (1074, 661), (1086, 661), (1092, 753), (1100, 742), (1122, 661), (1124, 701), (1130, 661), (1132, 666), (1128, 661), (1142, 695), (1148, 661), (1156, 661), (1168, 658), (1174, 661), (1180, 674), (1184, 661), (1164, 661), (1186, 661), (1188, 792), (1190, 713), (1192, 661), (1204, 661), (1196, 658), (1198, 670), (1200, 661), (1202, 661), (1206, 678), (1208, 661), (1210, 659), (1214, 661), (1216, 661), (1218, 661), (1220, 661), (1222, 689), (1170, 737), (1230, 695), (1232, 695), (1236, 661), (1238, 661), (1240, 661), (1244, 661), (1078, 722), (1226, 661), (1246, 678), (1248, 661), (1242, 792), (1250, 661), (1252, 661), (1254, 661), (1258, 771), (1264, 661), (1266, 661), (1270, 661), (1272, 740), (1274, 661), (1276, 661), (1294, 678), (1300, 661), (1310, 699), (1280, 674), (1308, 739), (1286, 692), (1288, 734), (1290, 766), (1292, 737), (1296, 657), (1090, 780), (1304, 661), (1306, 661), (1098, 661), (1316, 766), (1318, 675), (1134, 661), (1144, 669), (1150, 661), (1328, 695), (1158, 661), (1228, 661), (1160, 661), (1162, 721), (1330, 713), (1154, 661), (1332, 713), (1178, 721), (1182, 661)]
	f_eta = [1469.441, 1424.998, 1425.422, 1440.283, 1373.093, 1441.222, 1419.173, 1428.104, 1409.446, 1445.641, 1366.755, 1372.608, 1366.503, 1409.441, 1440.409, 1404.249, 1382.162, 1399.996, 1406.866, 1419.737, 1427.535, 1365.422, 1396.859, 1441.383, 1422.791, 1422.526, 1407.77, 1391.349, 1389.601, 1412.941, 1512.794, 1394.949, 1395.754, 1383.424, 1439.698, 1408.224, 1383.434, 1386.023, 1387.985, 1403.62, 1398.523, 1421.223, 1415.142, 1444.5060556527485, 1421.351, 1391.46, 1418.026, 1442.023, 1431.964, 1391.297, 1401.824, 1407.61, 1422.766, 1436.5140000000001, 1415.883, 1482.832, 1413.829, 1418.373, 1436.289, 1497.213, 1443.869, 1491.225, 1406.411, 1495.132, 1487.217, 1492.877, 1428.434, 1475.689, 1507.6424555208569, 1435.571, 1443.616, 1467.4797561063974, 1492.179, 1458.616, 1397.0253681667077, 1471.8367314119298, 1481.7071043764936, 1476.023, 1389.5352554237943, 1448.2949142557209, 1395.5645448915632, 1423.2847972164782, 1499.701, 1467.1994468439282, 1498.021, 1417.6948575706058, 1449.0810754288987]

	# f_eta = [1617.817, 1591.55, 1622.315, 1603.712, 1621.759, 1620.7469999999998, 1600.946, 1584.556, 1598.742, 1612.628, 1626.974, 1586.415, 1612.435, 1612.148, 1629.5210000000002, 1599.492, 1608.1155168866874, 1600.1703964621527, 1660.8967263927457, 1648.1796221298284, 1597.5595093881843]
	# f_airline = [(14592, 2269), (15361, 2613), (15646, 2613), (15662, 2367), (15676, 2613), (17087, 2280), (16923, 2268), (17135, 2613), (17280, 2613), (17483, 2531), (17576, 2613), (17604, 2613), (17687, 2613), (17704, 2408), (17849, 2613), (17894, 2613), (15645, 2613), (15659, 2613), (16178, 2291), (16254, 2613), (15937, 2613)]
	# slot_times = [1586, 1588, 1590, 1596, 1598, 1600, 1601, 1603, 1605, 1606, 1611, 1613, 1615, 1616, 1620, 1621, 1623, 1628, 1630, 1646, 1660]
	print ()
	print ("########### Custom example for Mercury {} ############".format(algo))
	n_f = len(f_eta)
	# mercury_flights = create_original_flights(n_f=n_f)
	# for i, flight in enumerate(mercury_flights):
	# 	flight.eta = f_eta[i]
	# 	flight.airlineName = f_airline[i][1]

	mercury_flights = create_original_flights2(etas=f_eta,
												flight_names=list(zip(*f_airline))[0],
												airline_names=list(zip(*f_airline))[1]
												)
	#print ('Flights/ETA:', [(flight.name, flight.eta) for flight in mercury_flights])
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
							alternative_allocation_rule=True)
	mercury_flights_dict = [{'flight_name':mf.name,
								'airline_name':mf.airlineName,
								'eta':mf.eta,
								} for mf in mercury_flights]
	hh_NM.prepare_hotspot_from_dict(attr_list=mercury_flights_dict,
									slot_times=slot_times) 
	#hh_NM.compute_FPFS()
	# Current allocation (FPFS)
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
							alternative_allocation_rule=True)

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

def other_examples8(algo='udpp_merge'):

	slot_times = [1389.7368421052631, 1391.3157894736842, 1394.4736842105262,
				1400.7894736842106, 1403.9473684210527, 1405.5263157894738,
				1407.1052631578948, 1410.2631578947369, 1413.421052631579,
				1415.0, 1416.578947368421, 1418.157894736842, 1419.7368421052631,
				1421.3157894736842, 1422.8947368421052, 1424.4736842105262,
				1426.0526315789473, 1427.6315789473683, 1429.2105263157894,
				1430.7894736842106, 1432.3684210526317, 1433.9473684210527,
				1435.5263157894738, 1437.1052631578948, 1438.6842105263158,
				1440.2631578947369, 1441.842105263158, 1443.421052631579, 1445.0,
				1446.578947368421, 1449.7368421052631, 1451.3157894736842,
				1452.8947368421052, 1454.4736842105262, 1456.0526315789473,
				1457.6315789473683, 1459.2105263157896, 1460.7894736842104,
				1462.3684210526317, 1463.9473684210527, 1465.0, 1465.0, 1465.0,
				1465.0, 1465.0, 1465.0, 1465.0, 1465.0, 1465.0, 1465.0, 1465.0,
				1465.0, 1465.0]
	f_airline = [(1061, 660), (1085, 660), (1097, 660), (1099, 741), (1129, 660),
				(1149, 660), (1167, 657), (1173, 660), (1181, 660), (1183, 660),
				(1163, 660), (1185, 660), (1187, 791), (1195, 657), (1197, 669),
				(1199, 660), (1201, 660), (1215, 660), (1221, 688), (1237, 660),
				(1239, 660), (1243, 660), (1077, 721), (1225, 660), (1245, 677),
				(1247, 660), (1241, 791), (1249, 660), (1251, 660), (1253, 660),
				(1257, 770), (1263, 660), (1265, 660), (1269, 660), (1273, 660),
				(1275, 660), (1293, 677), (1309, 698), (1307, 738), (1291, 736),
				(1303, 660), (1305, 660), (1317, 674), (1121, 660), (1227, 660),
				(1153, 660), (1177, 720), (1191, 660), (1205, 677), (1207, 660),
				(1213, 660), (1229, 694), (1235, 660)]
	f_eta = [1429.089, 1444.996, 1440.267, 1441.719, 1411.215, 1418.093,
			1440.409, 1404.938, 1428.402, 1395.477, 1406.866, 1419.737,
			1425.647, 1426.42, 1421.568, 1407.77, 1391.349, 1397.876,
			1408.224, 1398.523, 1421.223, 1415.142, 1446.5250378128183,
			1421.351, 1391.46, 1418.026, 1440.353, 1431.964,
			1389.7368421052631, 1401.824, 1407.61, 1422.766,
			1436.5140000000001, 1415.883, 1413.829, 1418.373, 1436.289,
			1443.869, 1406.411, 1428.434, 1435.571, 1443.616, 1458.616,
			1462.7298419991648, 1449.9197655210844, 1443.3771787371873,
			1427.0572854530026, 1414.1262462831542, 1415.494416407422,
			1433.5979256817707, 1424.1818519737287, 1425.3339195397627,
			1421.7224285879436]

	# Does not work without the sorting!!!
	f_eta, f_airline = list(sort_lists(f_eta, f_airline))
	#print (f_airline[:5])
	#f_eta = sorted(f_eta)
	#print (f_eta[:5])
	# n = 10
	# slot_times = slot_times[:n]
	# f_eta = sorted(f_eta[:n])
	# f_airline = f_airline[:n]

	print ()
	print ("########### Custom example for Mercury {} ############".format(algo))
	n_f = len(f_eta)
	# mercury_flights = create_original_flights(n_f=n_f)
	# for i, flight in enumerate(mercury_flights):
	# 	flight.eta = f_eta[i]
	# 	flight.airlineName = f_airline[i][1]

	mercury_flights = create_original_flights2(etas=f_eta,
											flight_names=list(zip(*f_airline))[0],
											airline_names=list(zip(*f_airline))[1]
											)
	
	#print ('Flights/ETA:', [(flight.name, flight.eta) for flight in mercury_flights])
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
							alternative_allocation_rule=True)
	mercury_flights_dict = [{'flight_name':mf.name,
								'airline_name':mf.airlineName,
								'eta':mf.eta,
								} for mf in mercury_flights]
	hh_NM.prepare_hotspot_from_dict(attr_list=mercury_flights_dict,
									slot_times=slot_times) 
	#hh_NM.compute_FPFS()
	# Current allocation (FPFS)
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
							alternative_allocation_rule=True)

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

def other_examples9(algo='udpp_merge'):

	slot_times = [1386.578947368421, 1388.157894736842, 1389.7368421052631, 1391.3157894736842, 1392.8947368421052, 1394.4736842105262, 1397.6315789473683, 1400.7894736842106, 1402.3684210526317, 1403.9473684210527, 1405.5263157894738, 1407.1052631578948, 1408.6842105263158, 1410.2631578947369, 1411.842105263158, 1413.421052631579, 1418.157894736842, 1419.7368421052631, 1424.4736842105262, 1427.6315789473683, 1429.2105263157894, 1430.7894736842106, 1432.3684210526317, 1433.9473684210527, 1440.2631578947369, 1441.842105263158, 1443.421052631579, 1445.0, 1446.578947368421, 1448.157894736842, 1449.7368421052631, 1451.3157894736842, 1452.8947368421052, 1454.4736842105262, 1456.0526315789473, 1457.6315789473683, 1459.2105263157896, 1460.7894736842104, 1462.3684210526317, 1463.9473684210527, 1465.0, 1465.0, 1465.0, 1465.0, 1465.0, 1465.0, 1465.0, 1465.0, 1465.0, 1465.0, 1465.0, 1465.0]
	f_airline = [(1231, 694), (1201, 660), (1205, 677), (1251, 660), (1245, 677), (1183, 660), (1227, 660), (1237, 660), (1253, 660), (1229, 694), (1191, 660), (1235, 660), (1307, 738), (1163, 660), (1173, 660), (1257, 770), (1199, 660), (1155, 660), (1129, 660), (1207, 660), (1273, 660), (1243, 660), (1269, 660), (1247, 660), (1275, 660), (1121, 660), (1263, 660), (1239, 660), (1225, 660), (1153, 660), (1123, 700), (1291, 736), (1181, 660), (1249, 660), (1303, 660), (1219, 660), (1293, 677), (1265, 660), (1167, 657), (1099, 741), (1241, 791), (1097, 660), (1305, 660), (1309, 698), (1197, 669), (1213, 660), (1131, 665), (1077, 721), (1195, 657), (1177, 720), (1135, 715), (1317, 674)]
	f_eta = [1386.578947368421, 1388.294, 1389.601, 1391.297, 1391.46, 1395.477, 1395.552, 1398.523, 1401.824, 1402.1741002620283, 1402.688, 1405.252, 1406.411, 1406.866, 1407.082, 1407.61, 1407.77, 1409.441, 1409.446, 1412.941, 1413.829, 1415.142, 1415.883, 1418.026, 1418.373, 1420.318, 1420.588, 1421.223, 1421.351, 1425.467, 1428.104, 1428.434, 1431.225, 1431.964, 1435.571, 1435.848, 1436.289, 1436.5140000000001, 1440.409, 1441.222, 1442.023, 1442.633, 1443.616, 1443.869, 1444.2002925540696, 1444.5142006059477, 1445.641, 1447.298095544139, 1448.2723970948307, 1451.3448355726907, 1457.512, 1458.616]

	# with open('cost_matrix_debug.pic', 'rb') as f:
	# 	f_cost = pickle.load(f)

	# Does not work without the sorting!!!
	f_eta, f_airline = list(sort_lists(f_eta, f_airline))
	#print (f_airline[:5])
	#f_eta = sorted(f_eta)
	#print (f_eta[:5])
	# n = 10
	# slot_times = slot_times[:n]
	# f_eta = sorted(f_eta[:n])
	# f_airline = f_airline[:n]

	print ()
	print ("########### Custom example for Mercury {} ############".format(algo))
	n_f = len(f_eta)
	# mercury_flights = create_original_flights(n_f=n_f)
	# for i, flight in enumerate(mercury_flights):
	# 	flight.eta = f_eta[i]
	# 	flight.airlineName = f_airline[i][1]

	mercury_flights = create_original_flights2(etas=f_eta,
												flight_names=list(zip(*f_airline))[0],
												airline_names=list(zip(*f_airline))[1]
												)
	
	#print ('Flights/ETA:', [(flight.name, flight.eta) for flight in mercury_flights])
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
							alternative_allocation_rule=True)
	mercury_flights_dict = [{'flight_name':mf.name,
								'airline_name':mf.airlineName,
								'eta':mf.eta,
								} for mf in mercury_flights]
	hh_NM.prepare_hotspot_from_dict(attr_list=mercury_flights_dict,
									slot_times=slot_times) 
	#hh_NM.compute_FPFS()
	# Current allocation (FPFS)
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
							alternative_allocation_rule=True)

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

def other_examples10(algo='udpp_merge'):
	"""
	This example only works if you write the cost matrix down from Mercury,
	and copy/paste the slote times etc below
	"""
	slot_times = [1385.0, 1389.7368421052631, 1391.3157894736842, 1392.8947368421052, 1394.4736842105262, 1397.6315789473683, 1399.2105263157894, 1400.7894736842106, 1405.5263157894738, 1407.1052631578948, 1408.6842105263158, 1411.842105263158, 1413.421052631579, 1415.0, 1416.578947368421, 1418.157894736842, 1419.7368421052631, 1421.3157894736842, 1422.8947368421052, 1424.4736842105262, 1426.0526315789473, 1427.6315789473683, 1429.2105263157894, 1430.7894736842106, 1432.3684210526317, 1433.9473684210527, 1435.5263157894738, 1437.1052631578948, 1438.6842105263158, 1440.2631578947369, 1441.842105263158, 1443.421052631579, 1445.0, 1446.578947368421, 1448.157894736842, 1449.7368421052631, 1451.3157894736842, 1452.8947368421052, 1454.4736842105262, 1456.0526315789473, 1457.6315789473683, 1459.2105263157896, 1460.7894736842104, 1462.3684210526317, 1463.9473684210527, 1465.0, 1465.0, 1465.0, 1465.0, 1465.0, 1465.0, 1465.0, 1465.0, 1465.0, 1465.0, 1465.0, 1465.0, 1465.0, 1465.0, 1465.0]
	f_airline = [(1229, 694), (1113, 712), (1205, 677), (1245, 677), (1251, 660), (1213, 660), (1183, 660), (1191, 660), (1237, 660), (1253, 660), (1235, 660), (1307, 738), (1163, 660), (1257, 770), (1199, 660), (1129, 660), (1221, 688), (1207, 660), (1273, 660), (1243, 660), (1269, 660), (1121, 660), (1247, 660), (1149, 660), (1275, 660), (1231, 694), (1185, 660), (1173, 660), (1239, 660), (1225, 660), (1177, 720), (1263, 660), (1187, 791), (1195, 657), (1291, 736), (1155, 660), (1249, 660), (1201, 660), (1227, 660), (1073, 660), (1303, 660), (1181, 660), (1219, 660), (1143, 668), (1293, 677), (1265, 660), (1085, 660), (1241, 791), (1167, 657), (1099, 741), (1131, 665), (1215, 660), (1305, 660), (1309, 698), (1077, 721), (1153, 660), (1135, 715), (1317, 674), (1123, 700), (1097, 660)]
	f_eta = [1385.0, 1386.091863299702, 1389.601, 1391.46, 1393.392, 1394.949, 1395.477, 1396.859, 1398.523, 1401.824, 1405.252, 1406.411, 1406.866, 1407.61, 1407.77, 1409.446, 1409.736, 1412.941, 1413.829, 1415.142, 1415.883, 1416.804, 1418.026, 1418.093, 1418.373, 1419.422616353123, 1419.737, 1419.9721230909076, 1421.223, 1421.351, 1421.7215659331648, 1422.766, 1425.647, 1426.42, 1428.434, 1429.4834196264414, 1431.964, 1432.1983121532376, 1433.3330406530686, 1434.997, 1435.571, 1435.661, 1435.848, 1435.868, 1436.289, 1436.5140000000001, 1440.283, 1440.353, 1440.409, 1441.222, 1441.26, 1441.4583532479269, 1443.616, 1443.869, 1444.5531961677175, 1454.3478982032382, 1455.413, 1458.616, 1461.4715008010896, 1463.8347237901144]
	# Does not work without the sorting!!!
	f_eta, f_airline = list(sort_lists(f_eta, f_airline))
	# for i, (flight, airline) in enumerate(f_airline):
	# 	if flight==1205:
	# 		print ('FLEINFLEIN', flight, f_eta[i])
	#print (f_airline[:5])
	#f_eta = sorted(f_eta)
	#print (f_eta[:5])
	# n = 10
	# slot_times = slot_times[:n]
	# f_eta = sorted(f_eta[:n])
	# f_airline = f_airline[:n]

	print ()
	print ("########### Custom example for Mercury {} ############".format(algo))
	n_f = len(f_eta)
	# mercury_flights = create_original_flights(n_f=n_f)
	# for i, flight in enumerate(mercury_flights):
	# 	flight.eta = f_eta[i]
	# 	flight.airlineName = f_airline[i][1]

	mercury_flights = create_original_flights2(etas=f_eta,
												flight_names=list(zip(*f_airline))[0],
												airline_names=list(zip(*f_airline))[1]
												)
	for flight in mercury_flights:
		if flight.name==1205:
			print ('YAWHOL', flight.name, flight.eta)
	#print ('Flights/ETA:', [(flight.name, flight.eta) for flight in mercury_flights])
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
							alternative_allocation_rule=True)
	mercury_flights_dict = [{'flight_name':mf.name,
								'airline_name':mf.airlineName,
								'eta':mf.eta,
								} for mf in mercury_flights]
	hh_NM.prepare_hotspot_from_dict(attr_list=mercury_flights_dict,
									slot_times=slot_times) 
	#hh_NM.compute_FPFS()
	# Current allocation (FPFS)
	all_allocated_slots = hh_NM.get_allocation()
	print_allocation(all_allocated_slots)
	to_be_sent_to_airlines = {}
	for airline, flights_in_airline in mercury_flights_per_airline.items():
		message_to_airline = {}
		for flight in flights_in_airline:
			message_to_airline[flight] = {'slot':all_allocated_slots[flight.name]}
		message_to_airline['cost_func_archetype'] = cost_func_archetype
		message_to_airline['slots'] = hh_NM.slots #list(all_allocated_slots.values())
		to_be_sent_to_airlines[airline] = message_to_airline
	# ------- Network Manager agents ends here ----- # 
	
	print ('ALL airlines', list(mercury_flights_per_airline.keys()))
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
							alternative_allocation_rule=True)

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
		for flight in hh.get_flight_list():
			if flight.name==1199:
				print ('POUETPOUET', flight.name, flight.eta, flight.costVect)

		print ('Airline:', airline)
		# THIS IS JUST FOR BUG TRACKING
		cost_matrix = pd.read_csv('../../../cost_matrix_before.csv', index_col=0)
		
		print ('Flight characs before setting cost matrix:', [(f.name, f.eta, f.costVect[-3:]) for f in hh.get_flight_list()])
		hh.set_cost_matrix(cost_matrix)
		print ('Flight characs after setting cost matrix:', [(f.name, f.eta, f.costVect[-3:]) for f in hh.get_flight_list()])
		
		preferences = engine_local.compute_optimal_parameters(hotspot_handler=hh,
																kwargs_init={})
		to_be_sent_to_NM = {}
		for i, (name, pref) in enumerate(preferences.items()):
			if name in [1199, 1243, 1269, 1275]:
				print ('PREFFFFF:', pref)
				pref['jump'] = 100.
				print ('PREFFFFF:', pref)
			to_be_sent_to_NM[name] = pref

			# ------- Flight agent ends here ------ #
		all_messages.append(to_be_sent_to_NM)

		print ()
								
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
	#print ('BIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIM', hh_NM.get_flight_list())
	for flight in hh_NM.get_flight_list():
		if flight.name==1199:
			print ('POUETPOUETPOUET', flight.name, flight.eta, flight.costVect)

	# # THIS IS JUST FOR BUG TRACKING
	# # Apply cost matrix from Mercury to these flights
	# cost_matrix = pd.read_csv('../../../cost_matrix_debug.csv', index_col=0)
	# hh_NM.set_cost_matrix(cost_matrix)
	allocation = engine.compute_optimal_allocation(hotspot_handler=hh_NM,
													kwargs_init={} # due to a weird bug, this line is required
													)
	print_allocation(allocation)
	# print (engine.model.report)
	# print (engine.model.solution)

# TO test
"""
Slots: [1365.0, 1366.7142857142858, 1368.4285714285713, 1371.857142857143, 1373.5714285714287, 1378.7142857142858, 1380.4285714285713, 1382.142857142857, 1383.857142857143, 1385.5714285714287, 1387.0, 1388.3333333333333, 1391.0, 1393.6666666666667, 1395.0, 1396.3333333333333, 1397.6666666666667, 1399.0, 1400.3333333333333, 1401.6666666666667, 1404.3333333333333, 1405.6666666666667, 1407.0, 1408.3333333333333, 1409.6666666666667, 1411.0, 1412.3333333333333, 1413.6666666666667, 1415.0, 1416.3333333333333, 1417.6666666666667, 1419.0, 1420.3333333333333, 1421.6666666666667, 1423.0, 1424.3333333333333, 1425.6666666666667, 1427.0, 1428.3333333333333, 1429.6666666666667, 1431.0, 1432.3333333333333, 1433.6666666666667, 1435.0, 1437.6666666666667, 1439.0, 1440.3333333333333, 1441.6666666666667, 1443.0, 1444.3333333333333, 1445.6666666666667, 1447.0, 1448.3333333333333, 1449.6666666666667, 1451.0, 1452.3333333333333, 1453.6666666666667, 1455.0, 1456.3333333333333, 1457.6666666666667, 1459.0, 1460.3333333333333, 1461.6666666666667, 1463.0, 1464.3333333333333, 1465.6666666666667, 1467.0, 1468.3333333333333, 1469.6666666666667, 1471.0, 1472.3333333333333, 1473.6666666666667, 1475.0, 1476.3333333333333, 1477.6666666666667, 1479.0, 1480.3333333333333, 1481.6666666666667, 1483.0, 1487.1818181818182, 1491.5454545454545, 1492.6363636363637, 1493.7272727272727, 1494.8181818181818, 1495.909090909091, 1497.0, 1498.090909090909, 1512.2727272727273, 1513.3636363636363, 1515.5454545454545, 1516.6363636363637, 1529.7272727272727, 1530.8181818181818, 1549.3636363636363, 1560.2727272727273, 1566.8181818181818, 1574.4545454545455, 1576.6363636363635, 1578.8181818181818, 1593.0, 1594.090909090909, 1598.4545454545455, 1610.4545454545455, 1618.090909090909, 1619.1818181818182]
Flights/airlines: [(1080, 702), (1062, 661), (1078, 722), (1074, 661), (1086, 661), (1090, 780), (1092, 753), (1096, 698), (1122, 661), (1130, 661), (1132, 666), (1136, 716), (1128, 661), (1148, 661), (1150, 661), (1158, 661), (1154, 661), (1168, 658), (1178, 721), (1180, 674), (1184, 661), (1164, 661), (1186, 661), (1188, 792), (1190, 713), (1192, 661), (1204, 661), (1196, 658), (1198, 670), (1200, 661), (1202, 661), (1206, 678), (1208, 661), (1210, 659), (1214, 661), (1216, 661), (1218, 661), (1220, 661), (1222, 689), (1224, 661), (1170, 737), (1230, 695), (1232, 695), (1236, 661), (1238, 661), (1240, 661), (1244, 661), (1226, 661), (1246, 678), (1248, 661), (1242, 792), (1250, 661), (1252, 661), (1254, 661), (1258, 771), (1262, 659), (1264, 661), (1266, 661), (1270, 661), (1272, 740), (1274, 661), (1276, 661), (1294, 678), (1300, 661), (1310, 699), (1280, 674), (1308, 739), (1286, 692), (1288, 734), (1290, 766), (1292, 737), (1296, 657), (1298, 684), (1302, 655), (1304, 661), (1306, 661), (1312, 765), (1098, 661), (1100, 742), (1316, 766), (1318, 675), (1320, 716), (1322, 691), (1324, 797), (1326, 661), (1124, 701), (1138, 780), (1140, 661), (1362, 661), (1142, 695), (1144, 669), (1152, 661), (1328, 695), (1156, 661), (1228, 661), (1162, 721), (1330, 713), (1368, 671), (1160, 661), (1332, 713), (1334, 661), (1372, 713), (1174, 661), (1176, 749), (1182, 661)]
ETA: [1474.4279999999999, 1429.335, 1429.135, 1425.095, 1444.996, 1477.354, 1372.133, 1366.938, 1422.992, 1410.523, 1441.26, 1457.9, 1366.755, 1366.271, 1418.093, 1365.896, 1421.907, 1440.409, 1391.417, 1382.162, 1399.996, 1406.866, 1419.737, 1425.647, 1365.0, 1396.859, 1441.383, 1422.791, 1421.568, 1407.77, 1391.349, 1389.601, 1412.941, 1512.794, 1394.949, 1395.754, 1383.424, 1435.848, 1409.736, 1619.675, 1383.434, 1386.023, 1381.466, 1405.252, 1398.523, 1421.223, 1415.142, 1421.351, 1391.46, 1418.026, 1440.353, 1431.964, 1393.392, 1401.824, 1407.61, 1550.362, 1422.766, 1436.5140000000001, 1415.883, 1482.832, 1413.829, 1418.373, 1436.289, 1497.213, 1443.869, 1491.225, 1406.411, 1495.132, 1487.217, 1492.433, 1428.434, 1475.689, 1530.719, 1517.144, 1435.571, 1443.616, 1560.612, 1459.0632864365184, 1467.9812216354167, 1496.004, 1458.616, 1567.387, 1514.137, 1574.913, 1530.198, 1451.2364736727225, 1516.1357160597952, 1593.7308727605246, 1579.078, 1414.5979480794585, 1474.4767542787304, 1618.4752685655558, 1476.023, 1432.4012381377083, 1432.1547751538608, 1424.5783221884876, 1492.039, 1610.543, 1396.2468523532002, 1498.021, 1577.481, 1593.7060000000001, 1428.1972835208906, 1599.1244470440324, 1460.0164943586387]
"""
if __name__=='__main__':
	for algo in ['udpp', 'istop', 'nnbound', 'globaloptimum']:
		examples_direct_cost_vector(algo=algo)

	for algo in ['istop_approx', 'nnbound_approx', 'globaloptimum_approx']:
	#for algo in ['istop_approx']:
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

	for algo in ['udpp_merge', 'nnbound', 'globaloptimum']: # istop not working yet
		other_examples3(algo=algo)

	for algo in ['udpp_merge', 'nnbound', 'globaloptimum']: # istop not working yet
		other_examples4(algo=algo)

	# for algo in ['udpp_merge', 'nnbound', 'globaloptimum']: # istop not working yet
	# 	other_examples5(algo=algo)

	print ('############# Example 6 ##############')
	for algo in ['udpp_merge', 'nnbound', 'udpp_merge_istop', 'globaloptimum']: # istop not working yet
		other_examples6(algo=algo)

	print ('############# Example 7 ##############')	
	for algo in ['udpp_merge', 'nnbound', 'udpp_merge_istop', 'globaloptimum']: # istop not working yet
	#for algo in ['istop']:#, 'globaloptimum']: # istop not working yet
		other_examples7(algo=algo)

	print ('############# Example 8 ##############')		
	for algo in ['udpp_merge', 'nnbound', 'udpp_merge_istop', 'globaloptimum']: # istop not working yet
	#for algo in ['istop']: # istop not working yet
		other_examples8(algo=algo)

	print ('############# Example 9 ##############')		
	for algo in ['udpp_merge', 'nnbound', 'udpp_merge_istop', 'globaloptimum']: # istop not working yet
		other_examples9(algo=algo)

	# print ('############# Example 10 ##############')		
	# for algo in ['udpp_merge', 'nnbound', 'udpp_merge_istop', 'globaloptimum']: # istop not working yet
	# #for algo in ['udpp_merge_istop']: # istop not working yet
	# 	other_examples10(algo=algo)
