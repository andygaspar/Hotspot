"""
Version using the wrapper
"""

import sys
sys.path.insert(1, '../..')

import numpy as np
from typing import Callable
import pandas as pd

from Hotspot import models, print_allocation
from Hotspot import HotspotHandler, OptimalAllocationEngine as Engine, LocalEngine

np.random.seed(0)

# First bit to simulate things coming from another model, e.g. 
# flight objects

class DavidFlight:
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
		david_flights.append(DavidFlight(line["flight"], line["airline"], line["time"], line["eta"]))
	
	return david_flights

n_f = 10


print ('Available models:', models.keys())
print ()

print ("\n################### First Example ####################")

# From your model we get
david_flights = create_original_flights(n_f=10)
slot_times = list(range(0, 2*n_f, 2))  # or an np array or list or whatever

# Build slots and flights for model. 
# david_flights is unmodified by this operation.
# paras is dic to get the attributes needed inside the flights
# in the david_flights list. The keys are the names of the 
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
hotspot_handler = HotspotHandler()
slots, flights = hotspot_handler.prepare_hotspot_from_flights_ext(flights_ext=david_flights,
															slot_times=slot_times,
															attr_map={'name':'flight_name', #
																	'airlineName':'airline_name',
																	'eta':'eta'
																	},
															set_cost_function_with={'cost_function':'cost_func',
																					'kind':'lambda',
																					'absolute':False,
																					'eta':'eta'})

print("NN bound")

# Initialise model
computer = Engine(algo='nnbound')

# Run model
allocation = computer.compute_optimal_allocation(hotspot_handler=hotspot_handler)

# Allocation is an ordered dict linking flight -> slot
print_allocation(allocation)
computer.print_optimisation_performance()

# Put slots in the original flight object as attribute "attr_slot_ext"
hotspot_handler.assign_slots_to_flights_ext_from_allocation(allocation, attr_slot_ext='slot')
print ('Slot assigned to first flight:', david_flights[0].slot)

# One can also use this method to use internal flights object to get the slot, instead of the
# allocation dict.
hotspot_handler.update_flight_attributes_int_to_ext({'newSlot':'slot'})
print ('Slot assigned to last flight:', david_flights[-1].slot)

# You can also get another shiny list of flights
new_flights = hotspot_handler.get_new_flight_list()
print (new_flights[0])


print ("\n################### Second Example ####################")

print("NN bound with detached functions")
# Another example where the cost function is not attached to the flight
david_flights = create_original_flights(n_f=10)
slot_times = list(range(0, 2*n_f, 2))  # or an np array or list or whatever
def build_random_cost_function(margin=25, jump=100.):
	def f(delay):
		# Relative cost function (input is delay)
		if delay<margin*np.random.uniform(0.5, 1.5):
			return 0.
		else:
			return jump*np.random.uniform(0.5, 1.5)
	return f

cost_functions = [build_random_cost_function() for flight in david_flights]
hotspot_handler = HotspotHandler()
slots, flights = hotspot_handler.prepare_hotspot_from_flights_ext(flights_ext=david_flights,
															slot_times=slot_times,
															attr_map={'name':'flight_name', #
																	'airlineName':'airline_name',
																	'eta':'eta'
																	},
															set_cost_function_with={'cost_function':cost_functions,
																					'kind':'lambda',
																					'absolute':False,
																					'eta':'eta'})
# Initialise model
computer = Engine(algo='nnbound')
# Run model
allocation = computer.compute_optimal_allocation(hotspot_handler=hotspot_handler)
# Allocation is an ordered dict linking flight -> slot
print_allocation(allocation)


print ("\n################### Third Example ####################")


# Another example where we use a dataframe to build the flights.
# In this case, a parametrisable cost function can be passed.
# One can also pass a list of cost functions in this case, just like
# above.
print ("NNBound with flights from dataframe")
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

hotspot_handler = HotspotHandler()
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
computer = Engine(algo='nnbound')
allocation = computer.compute_optimal_allocation(hotspot_handler=hotspot_handler)
computer.print_optimisation_performance()

print ("\n################### Fourth Example ####################")

# Another example where we compute priorities for each airline independently and
# merge everything afterwards.

print ("UDPP merging")
david_flights = create_original_flights(n_f=10)
slot_times = list(range(0, 2*n_f, 2))  # or an np array or list or whatever

# Bundle flights for each airline
david_flights_per_airline = {}
for flight in david_flights:
	david_flights_per_airline[flight.airlineName] = david_flights_per_airline.get(flight.airlineName, []) + [flight]

# Shortcut to define cost function.
cost_func_dict = {'cost_function':'cost_func',
				'kind':'lambda',
				'absolute':False,
				'eta':'eta'}

# ------- Network Manager agent starts here ----- # 
# Start by registering flights in the hotspot. This includes computing FPFS allocation.
engine = Engine(algo='udpp_merge')
hotspot_handler_NM = HotspotHandler(engine=engine)
slots, flights_NM = hotspot_handler_NM.prepare_hotspot_from_flights_ext(flights_ext=david_flights,
																slot_times=slot_times,
																attr_map={'name':'flight_name',
																		'airlineName':'airline_name',
																		'eta':'eta',
																		},
																set_cost_function_with=cost_func_dict)
# In model, FPFS allocation is sent back to flight or airline agent
# Here we include it in the .slot attribute in flight objects.
hotspot_handler_NM.update_flight_attributes_int_to_ext({'slot':'slot'})

# Note that the NM can cheat at this stage by using the udpp optimiser to compute priorities, but
# here we assume that this is a privilege of the Flight or the Airline object in the model.
# ------- Network Manager agents ends here ----- # 
# This can be run async.
for airline, david_flights_airline in david_flights_per_airline.items():
	# ------ Flight agent begins here ------ #
	# One easy way would be to pass the hotspot_handler above to the NM, but this breaks the agent paradigm.
	# So instead we create a flight handler per airline, but make sure to use the already created slots.
	# For this, one needs to map the 'slot' attribute and to pass the list of slots (instead of the list of slot_times)
	# FPFS computation is deactivated whenever this argument is passed.
	hotspot_handler = HotspotHandler()
	slots, flights_airline = hotspot_handler.prepare_hotspot_from_flights_ext(flights_ext=david_flights_airline,
																slots=slots,
																attr_map={'name':'flight_name',
																		'airlineName':'airline_name',
																		'eta':'eta',
																		'slot':'slot'
																		},
																set_cost_function_with=cost_func_dict,
																)


	# Compute priorities
	engine_local = LocalEngine(algo='udpp_local')
	#print ('flights', flights)
	priorities = engine_local.compute_optimal_parameters(hotspot_handler=hotspot_handler)

	# To send back the priorities, one can send a dictionary or assign
	# priorities back to original flight object.
	hotspot_handler.update_flight_attributes_dict_to_ext(priorities)
	# ------- Flight agent ends here ------ #
															
# ------- Network Manager agent starts here again ----- # 
# One can update the hotspot with the extra attributes now present in the original flight objects
hotspot_handler_NM.update_flight_attributes_ext_to_int(attr_map={#'udppPriority':'udpp_priority', 
															'udppPriority':'udppPriority', 	
															'udppPriorityNumber':'udppPriorityNumber',
															'tna':'tna'
															})
#flights = hotspot_handler_NM.get_flights()

hotspot_handler_NM.prepare_all_flights()

# Merge udpp priorities
allocation = engine.compute_optimal_allocation(hotspot_handler=hotspot_handler_NM,
												kwargs_run={'optimised':True})
print_allocation(allocation)

print ("\n################### Fourth and a half Example ####################")

# Same than previous but with Istop.

print ("ISTOP merging")
david_flights = create_original_flights(n_f=10)
slot_times = list(range(0, 2*n_f, 2))  # or an np array or list or whatever

# Bundle flights for each airline
david_flights_per_airline = {}
for flight in david_flights:
	david_flights_per_airline[flight.airlineName] = david_flights_per_airline.get(flight.airlineName, []) + [flight]

# Shortcut to define cost function.
cost_func_dict = {'cost_function':'cost_func',
				'kind':'lambda',
				'absolute':False,
				'eta':'eta'}

# ------- Network Manager agent starts here ----- # 
# Start by registering flights in the hotspot. This includes computing FPFS allocation.
engine = Engine(algo='istop')
hotspot_handler_NM = HotspotHandler(engine=engine)
slots, flights_NM = hotspot_handler_NM.prepare_hotspot_from_flights_ext(flights_ext=david_flights,
																slot_times=slot_times,
																attr_map={'name':'flight_name',
																		'airlineName':'airline_name',
																		'eta':'eta',
																		},
																set_cost_function_with=cost_func_dict)
# In model, FPFS allocation is sent back to flight or airline agent
# Here we include it in the .slot attribute in flight objects.
hotspot_handler_NM.update_flight_attributes_int_to_ext({'slot':'slot'})

# Note that the NM can cheat at this stage by using the udpp optimiser to compute priorities, but
# here we assume that this is a privilege of the Flight or the Airline object in the model.
# ------- Network Manager agents ends here ----- # 
# This can be run async.
for airline, david_flights_airline in david_flights_per_airline.items():
	# ------ Flight agent begins here ------ #
	# One easy way would be to pass the hotspot_handler above to the NM, but this breaks the agent paradigm.
	# So instead we create a flight handler per airline, but make sure to use the already created slots.
	# For this, one needs to map the 'slot' attribute and to pass the list of slots (instead of the list of slot_times)
	# FPFS computation is deactivated whenever this argument is passed.
	hotspot_handler = HotspotHandler()
	slots, flights_airline = hotspot_handler.prepare_hotspot_from_flights_ext(flights_ext=david_flights_airline,
																slots=slots,
																attr_map={'name':'flight_name',
																		'airlineName':'airline_name',
																		'eta':'eta',
																		'slot':'slot'
																		},
																set_cost_function_with=cost_func_dict,
																)


	# Compute priorities
	engine_local = LocalEngine(algo='udpp_local_function_approx')
	#print ('flights', flights)
	priorities = engine_local.compute_optimal_parameters(hotspot_handler=hotspot_handler)

	# To send back the priorities, one can send a dictionary or assign
	# priorities back to original flight object.
	hotspot_handler.update_flight_attributes_dict_to_ext(priorities)
	# ------- Flight agent ends here ------ #
															
# ------- Network Manager agent starts here again ----- # 
# One can update the hotspot with the extra attributes now present in the original flight objects
hotspot_handler_NM.update_flight_attributes_ext_to_int(attr_map={'slope':'slope', 
																'margin1':'margin1',
																'jump1':'jump1',
																'margin2':'margin2',
																'jump2':'jump2',
																})
flights = hotspot_handler_NM.get_flights()

# Merge udpp priorities
allocation = engine.compute_optimal_allocation(hotspot_handler=hotspot_handler_NM)
print_allocation(allocation)


print ("\n################### 4.75th Example ####################")

# Use case for Mercury.

print ("ISTOP merging (Mercury)")
mercury_flights = create_original_flights(n_f=10)
slot_times = list(range(0, 2*n_f, 2))  # or an np array or list or whatever

# Bundle flights for each airline
mercury_flights_per_airline = {}
for flight in mercury_flights:
	mercury_flights_per_airline[flight.airlineName] = mercury_flights_per_airline.get(flight.airlineName, []) + [flight]

# ------- Network Manager agent starts here ----- # 
hotspot_handler_NM = HotspotHandler()
# Only request slot creation
slots = hotspot_handler_NM.compute_slots(slot_times=slot_times)
# Slots then need to be sent to the airlines.
# ------- Network Manager agents ends here ----- # 

# The following can be run async (even with shared slots?).
all_flights = []
for airline, mercury_flights_airline in mercury_flights_per_airline.items():
	# ------ Flight agent begins here ------ #
	hotspot_handler = HotspotHandler()
	mercury_flights_dict = [{'flight_name':mf.name,
							'airline_name':mf.airlineName,
							'eta':mf.eta,
							'cost_function':mf.cost_func # pass real cost function here
							} for mf in mercury_flights_airline]
	_, flights_airline = hotspot_handler.prepare_hotspot_from_dict(flights_ext=mercury_flights_dict,
																		slots=slots,
																		set_cost_function_with={'cost_function':'cost_function',
																								'kind':'lambda',
																								'absolute':False,
																								'eta':'eta'}
																		)

	# Compute priorities
	computer = Engine(algo='istop')
	preferences = computer.compute_preferences(slots, flights_airline)

	# Preferences are then sent out to the NM, together with other information (like airline name etc)
	for i, (name, pref) in enumerate(preferences.items()):
		to_be_sent_to_NM = {'airline_name':airline,
							'flight_name':name,
							'eta':mercury_flights_airline[i].eta}
		for k, v in pref.items():
			to_be_sent_to_NM[k] = v

		# ------- Flight agent ends here ------ #

		# Send to NM here
		all_flights.append(to_be_sent_to_NM)
															
# ------- Network Manager agent starts here again ----- # 
# NM can now prepare the flights with the preferences sent by the airline
# no need to specificy cost function here, preference parameters will be used instead.
hotspot_handler_NM.prepare_flights_from_dict(flights_ext=all_flights) 

# Merge ISTOP preferences
computer = Engine(algo='istop')
allocation = computer.compute_optimal_allocation(slots,
												hotspot_handler_NM.get_flights())
print_allocation(allocation)

print ("\n################### Fifth Example ####################")

# Another example where we use udpp first, and then the global optimiser
david_flights = create_original_flights(n_f=10)
slot_times = list(range(0, 2*n_f, 2))  # or an np array or list or whatever
hotspot_handler = HotspotHandler()
slots, flights = hotspot_handler.prepare_hotspot_from_flights_ext(flights_ext=david_flights,
															slot_times=slot_times,
															attr_map={'name':'flight_name', #
																	'airlineName':'airline_name',
																	'eta':'eta'
																	},
															set_cost_function_with={'cost_function':'cost_func',
																					'kind':'lambda',
																					'absolute':False,
																					'eta':'eta'})


print("\nUDPP+GlobalOptimum")
# Initialise model
computer = Engine(algo='udpp')
# Run model
allocation = computer.compute_optimal_allocation(slots, flights, kwargs_run={'optimised':True})
print ('Allocation out of UDPP:')
print_allocation(allocation)
computer.print_optimisation_performance()

# Update the internal flight objects with the new slots
hotspot_handler.update_slots_internal()

# And run globaloptimum, starting from the UDPP output.
computer = Engine(algo='globaloptimum')
# Run model
allocation = computer.compute_optimal_allocation(slots, flights)
print ('Allocation out of Global Optimiser:')
print_allocation(allocation)
computer.print_optimisation_performance()


print ("\n################### Sixth Example ####################")

# with Istop
david_flights = create_original_flights(n_f=10)
slot_times = list(range(0, 2*n_f, 2))  # or an np array or list or whatever
hotspot_handler = HotspotHandler()
slots, flights = hotspot_handler.prepare_hotspot_from_flights_ext(flights_ext=david_flights,
															slot_times=slot_times,
															attr_map={'name':'flight_name', #
																	'airlineName':'airline_name',
																	'eta':'eta'
																	},
															set_cost_function_with={'cost_function':'cost_func',
																					'kind':'lambda',
																					'absolute':False,
																					'eta':'eta'})
print("\nIstop")
# Initialise model
computer = Engine(algo='istop')
# Run model
allocation = computer.compute_optimal_allocation(slots, flights, kwargs_init={'triples':False})
#print ('Allocation out of UDPP:')
print_allocation(allocation)
computer.print_optimisation_performance()

# Update the internal flight objects with the new slots
hotspot_handler.update_slots_internal()

