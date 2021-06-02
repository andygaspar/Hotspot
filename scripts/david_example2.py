"""
Version using the wrapper
"""

import sys
sys.path.insert(1, '../..')

import numpy as np
from typing import Callable
import pandas as pd

from Hotspot import models, print_allocation
from Hotspot import FlightHandler, OptimalAllocationComputer as OptAllComp

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
flight_handler = FlightHandler()
slots, flights = flight_handler.prepare_hotspot_from_objects(flights_ext=david_flights,
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
computer = OptAllComp(algo='nnbound')
# Run model
allocation = computer.compute_optimal_allocation(slots, flights)
# Allocation is an ordered dict linking flight -> slot
print_allocation(allocation)
computer.print_optimisation_performance()
# Put slots in the original flight object as attribute "attr_slot_ext"
flight_handler.assign_slots_to_objects_from_allocation(allocation, attr_slot_ext='slot')
print ('Slot assigned to first flight:', david_flights[0].slot)
# One can also use this method to use internal flights object to get the slot, instead of the
# allocation dict.
flight_handler.assign_slots_to_objects_from_internal_flights(attr_slot_ext='slot',
															attr_slot_int='newSlot')
print ('Slot assigned to last flight:', david_flights[-1].slot)

# You can also get another shiny list of flights
new_flights = flight_handler.get_new_flight_list()
print (new_flights[0])


print ("\n################### Second Example ####################")

print("NN bound with detached functions")
# Another example where we the cost function is not attached to the flight
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
flight_handler = FlightHandler()
slots, flights = flight_handler.prepare_hotspot_from_objects(flights_ext=david_flights,
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
computer = OptAllComp(algo='nnbound')
# Run model
allocation = computer.compute_optimal_allocation(slots, flights)
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

flight_handler = FlightHandler()
slots, flights = flight_handler.prepare_hotspot_from_dataframe(df=df,
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
computer = OptAllComp(algo='nnbound')
allocation = computer.compute_optimal_allocation(slots, flights)
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
flight_handler_NM = FlightHandler()
slots, flights_NM = flight_handler_NM.prepare_hotspot_from_objects(flights_ext=david_flights,
																slot_times=slot_times,
																attr_map={'name':'flight_name',
																		'airlineName':'airline_name',
																		'eta':'eta',
																		},
																set_cost_function_with=cost_func_dict)
# In model, FPFS allocation is sent back to flight or airline agent
# Here we include it in the .slot attribute in flight objects.
flight_handler_NM.update_flight_attributes_int_to_ext({'slot':'slot'})

# Note that the NM can cheat at this stage by using the udpp optimiser to compute priorities, but
# here we assume that this is a privilege of the Flight or the Airline object in the model.
# ------- Network Manager agents ends here ----- # 

for airline, david_flights_airline in david_flights_per_airline.items():
	# ------ Flight agent begins here ------ #
	# One easy way would be to pass the flight_handler above to the NM, but this breaks the agent paradigm.
	# So instead we create a flight handler per airline, but make sure to use the already created slots.
	# For this, one needs to map the 'slot' attribute and to pass the list of slots (instead of the list of slot_times)
	# FPFS computation is deactivated whenever this argument is passed.
	flight_handler = FlightHandler()
	slots, flights_airline = flight_handler.prepare_hotspot_from_objects(flights_ext=david_flights_airline,
																slots=slots,
																attr_map={'name':'flight_name',
																		'airlineName':'airline_name',
																		'eta':'eta',
																		'slot':'slot'
																		},
																set_cost_function_with=cost_func_dict,
																)


	# Compute priorities
	computer = OptAllComp(algo='udpp')
	#print ('flights', flights)
	priorities = computer.compute_local_priorities(slots, flights_airline)

	# To send back the priorities, one can send a dictionary or assign
	# priorities back to original flight object.
	flight_handler.assign_priorities_to_objects(priorities)
	# ------- Flight agent ends here ------ #
															
# ------- Network Manager agent starts here again ----- # 
# One can update the hotspot with the extra attributes now present in the original flight objects
flight_handler_NM.update_flight_attributes_ext_to_int(attr_map={'udppPriority':'udpp_priority', 
															'udppPriorityNumber':'udpp_priority_number',
															'tna':'tna'
															})
flights = flight_handler_NM.get_flights()

# Merge udpp priorities
computer = OptAllComp(algo='udpp')
allocation = computer.compute_optimal_allocation(slots,
												flights,
												kwargs_run={'optimised':True})
print_allocation(allocation)


print ("\n################### Fifth Example ####################")

# Another example where we use udpp first, and then the global optimiser
david_flights = create_original_flights(n_f=10)
slot_times = list(range(0, 2*n_f, 2))  # or an np array or list or whatever
flight_handler = FlightHandler()
slots, flights = flight_handler.prepare_hotspot_from_objects(flights_ext=david_flights,
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
computer = OptAllComp(algo='udpp')
# Run model
allocation = computer.compute_optimal_allocation(slots, flights, kwargs_run={'optimised':True})
print ('Allocation out of UDPP:')
print_allocation(allocation)
computer.print_optimisation_performance()

# Update the internal flight objects with the new slots
flight_handler.update_slots_internal()

# And run globaloptimum, starting from the UDPP output.
computer = OptAllComp(algo='globaloptimum')
# Run model
allocation = computer.compute_optimal_allocation(slots, flights)
print ('Allocation out of Global Optimiser:')
print_allocation(allocation)
computer.print_optimisation_performance()


print ("\n################### Sixth Example ####################")

# with Istop
david_flights = create_original_flights(n_f=10)
slot_times = list(range(0, 2*n_f, 2))  # or an np array or list or whatever
flight_handler = FlightHandler()
slots, flights = flight_handler.prepare_hotspot_from_objects(flights_ext=david_flights,
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
computer = OptAllComp(algo='istop')
# Run model
allocation = computer.compute_optimal_allocation(slots, flights, kwargs_init={'triples':False})
#print ('Allocation out of UDPP:')
print_allocation(allocation)
computer.print_optimisation_performance()

# Update the internal flight objects with the new slots
flight_handler.update_slots_internal()

