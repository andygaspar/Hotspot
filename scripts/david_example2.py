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


# From your model we get
n_f = 10
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

print ('Available models:', models.keys())
print ()

print("\nnn bound")
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


# Another example where we use a dataframe to build the flights.
# In this case, a parametrisable cost function can be passed.
print ("\n nnBound with flights from dataframe")
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

