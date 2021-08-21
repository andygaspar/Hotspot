#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '../..')

import numpy as np
from typing import Callable
from collections import OrderedDict
import pandas as pd

from Hotspot import models, models_correspondence_cost_vect, models_correspondence_approx
from Hotspot import Engine, LocalEngine, HotspotHandler, print_allocation

np.random.seed(0)


class ExternalFlight:
	def __init__(self, name: str, airline_name: str, time: float, eta: float,
		cost_coefficient=1.):
		self.eta = eta
		self.airlineName = airline_name
		self.time = time
		self.name = name
		self.cost_coefficient = cost_coefficient#np.random.uniform(0.5, 2, 1)[0]
		self.cost_func = lambda delay: self.cost_coefficient * delay ** 2


def example_delta_t(algo='globaloptimum', delta_t=0.):
	print ("\n################### Delta t={} ({}) ###################".format(delta_t, algo))
	
	cap_drop = 2
	#etas = [0.5, 2.1, 3.5, 5., 5.5]
	#etas = [0., 2.1, 3., 5., 6.] can't put floats apparently
	#etas = [0, 4, 5, 8, 10]
	etas = [0, 1, 3, 5]
	cc = [1., 0.01, 10., 0.01]
	#cc = [1., 1., 1., 1.]
	external_flights = [ExternalFlight('F'+str(i), 'A0', cap_drop*i, eta, cost_coefficient=cc[i]) for i, eta in enumerate(etas)]
	slot_times = list(range(0, cap_drop*len(etas), cap_drop))

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
	hotspot_handler.prepare_all_flights()
	print ('FPFS allocation:')
	print_allocation (hotspot_handler.get_allocation())
	allocation = engine.compute_optimal_allocation(hotspot_handler=hotspot_handler,
													kwargs_init={'delta_t':delta_t})
	allocation_int = OrderedDict([(flight, slot.index) for flight, slot in allocation.items()])
		
	if delta_t <2.:
		assert allocation_int == OrderedDict([('F{}'.format(i), i) for i in range(len(etas))])
	else:
		assert allocation_int == OrderedDict([('F0', 0), ('F2', 1), ('F1', 2), ('F3', 3)])

	# Allocation is an ordered dict linking flight -> slot
	print ('Optimal allocation:')
	print_allocation(allocation)
	engine.print_optimisation_performance()

def example_automatic(algo='globaloptimum'):
	print ("\n################### Automatic delta_t setting ({}) ###################".format(algo))
	
	cap_drop = 2
	etas = [0, 1, 3, 5]
	cc = [1., 0.01, 10., 0.01]
	#cc = [1., 1., 1., 1.]
	external_flights = [ExternalFlight('F'+str(i), 'A0', cap_drop*i, eta, cost_coefficient=cc[i]) for i, eta in enumerate(etas)]
	slot_times = list(range(0, cap_drop*len(etas), cap_drop))

	engine = Engine(algo=algo)
	hotspot_handler = HotspotHandler(engine=engine,
									alternative_slot_allocation_rule=True)
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
	hotspot_handler.prepare_all_flights()
	print ('FPFS allocation:')
	print_allocation (hotspot_handler.get_allocation())
	allocation = engine.compute_optimal_allocation(hotspot_handler=hotspot_handler,
													kwargs_init={})
	allocation_int = OrderedDict([(flight, slot.index) for flight, slot in allocation.items()])
		
	assert allocation_int == OrderedDict([('F0', 0), ('F2', 1), ('F1', 2), ('F3', 3)])

	# Allocation is an ordered dict linking flight -> slot
	print ('Optimal allocation:')
	print_allocation(allocation)
	engine.print_optimisation_performance()

if __name__=='__main__':
	for algo in ['globaloptimum', 'nnbound', 'udpp']:
		example_delta_t(algo=algo, delta_t=0.)
		example_delta_t(algo=algo, delta_t=2.)
		example_automatic(algo=algo)

