from collections import OrderedDict
import inspect

import numpy as np
import pandas as pd
import inspect

from ..ModelStructure.Slot.slot import Slot
from ..ModelStructure.Flight.flight import Flight as HFlight
from ..Istop.istop import Istop
from ..NNBound.nnBound import NNBoundModel
from ..UDPP.udppMerge import UDPPMerge
from ..UDPP.udppLocal import UDPPLocal
from ..UDPP.functionApprox import FunctionApprox
from ..GlobalOptimum.globalOptimum import GlobalOptimum
from ..ModelStructure.Costs.costFunctionDict import archetypes_cost_functions
from ..libs.uow_tool_belt.general_tools import write_on_file as print_to_void, clock_time
from ..combined_models import UDPPMergeIstop, UDPPLocalFunctionApprox, UDPPTotal#, UDPPFullIstop
from ..combined_models import IstopTotalApprox, NNBoundTotalApprox, GlobalOptimumTotalApprox
#from Hotspot.Istop.AirlineAndFlight.istopFlight import set_automatic_preference_vect

models = {'istop':Istop,
		'nnbound':NNBoundModel,
		'udpp_merge':UDPPMerge,
		'globaloptimum':GlobalOptimum,
		'udpp_local':UDPPLocal,
		'function_approx':FunctionApprox,
		'udpp_local_function_approx':UDPPLocalFunctionApprox,
		'udpp_merge_istop':UDPPMergeIstop,
		#'udpp_full_istop':UDPPFullIstop,
		'udpp':UDPPTotal,
		'istop_approx':IstopTotalApprox,
		'globaloptimum_approx':GlobalOptimumTotalApprox,
		'nnbound_approx':NNBoundTotalApprox}

# Models (values) to be run with models (keys) if cost vectors are used.
models_correspondence_cost_vect = {'istop':'get_cost_vectors',
								'udpp_merge':'udpp_local',
								'nnbound':'get_cost_vectors',
								'globaloptimum':'get_cost_vectors'}

# Models (values) to be run with models (keys) if approximation are used.
# Note: one can always make a combined model out of these combination, see 
# combined_models.py
models_correspondence_approx = {'istop':'function_approx',
								'globaloptimum':'function_approx',
								'nnbound':'function_approx',
								'udpp_merge_istop':'udpp_local_function_approx'}

def allocation_from_df(df, name_slot='new slot'):
	return OrderedDict(df[['flight', name_slot]].set_index('flight').to_dict()[name_slot])

def allocation_from_flights(flights, name_slot='newSlot'):
	#return OrderedDict([(flight.name, getattr(flight, name_slot).index) for flight in flights])
	return OrderedDict(sorted([(flight.name, getattr(flight, name_slot)) for flight in flights], key=lambda x:x[1].time))

def df_from_flights(flights, name_slot='newSlot'):
	"""
	Mostly for showing purposes, is not used for internal calculations anymore.
	"""
	a = {}

	for flight in flights.values():
		a["slot"] = a.get("slot", []) + [getattr(flight, name_slot).index]
		a["flight"] = a.get("flight", []) + [flight.name]
		a["eta"] = a.get("eta", []) + [flight.eta]
		a["fpfs"] = a.get("fpfs", []) + [flight.eta]
		a["time"] = a.get("time", []) + [getattr(flight, name_slot).time]
		#a["priority"] = a.get("priority", []) + [flight.udpp_priority]
		a["margins"] = a.get("margins", []) + [flight.margin1]
		a["margins_declared"] = a.get("margins_declared", []) + [flight.margin1_declared]
		a["airline"] = a.get("airline", []) + [flight.airlineName]
		a["slope"] = a.get("slope", []) + [flight.slope]
		#a["num"] = 
		a["jump"] = a.get("jump", []) + [flight.jump1]
		a["jump_declared"] = a.get("jump_declared", []) + [flight.jump1_declared]
		a["real_cost"] = a.get("real_cost", []) + [flight.cost_f_true(getattr(flight, name_slot).time)]
		a["declared_cost"] = a.get("declared_cost", []) + [flight.cost_f_declared(getattr(flight, name_slot).time)]

	df = pd.DataFrame(a, index=list(range(len(a["slot"]))))
	df.sort_values("slot", inplace=True)
	return df

def assign_FPFS_slot(slots, flights, delta_t=0.):
	# TODO: TO BE MODIFIED: flight should be assigned to the first slot with x.time>x.eta-delta_t
	# USe compatibleSlots in flight?
	flights_ordered = sorted(flights.values(), key=lambda x:x.eta)
	# slots_ordered = sorted(slots, key=lambda x:x.time)

	# for i, flight in enumerate(flights_ordered):
	# 	flight.slot = slots_ordered[i]

	assigned = []

	for flight in flights_ordered:
		compatible_slots = [slot for slot in slots if slot.time >= flight.eta-delta_t]
		for slot in compatible_slots:
			if not slot in assigned:
				flight.slot = slot
				assigned.append(slot)
				break

def compute_delta_t_from_slots(slots):
	return min([slots[i+1].time - slots[i].time for i in range(len(slots)-1)])

class HotspotHandler:
	"""
	In the following, "ext" corresponds to external flight objects,
	created outside of the Hotspot scope. "int" refers to internal
	flight objects.

	This is the main interface between the engine and the user.
	"""
	def __init__(self, engine=None, cost_func_archetype=None, alternative_slot_allocation_rule=False):
		self.engine = engine

		if not cost_func_archetype is None:
			self.set_archetype_cost_function(cost_func_archetype)

		if not engine is None:
			self.set_engine(engine)

		self.alternative_slot_allocation_rule = alternative_slot_allocation_rule

	def set_engine(self, engine):
		self.engine = engine

	def get_requirements_from_engine(self):
		return self.engine.get_requirements()

	def set_archetype_cost_function(self, cf_paras):
		"""
		This is the cost function archetype that is used to build the 
		true cost function internally, for instance to compute the
		costVect and delayCostVect.

		Note that all models should take costVect and delayCostVect or other
		types of parameters, but all the cost function building should happen here.
		"""
		if type(cf_paras)==str:
			try:
				cf_paras = archetypes_cost_functions[cf_paras]()
			except KeyError:
				raise Exception("Unknown function archetype:", cf_paras)

		self.cf_paras = cf_paras

	def get_requirements_for_user(self):
		"""
		Useless
		"""
		reqs_engine = self.get_requirements_from_engine()
		reqs = reqs_engine
		if 'delayCostVect' in reqs or 'costVect' in reqs:
			reqs += ['cost_lambda', ['cost_archetype', 'cost_params']]

	def get_internal_flight(self, flight_name):
		return self.flights[flight_name]

	def get_allocation(self, name_slot='slot'):
		return allocation_from_flights(self.get_flight_list(), name_slot=name_slot)

	def get_cost_vectors(self):
		return {flight.name:{'costVect':flight.costVect, 'delayCostVect':flight.delayCostVect} for flight in self.get_flight_list()}

	def compute_slots(self, slot_times=[]):
		slots = [Slot(i, time) for i, time in enumerate(slot_times)]

		self.set_slots(slots)

		return slots

	def set_slots(self, slots):
		self.slots = slots

	def get_flights(self):
		return self.flights

	def get_flight_list(self):
		return list(self.flights.values())

	def prepare_all_flights(self):
		reqs = self.get_requirements_from_engine()
		#print ('Reqs:', reqs)
		for flight in self.flights.values():
			if 'delayCostVect' in reqs or 'costVect' in reqs:
				flight.compute_cost_vectors(self.slots)

	def update_flight_attributes_dict_to_ext(self, attr_dict):
		for flight, d in attr_dict.items():
			flight_ext = self.dic_objs[flight]

			for k, v in d.items():
				setattr(flight_ext, k, v)

	def update_flight_attributes_int_to_ext(self, attr_map={}):
		"""
		attr_map is in the int -> ext direction
		"""
		for flight_int in self.flights.values():
			flight_ext = self.dic_objs[flight_int]

			for k, v in attr_map.items():
				setattr(flight_ext, v, getattr(flight_int, k))

	def update_flight_attributes_ext_to_int(self, attr_map={}):
		"""
		attr_map is in the ext -> int direction
		"""
		for flight_int in self.flights.values():
			flight_ext = self.dic_objs[flight_int]

			for k, v in attr_map.items():
				setattr(flight_int, v, getattr(flight_ext, k))

	def update_flight_attributes_int_from_dict(self, attr_list={},
		set_cost_function_with=None, attr_map=None):
		"""
		attr_map is in the ext -> int direction
		"""
		for flight_name, attrs in attr_list.items():
			flight_int = self.flights[flight_name]
			for k, v in attrs.items():
				setattr(flight_int, k, v)

			self.set_cost_function(flight_int, attrs, set_cost_function_with)

	def set_cost_function(self, flight, attr_list, set_cost_function_with):
		if not set_cost_function_with is None:
			if set_cost_function_with=='default_cf_paras':
				dd = {'kind':'paras',
						'cost_function':self.cf_paras}
			else:
				dd = {}
				for k, v in set_cost_function_with.items():
					if k=='cost_function':
						if type(v) is str and v in attr_list.keys():
							# Cost function is a lambda function passed in the attr_list
							dd[k] = attr_list[v]
						else:
							# Cost function is passed as a lambda function in set_cost_function_with
							try:
								_ = iter(v)
							except TypeError:
								# not iterable
								# cost function is the same for everyone
								dd[k] = v
							else:
								# iterable
								# cost functions are different for each flight
								dd[k] = v[i]
					else:
						if type(v) is str and v in attr_list.keys():
							dd[k] = attr_list[v]
						else:
							dd[k] = v
			
			if 'cost_function' in dd.keys():
				flight.set_cost_function(**dd)

	def prepare_hotspot_from_dataframe(self, df=None, slot_times=[], attr_map={'flight_name':'flight_name',
		'flight_name':'airline_name', 'eta':'eta'}, set_cost_function_with={}, assign_FPFS=True):
		"""
		attr_map is in the ext -> int direction
		"""

		self.slots = [Slot(i, time) for i, time in enumerate(slot_times)]
		self.flights = {}
		for i, row in df.iterrows():
			row2 = row[list(attr_map.keys())].rename(attr_map)
			flight = Flight(**row2)

			if len(set_cost_function_with)>0:
				# build lambda cost function
				if 'args' in set_cost_function_with.keys():
					args = [row[v] for v in set_cost_function_with['args']]
				else:
					args = []
				if 'kwargs' in set_cost_function_with.keys():
					kwargs = [row[v] for v in set_cost_function_with['kwargs']]
				else:
					kwargs = []
				kwargs = {v:row[v] for v in set_cost_function_with['kwargs']}
				
				def f(x):
					return set_cost_function_with['cost_function'](x, *args, **kwargs)

				if 'eta' in set_cost_function_with.keys():
					eta = row[set_cost_function_with['eta']]
				else:
					eta = None

				flight.set_cost_function(f,
										kind='lambda',
										absolute=set_cost_function_with['absolute'],
										eta=eta)
				flight.compute_cost_vectors(self.slots)
			
			self.flights[flight.name] = flight
		
		if assign_FPFS:
			assign_FPFS_slot(self.slots, self.flights)

		return self.slots, self.flights

	def prepare_hotspot_from_flights_ext(self, flights_ext=None, slot_times=[], slots=[], attr_map={'flight_name':'flight_name',
		'flight_name':'airline_name', 'eta':'eta'},
		set_cost_function_with={}, assign_FPFS=True):
		"""
		attr_map is in the ext -> int direction
		"""

		if 'slot' in attr_map.keys():
			assign_FPFS = False

		self.flights_ext = flights_ext
		self.slot_times = slot_times
		self.attr_map = attr_map
		self.set_cost_function_with = set_cost_function_with

		if len(slots)>1:
			self.set_slots(slots)
		else:
			self.compute_slots(slot_times=slot_times)

		self.flights = {}
		self.dic_objs = {}
		for i, flight_ext in enumerate(self.flights_ext):
			d = {v:getattr(flight_ext, k) for k, v in self.attr_map.items()}

			flight = Flight(**d)

			self.dic_objs[flight] = flight_ext
			
			if len(self.set_cost_function_with)>0:
				dd = {}
				for k, v in self.set_cost_function_with.items():
					if k=='cost_function':
						if type(v) is str and hasattr(flight_ext, v):
							dd[k] = getattr(flight_ext, v)
						else:
							try:
								_ = iter(v)
							except TypeError:
								# not iterable
								# cost function is the same for everyone
								dd[k] = v
							else:
								# iterable
								# cost functions are different for each flight
								dd[k] = v[i]
					else:
						if type(v) is str and hasattr(flight_ext, v):
							dd[k] = getattr(flight_ext, v)
						else:
							dd[k] = v
							
				flight.set_cost_function(**dd)

			flight.compute_cost_vectors(self.slots)
			
			self.flights[flight.name] = flight

		if assign_FPFS:
			assign_FPFS_slot(self.slots, self.flights)

		return self.slots, self.flights

	def prepare_flights_from_dict(self, attr_list=[], set_cost_function_with='default_cf_paras', attr_map=None):
		"""
		Specificy set_cost_function_with only if you want an actual computation of CostVect,
		for instance if you are not passing references for ISTOP/UDPP
		"""
		if attr_map is None:
			attr_map = {k:k for k in attr_list.keys()}

		self.flights = OrderedDict()
		for i, attr_list_f in enumerate(attr_list):
			d = {v:attr_list_f[k] for k, v in attr_map.items()}
			flight = Flight(**d)
			
			self.set_cost_function(flight, attr_list_f, set_cost_function_with)

			self.flights[flight.name] = flight

	def prepare_hotspot_from_dict(self, attr_list=None, slot_times=[], slots=[], attr_map=None,
		set_cost_function_with={}, assign_FPFS=True, delta_t=None):

		# {'flight_name':'flight_name',
		# 'airline_name':'airline_name', 'eta':'eta'}

		if attr_map is None:
			attr_map = {k:k for k in attr_list[0].keys()}

		if 'slot' in attr_map.keys():
			assign_FPFS = False

		self.attr_list = attr_list
		self.slot_times = slot_times
		self.attr_map = attr_map
		self.set_cost_function_with = set_cost_function_with

		if len(slots)>0:
			self.set_slots(slots)
		else:
			self.compute_slots(slot_times=slot_times)

		self.prepare_flights_from_dict(attr_list=attr_list,
										set_cost_function_with=set_cost_function_with,
										attr_map=attr_map)

		# Check that the first flight has an ETA corresponding to 
		# the first slot, otherwise change its ETA internally.
		first_flight = self.get_flight_list()[np.argmin([flight.eta for flight in self.get_flight_list()])]		 
		if first_flight.eta > self.slots[0].time:
			first_flight.eta = self.slots[0].time

		if assign_FPFS:
			if delta_t is None:
				if self.alternative_slot_allocation_rule and len(self.slots)>1:
					delta_t = compute_delta_t_from_slots(self.slots)
				else:
					delta_t = 0.
			assign_FPFS_slot(self.slots, self.flights, delta_t=delta_t)

			# for flight in self.get_flight_list():
			# 	print ('FPFS:', flight.name, flight.slot)

		return self.slots, self.get_flight_list()

	def assign_slots_to_flights_ext_from_allocation(self, allocation, attr_slot_ext='slot'):
		for flight, slot in allocation.items():
			flight_ext = self.dic_objs[flight]

			setattr(flight_ext, attr_slot_ext, slot)

	def update_slots_internal(self):
		[flight.update_slot() for flight in self.flights]

	def get_new_flight_list(self):
		"""
		Creates a new list of flight objects with newSlot as slot,
		except if the former is None, in which case the new objects
		have the same slot attribute than the old one.
		"""
		new_flight_list = []
		for flight in self.flights.values():
			new_flight = HFlight(**flight.get_attributes())

			if not flight.newSlot is None:
				new_flight.slot = flight.newSlot
				new_flight.newSlot = None
			new_flight_list.append(new_flight)

		return sorted(new_flight_list, key=lambda f: f.slot)

	def print_summary(self):
		print ('Slots:', self.slots)
		print ('Flights/ETA:', [(flight.name, flight.eta) for flight in self.get_flight_list()])

class RLFlight(HFlight):
	pass


class Flight(HFlight):
	# def __init__(self, hflight=None, cost_function_lambda=None, cost_function_paras=None,
	# 	**kwargs):

	# 	"""
	# 	You can just extend ah HFlight instance by passing it in argument. Note that in this case,
	# 	all other arguments are ignored, except cost_function which is mandatory.
	# 	"""
	# 	kwargs_copy = kwargs.copy()
		# # for k, v in kwargs_copy.items():
		# # 	if k=='margin':
		# # 		kwargs['margin_1'] = v
		# # 		del kwargs[k]
		# # 	elif k=='jump':
		# # 		kwargs['jump_1'] = v
		# # 		del kwargs[k]

		# if hflight is None:
		# 	kwargs2 = {k:v for k, v in kwargs.items() if not k in ['slot', 'flight_name', 'airline_name', 'eta']}
		# 	super().__init__(slot=kwargs.get('slot', None),
		# 					flight_name=kwargs.get('flight_name', None),
		# 					airline_name=kwargs.get('airline_name', None),
		# 					eta=kwargs.get('eta', None),
		# 					delay_cost_vect=None,
		# 					**kwargs2)

		# else:
		# 	# When you pass an hflight instance, 
		# 	# get all the attributes
		# 	self.__dict__.update(hflight.__dict__)

		# Following is useful because of clipping
		# self.set_true_charac(margin=self.margin1,
		# 					slope=self.slope,
		# 					jump=self.jump1)
		
		# self.set_declared_charac(margin=self.margin1,
		# 						slope=self.slope,
		# 						jump=self.jump1)
		
		# if not cost_function_paras is None:
		# 	self.set_cost_function_from_cf_paras(cost_function_paras)
		# else:
		# 	if not cost_function_lambda is None:
		# 		self.set_cost_function_from_lambda_cf(cost_function_lambda)

	# def set_true_charac(self, margin=None, slope=None, jump=None):
	# 	if not margin is None:
	# 		self.margin1 = max(0., margin)
	# 	if not slope is None:
	# 		self.slope = max(0., slope)
	# 	if not slope is jump:
	# 		self.jump1 = max(0., jump)

	# def set_declared_charac(self, margin=None, slope=None, jump=None):
	# 	if not margin is None:
	# 		self.margin1_declared = max(0., margin)
	# 	if not slope is None:
	# 		self.slope_declared = max(0., slope)
	# 	if not slope is jump:
	# 		self.jump1_declared = max(0., jump)

	def set_cost_function(self, cost_function, kind='lambda', **kwargs):
		if kind=='lambda':
			self.set_cost_function_from_lambda(cost_function, **kwargs)
		elif kind=='paras':
			self.set_cost_function_from_paras(cost_function)

	def compute_lambda_from_paras(self, cost_function_paras):
		"""
		cost_function_paras takes a flight and slot as input.
		Not that final cost function argument is absolute time, 
		not delay
		"""
		class DummyFlight:
			pass
		
		class DummySlot:
			pass
		
		# Real cost function
		def f(time):
			slot = DummySlot()
			slot.time = time
			return cost_function_paras(time, **{attr:getattr(self, attr) for attr in cost_function_paras.paras})
		
		#self.cost_f_true = f 
		
		# def ff(time):
		# 	# Declared cost function
		# 	declared_flight = DummyFlight()
		# 	declared_flight.eta = self.eta
		# 	declared_flight.margin1 = getattr(self, 'margin1_declared', self.margin1)
		# 	declared_flight.slope = getattr(self, 'slope_declared', self.slope)
		# 	declared_flight.jump1 = getattr(self, 'jump1_declared', self.jump1)
			
		# 	slot = DummySlot()
		# 	slot.time = time
		# 	return cost_function_paras(declared_flight, slot)
		
		#self.cost_f_declared = ff

		return f#, ff

	def set_cost_function_from_paras(self, cost_function_paras):
		f = self.compute_lambda_from_paras(cost_function_paras)
		self.set_cost_function_from_lambda(f,
											#cost_function_declared=fd,
											absolute=True)

	def set_cost_function_from_lambda(self, cost_function, #cost_function_declared=None,
		absolute=True, eta=None):
		"""
		Useful to set a cost function which has the signature delay -> cost or
		absolute time -> cost. In the former case, one should select absolute=False
		and provide an eta from the flight to shift the cost function.
		"""
		if absolute:
			self.cost_f_true = cost_function
			#if not cost_function_declared is None:
			#	self.cost_f_declared = cost_function_declared
			#else:
			#	self.cost_f_declared = cost_function
		else:
			def f(x):
				if x-eta<0.:
					return 0.
				else:
					return cost_function(x-eta)

			self.cost_f_true = f

			# if not cost_function_declared is None:
			# 	def ff(x):
			# 		if x-eta<0.:
			# 			return 0.
			# 		else:
			# 			return cost_function_declared(x-eta)

			# 	self.cost_f_declared = ff
			# else:
			# 	self.cost_f_declared = f

	def compute_cost_vectors(self, slots):
		"""
		Compute costVect and delayCostVect, required for all models. If one or the 
		other is given, the other is computed from it. If none are given, costVect
		is computed from the cost function, and delayCostVect from costVect.
		"""
		if self.costVect is None:
			if self.delayCostVect is None:
				self.compute_cost_vect_from_cost_function(slots)
			else:
				self.compute_cost_vect_from_delay_cost_vect(slots)

		if self.delayCostVect is None:
			# This is computed always from costVect.
			self.compute_delay_cost_vect(slots)

	def compute_cost_vect_from_delay_cost_vect(self, slots):
		if self.costVect is None:
			i = 0
			self.costVect = []
			for slot in slots:
				if slot.time < self.eta:
					self.costVect.append(0)
				else:
					self.costVect.append(self.delayCostVect[i])
					i += 1

			self.costVect = np.array(self.costVect)

	def compute_cost_vect_from_cost_function(self, slots):#, declared=True):
		"""
		This method allows to compute the vector costVect and delayCostVect, 
		which are required by different optimiser.
		"""
		self.costVect = np.array([self.cost_f_true(slot.time) for slot in slots])

	def compute_delay_cost_vect(self, slots):
		"""
		This is used when costVect is given instead of delayCostVect, but
		the latter is still required, for instance for ISTOP.
		"""
		if self.delayCostVect is None:
			self.delayCostVect = []
			i = 0
			for slot in slots:
				if slot.time >= self.eta:
					self.delayCostVect.append(self.costVect[i])
				i += 1

			self.delayCostVect = np.array(self.delayCostVect)


class Engine:
	def __init__(self, algo):
		self.algo = algo

		if not algo in models.keys():
			raise Exception('Unknown algorithm:', algo, '. Available algorithms:', list(models.keys()))

		# if self.algo=='nnbound':
		# 	self.model = models[self.algo]()

	def get_requirements(self):
		"""
		List the required attributes that should be attached to the flights passed 
		to the engine itself.
		"""

		return models[self.algo].requirements

	def compute_optimal_allocation(self, hotspot_handler=None, slots=None, flights=None,
		use_priorities=False, kwargs_init={}, kwargs_run={}):
		"""
		flights is a dict {name:Flight}
		"""

		Model = models[self.algo]
		
		if not hotspot_handler is None:
			slots = hotspot_handler.slots
			flights = hotspot_handler.get_flight_list()

			if (not 'cost_func_archetype' in kwargs_init.keys()) \
				and (hasattr(hotspot_handler, 'cf_paras')) \
				and ('cost_func_archetype' in Model.get_kwargs_init(Model)):
				kwargs_init['cost_func_archetype'] = hotspot_handler.cf_paras

			if (not 'delta_t' in kwargs_init.keys()) \
				and hotspot_handler.alternative_slot_allocation_rule \
				and len(hotspot_handler.slots)>1:
				# In this case, slots should be allocated to if t1 < eta < t2,
				# where t1 and t2 are the slot boundary.
				# We compute the delta_t based on the minimum difference between slots
				delta_t = compute_delta_t_from_slots(hotspot_handler.slots)
				kwargs_init['delta_t'] = delta_t

		if use_priorities:
			try:
				assert hasattr(flights[0], 'udppPriority')
			except AssertionError:
				raise Exception("Flights need to have a udppPriority attribute when merging priorities.")
			try:
				assert self.algo == 'udpp_merge'
			except AssertionError:
				raise Exception("You asked to use priorities for optimal allocation, this is only possible when selecting 'uddp_merge' as optimiser.")
			kwargs_run['optimised'] = False

		# if self.algo=='nnbound':
		# 	self.model.reset(slots, flights, **kwargs_init)
		# else:
		self.model = Model(slots, flights, **kwargs_init)
		self.model.run(**kwargs_run)
		
		self.allocation = allocation_from_flights(flights, name_slot='newSlot')

		del kwargs_init
		
		return self.allocation

	def print_optimisation_performance(self):
		self.model.print_performance()


class LocalEngine(Engine):
	def compute_optimal_parameters(self, hotspot_handler=None, slots=None, flights=None,
		kwargs_init={}, kwargs_run={}):
		"""
		Merge with previous?
		"""
		Model = models[self.algo]

		if not hotspot_handler is None:
			slots = hotspot_handler.slots
			flights = hotspot_handler.get_flight_list()
			#and ('cost_func_archetype' in inspect.signature(models[self.algo].__init__).parameters.keys()):
			
			if (not 'cost_func_archetype' in kwargs_init.keys()) \
				and (hasattr(hotspot_handler, 'cf_paras')) \
				and ('cost_func_archetype' in Model.get_kwargs_init(Model)):
				kwargs_init['cost_func_archetype'] = hotspot_handler.cf_paras

			if (not 'delta_t' in kwargs_init.keys()) \
				and hotspot_handler.alternative_slot_allocation_rule \
				and len(hotspot_handler.slots)>1 \
				and ('delta_t' in Model.get_kwargs_init(Model)):
				# In this case, slots should be allocated to if t1 < eta < t2,
				# where t1 and t2 are the slot boundary.
				# We compute the delta_t based on the minimum difference between slots
				delta_t = compute_delta_t_from_slots(hotspot_handler.slots)
				kwargs_init['delta_t'] = delta_t

		self.model = Model(slots=slots, 
							flights=flights, 
							**kwargs_init)

		paras = self.model.run(**kwargs_run)

		return paras


class GetCostVectors:
	"""
	Dummy model for getting cost vectors of all flights in the
	same format than the other local models.
	"""

	requirements = ['costVect', 'delayCostVect']

	def __init__(self, slots, flights):
		self.slots = slots
		self.flights = flights

	def run(self):
		paras = OrderedDict((flight.name, {'costVect':flight.costVect,
											'delayCostVect':flight.delayCostVect}) for flight in self.flights)
		return paras


models['get_cost_vectors'] = GetCostVectors
