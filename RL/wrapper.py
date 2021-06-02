from collections import OrderedDict
import inspect

import numpy as np
import pandas as pd

from Hotspot.ModelStructure.Slot.slot import Slot
from Hotspot.ModelStructure.Flight.flight import Flight as HFlight
from Hotspot.Istop.istop import Istop
from Hotspot.NNBound.nnBound import NNBoundModel
from Hotspot.UDPP.udppModel import UDPPmodel
from Hotspot.GlobalOptimum.globalOptimum import GlobalOptimum
from Hotspot.ModelStructure.Costs.costFunctionDict import CostFuns
from Hotspot.libs.uow_tool_belt.general_tools import write_on_file as print_to_void, clock_time

models = {'istop':Istop,
		'nnbound':NNBoundModel,
		'udpp':UDPPmodel,
		'globaloptimum':GlobalOptimum}

def allocation_from_df(df, name_slot='new slot'):
	return OrderedDict(df[['flight', name_slot]].set_index('flight').to_dict()[name_slot])

def allocation_from_flights(flights, name_slot='newSlot'):
	return OrderedDict([(flight.name, getattr(flight, name_slot).index) for flight in flights])

def priorities_from_flights(flights):
	d = OrderedDict([(flight.name, {}) for flight in flights])

	for flight in flights:
		d[flight.name]['udppPriority'] = flight.udppPriority
		d[flight.name]['udppPriorityNumber'] = flight.udppPriorityNumber
		d[flight.name]['tna'] = flight.tna

	return d

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

def assign_FPFS_slot(slots, flights):
	flights_ordered = sorted(flights, key=lambda x:x.eta)
	slots_ordered = sorted(slots, key=lambda x:x.time)

	for i, flight in enumerate(flights_ordered):
		flight.slot = slots_ordered[i]


class FlightHandler:
	"""
	In the following, "ext" corresponds to external flight objects,
	created outside of the Hotspot scope. "int" refers to internal
	flight objects.
	"""
	def __init__(self):
		pass

	def compute_slots(self, slot_times=[]):
		slots = [Slot(i, time) for i, time in enumerate(self.slot_times)]

		self.assign_slots(slots)

	def assign_slots(self, slots):
		self.slots = slots

	def get_flights(self):
		return self.flights

	def update_flight_attributes_int_to_ext(self, attr_map={}):
		"""
		attr_map is in the int -> ext direction
		"""
		for flight_int in self.flights:
			flight_ext = self.dic_objs[flight_int]

			for k, v in attr_map.items():
				setattr(flight_ext, v, getattr(flight_int, k))

	def update_flight_attributes_ext_to_int(self, attr_map={}):
		"""
		attr_map is in the ext -> int direction
		"""
		for flight_int in self.flights:
			flight_ext = self.dic_objs[flight_int]

			for k, v in attr_map.items():
				setattr(flight_int, v, getattr(flight_ext, k))

	def prepare_hotspot_from_dataframe(self, df=None, slot_times=[], attr_map={},
		set_cost_function_with={}):
		"""
		attr_map is in the ext -> int direction
		"""

		self.slots = [Slot(i, time) for i, time in enumerate(slot_times)]
		self.flights = []
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
				flight.compute_cost_vect(self.slots)
			
			self.flights.append(flight)

		assign_FPFS_slot(self.slots, self.flights)

		return self.slots, self.flights

	def prepare_hotspot_from_objects(self, flights_ext=None, slot_times=[], slots=[], attr_map={},
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
			self.assign_slots(slots)
		else:
			self.compute_slots(slot_times=slot_times)

		self.flights = []
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

			flight.compute_cost_vect(self.slots)
			
			self.flights.append(flight)
		if assign_FPFS:
			assign_FPFS_slot(self.slots, self.flights)

		return self.slots, self.flights

	def assign_priorities_to_objects(self, priorities):
		for flight, d in priorities.items():
			flight_ext = self.dic_objs[flight]

			for k, v in d.items():
				setattr(flight_ext, k, v)

	def assign_slots_to_objects_from_allocation(self, allocation, attr_slot_ext='slot'):
		for flight, slot in allocation.items():
			flight_ext = self.dic_objs[flight]

			setattr(flight_ext, attr_slot_ext, slot)
			
	def assign_slots_to_objects_from_internal_flights(self, attr_slot_ext='slot', attr_slot_int='newSlot'):
		self.update_flight_attributes_int_to_ext({attr_slot_ext:attr_slot_int})

	def update_slots_internal(self):
		[flight.update_slot() for flight in self.flights]

	def get_new_flight_list(self):
		"""
		Creates a new list of flight objects with newSlot as slot,
		except if the former is None, in which case the new objects
		have the same slot attribute than the old one.
		"""
		new_flight_list = []
		for flight in self.flights:
			new_flight = HFlight(**flight.get_attributes())
			if not flight.newSlot is None:

				new_flight.slot = flight.newSlot
				new_flight.newSlot = None
			new_flight_list.append(new_flight)

		return sorted(new_flight_list, key=lambda f: f.slot)


class Flight(HFlight):
	def __init__(self, hflight=None, cost_function_lambda=None, cost_function_paras=None,
		**kwargs):

		"""
		You can just extend ah HFlight instance by passing it in argument. Note that in this case,
		all other arguments are ignored, except cost_function which is mandatory.
		"""
		kwargs_copy = kwargs.copy()
		for k, v in kwargs_copy.items():
			if k=='margin':
				kwargs['margin_1'] = v
				del kwargs[k]
			elif k=='jump':
				kwargs['jump_1'] = v
				del kwargs[k]

		if hflight is None:
			kwargs2 = {k:v for k, v in kwargs.items() if not k in ['slot', 'flight_name', 'airline_name', 'eta']}
			super().__init__(slot=kwargs.get('slot', None),
							flight_name=kwargs.get('flight_name', None),
							airline_name=kwargs.get('airline_name', None),
							eta=kwargs.get('eta', None),
							delay_cost_vect=None,
							**kwargs2)

		else:
			# When you pass an hflight instance, 
			# get all the attributes
			self.__dict__.update(hflight.__dict__)

		# Following is useful because of clipping
		self.set_true_charac(margin=self.margin1,
							slope=self.slope,
							jump=self.jump1)
		
		self.set_declared_charac(margin=self.margin1,
								slope=self.slope,
								jump=self.jump1)
		
		if not cost_function_paras is None:
			self.set_cost_function_from_cf_paras(cost_function_paras)
		else:
			if not cost_function_lambda is None:
				self.set_cost_function_from_lambda_cf(cost_function_lambda)

	def set_true_charac(self, margin=None, slope=None, jump=None):
		if not margin is None:
			self.margin1 = max(0., margin)
		if not slope is None:
			self.slope = max(0., slope)
		if not slope is jump:
			self.jump1 = max(0., jump)

	def set_declared_charac(self, margin=None, slope=None, jump=None):
		if not margin is None:
			self.margin1_declared = max(0., margin)
		if not slope is None:
			self.slope_declared = max(0., slope)
		if not slope is jump:
			self.jump1_declared = max(0., jump)

	def set_cost_function(self, cost_function, kind='lambda', **kwargs):
		if kind=='lambda':
			self.set_cost_function_from_lambda_cf(cost_function, **kwargs)
		elif kind=='paras':
			self.set_cost_function_from_cf_paras(cost_function)

	def compute_lambda_cf_from_cf_paras(self, cost_function_paras):
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
			return cost_function_paras(self, slot)
		
		#self.cost_f_true = f 
		
		def ff(time):
			# Declared cost function
			declared_flight = DummyFlight()
			declared_flight.eta = self.eta
			declared_flight.margin1 = self.margin1_declared
			declared_flight.slope = self.slope_declared
			declared_flight.jump1 = self.jump1_declared
			
			slot = DummySlot()
			slot.time = time
			return cost_function_paras(declared_flight, slot)
		
		#self.cost_f_declared = ff

		return f, ff

	def set_cost_function_from_cf_paras(self, cost_function_paras):
		f, fd = self.compute_lambda_cf_from_cf_paras(cost_function_paras)
		self.set_cost_function_from_lambda_cf(f, cost_function_declared=fd, absolute=True)

	def set_cost_function_from_lambda_cf(self, cost_function, cost_function_declared=None,
		absolute=True, eta=None):
		"""
		Useful to set a cost function which has the signature delay -> cost or
		absolute time -> cost. In the former case, one should select absolute=False
		and provide an eta from the flight to shift the cost function.
		"""
		if absolute:
			self.cost_f_true = cost_function
			if not cost_function_declared is None:
				self.cost_f_declared = cost_function_declared
			else:
				self.cost_f_declared = cost_function
		else:
			def f(x):
				if x-eta<0.:
					return 0.
				else:
					return cost_function(x-eta)

			self.cost_f_true = f

			if not cost_function_declared is None:
				def ff(x):
					if x-eta<0.:
						return 0.
					else:
						return cost_function_declared(x-eta)

				self.cost_f_declared = ff
			else:
				self.cost_f_declared = f

	def compute_cost_vect(self, slots, declared=True):
		"""
		In theory, this attribute is only used by the UDPP optimiser,
		and thus it should always be computed using the declared cost function.
		"""
		if declared:
			self.costVect = np.array([self.cost_f_declared(slot.time) for slot in slots])
		else:
			self.costVect = np.array([self.cost_f_true(slot.time) for slot in slots])


class OptimalAllocationComputer:
	def __init__(self, algo='istop'):#, **kwargs):
		self.algo = algo
		if self.algo=='nnbound':
			self.model = models[self.algo]()

	def compute_local_priorities(self, slots, flights, kwargs_init={}):
		try:
			assert self.algo=='udpp'
		except AssertionError:
			raise Exception('You need to select "udpp" as the optimiser for copmutation of priorities.')

		self.model = models[self.algo](slots, flights, **kwargs_init)

		self.model.compute_optimal_prioritisation()

		self.priorities = priorities_from_flights(flights)

		return self.priorities

	def compute_optimal_allocation(self, slots, flights, use_priorities=False, kwargs_init={}, kwargs_run={}):
		"""
		Flights is a dict {name:Flight}
		"""
		#flights = list(flights.values())
		# TODO: automatically detect arguments going in init and arguments going in run
		if use_priorities:
			try:
				assert hasattr(flights[0], 'udppPriority')
			except AssertionError:
				raise Exception("Flights need to have a udppPriority attribute when merging priorities.")
			try:
				assert self.algo == 'udpp'
			except AssertionError:
				raise Exception("You asked to use priorities for optimal allocation, this is only possible when selecting 'uddp' as optimiser.")
			kwargs_run['optimised'] = False

		if self.algo=='nnbound':
			self.model.reset(slots, flights, **kwargs_init)
		else:
			self.model = models[self.algo](slots, flights, **kwargs_init)
		self.model.run(**kwargs_run)
		self.allocation = allocation_from_flights(flights, name_slot='newSlot')

		return self.allocation

	def print_optimisation_performance(self):
		self.model.print_performance()

