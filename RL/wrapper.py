from collections import OrderedDict

import numpy as np
import pandas as pd

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

class Flight(HFlight):
	def __init__(self, hflight=None, eta=None, name=None, margin=None, slope=None, cost_function_paras=None,
		jump=None, cost_function_lambda=None):

		"""
		You can just extend ah HFlight instance by passing it in argument. Note that in this case,
		all other arguments are ignored, except cost_function which is mandatory.
		"""
		if hflight is None:
			super().__init__(slot=None,
							flight_name=name,
							airline_name='',
							eta=eta,
							delay_cost_vect=None,
							udpp_priority=None,
							udpp_priority_number=None,
							tna=None,
							slope=slope,
							margin_1=margin,
							jump_1=jump,
							margin_2=None,
							jump_2=None)

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
		self.margin1 = max(0., margin)
		self.slope = max(0., slope)
		self.jump1 = max(0., jump)

	def set_declared_charac(self, margin=None, slope=None, jump=None):
		self.margin1_declared = max(0., margin)
		self.slope_declared = max(0., slope)
		self.jump1_declared = max(0., jump)

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
	def __init__(self, trading_alg='istop'):

		self.trading_alg = trading_alg
		
		self.model = models[self.trading_alg]()

	def compute_optimal_allocation(self, slots, flights):
		"""
		Flights is a dict {name:Flight}
		"""
		flights = list(flights.values())
		self.model.reset(slots, flights)
		# with clock_time(message_after='model run executed in'):
		self.model.run()
		allocation = allocation_from_flights(flights, name_slot='newSlot')

		return allocation
