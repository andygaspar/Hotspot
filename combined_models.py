"""
Combined models are collections of models which are executed one after the 
other. Usual comined models corresponds to a local model + global one, for
instance UDPPLocal and UDPPMerge, in order to allow for a easy end-to-end 
computation, also using the same api than other models. Some operations 
sometimes need to be performed between models, like recomputing cost vectors.
"""
import inspect

from typing import Callable, Union, List

from .ModelStructure.Flight.flight import Flight
from .ModelStructure.Slot.slot import Slot

from .ModelStructure import modelStructure as mS
from .UDPP.udppMerge import UDPPMerge
from .UDPP.udppLocal import UDPPLocal
from .UDPP.functionApprox import FunctionApprox
from .Istop.istop import Istop
from .GlobalOptimum.globalOptimum import GlobalOptimum
from .NNBound.nnBound import NNBoundModel


def init_and_run(Model, slots, flights, kwargs_init, kwargs_run):
	# Get all kwargs for init and init
	all_vars = inspect.signature(Model.__init__).parameters
	kwargs_init_res = {k:kwargs_init.get(k, v.default) for k, v in all_vars.items() if not k in ['self', 'slots', 'flights']}
	model = Model(slots, flights, **kwargs_init_res)

	# Get all kwargs for run and run
	all_vars = inspect.signature(Model.run).parameters
	kwargs_run_res = {k:kwargs_run.get(k, v.default) for k, v in all_vars.items() if not k in ['self', 'slots', 'flights']}
	results = model.run(**kwargs_run_res)

	return model, results

def combine_model(Models, assign_slots_after_models=False, sequential_requirements=True):
	"""
	Creates a model that combines sequentially the models in Models.
	"""

	class CombinedModel(mS.ModelStructure):
		if not sequential_requirements:
			requirements = list(set([req for Model in Models for req in Model.requirements]))
		else:
			requirements = Models[0].requirements

		@staticmethod
		def get_kwargs_init(dummy):
			"""
			Gets all kwargs from models' init functions to avoid inspection issues
			with kwargs_init.
			"""
			all_vars = list(set([k for Model in Models for k in inspect.signature(Model.__init__).parameters.keys() if not k in ['self', 'slots', 'flights']]))

			return all_vars

		@staticmethod
		def get_kwargs_run(dummy):
			all_vars = list(set([k for Model in Models for k in inspect.signature(Model.run).parameters.keys() if not k in ['self', 'slots', 'flights']]))

			return all_vars
			
		def __init__(self, slots: List[Slot]=None, flights: List[Flight]=None, checks=True,
			**kwargs_init):
			self.slots = slots
			self.flights = flights
			self.kwargs_init = kwargs_init
			
			# Check that all flights have the attributes required by the model(s)
			if checks:
				self.check_requirements()

		def run(self, **kwargs_run):
			merge_results = {f.name:{} for f in self.flights}

			for i, Model in enumerate(Models):
				# Cost vectors will be recomputed if they are required during the
				# next step AND a cost archetype function is given in input.
				# This is typically the case for FuncApprox + Istop, NN bound etc.
				if i>0 and ('delayCostVect' in Model.requirements or 'costVect' in Model.requirements):
					for flight in self.flights:
						if 'cost_func_archetype' in self.kwargs_init.keys() and self.kwargs_init['cost_func_archetype'] is not None:
							flight.set_cost_function(kind='paras',
													cost_function=self.kwargs_init['cost_func_archetype'])
						flight.compute_cost_vectors(self.slots)

				model, results = init_and_run(Model, self.slots, self.flights, self.kwargs_init, kwargs_run)
				
				# One can reassign slots between models. This is useful for isntance
				# if using to global models one after the other, like UDPP merge and Istop.
				if type(assign_slots_after_models) in [tuple, list]:
					assign = assign_slots_after_models[i]
				else:
					assign = assign_slots_after_models

				if assign:
					for flight in self.flights:
						flight.slot = flight.newSlot
						flight.newSlot = None

				for f in self.flights:
					if results is not None:
						for k, v in results[f.name].items():
							merge_results[f.name][k] = v
		
			# Assign final solution and report
			self.solution = model.solution
			self.report = model.report

			return merge_results

	return CombinedModel

# UDPP from scratch
UDPPTotal = combine_model([UDPPLocal, UDPPMerge])

# Istop from scratch with approximation (otherwise use only istop)
IstopTotalApprox = combine_model([FunctionApprox, Istop])

# NNbound from scratch with approximation
NNBoundTotalApprox = combine_model([FunctionApprox, NNBoundModel])

# GlobalOptimum from scratch with approximation
GlobalOptimumTotalApprox = combine_model([FunctionApprox, GlobalOptimum])

# UDPP merge first, Istop second
UDPPMergeIstop = combine_model([UDPPMerge, Istop], assign_slots_after_models=[True, False], sequential_requirements=False)

# Computes preferences AND function approximation (for UDPPMergeIstop for instance)
UDPPLocalFunctionApprox = combine_model([UDPPLocal, FunctionApprox], sequential_requirements=False)
