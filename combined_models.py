"""
This is a meta model using Istopand UDPP under the hood. There are two flavours:
- For the first one, the input to the model is UDPP priorities + cost vectors.
The model then run UDPP with the preferences, and ISTOP with the cost vectors
- for the second one, the input is only cost vectors. In this case, the model
computes preferenecs internally based on the cost vectors, apply them, 
and then apply ISTOP with the cost vectors.
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


def combine_model(models):
	#TODO
	pass

class UDPPMergeIstop(mS.ModelStructure):
	"""
	Version in which you pass the priorities from the UDPP Local
	for the UDPP merge and the cost vectors for ISTOP. To be used with the 
	FuncApprox (or direclty the true costs) and UDPPLocal optimiser.
	"""
	requirements = UDPPMerge.requirements + Istop.requirements

	def __init__(self, slots: List[Slot]=None, flights: List[Flight]=None, checks=True,
		**kwargs_init):
		self.slots = slots
		self.flights = flights
		self.kwargs_init = kwargs_init

		# Check that all flights have the attributes required by the model
		if checks:
			self.check_requirements()

	def run(self, **kwargs_all):
		# Merge priorities from UDPP local first
		# Split kwargs
		all_vars = inspect.signature(UDPPMerge.__init__).parameters
		kwargs_init = {k:self.kwargs_init.get(k, v.default) for k, v in all_vars.items() if not k in ['self', 'slots', 'flights']}
		udpp_merge_model = UDPPMerge(self.slots, self.flights, **kwargs_init)

		all_vars = inspect.signature(UDPPMerge.run).parameters
		kwargs = {k:kwargs_all.get(k, v.default) for k, v in all_vars.items() if not k in ['self', 'slots', 'flights']}
		udpp_merge_model.run(**kwargs)

		# Allocate new slot to old ones.
		for flight in self.flights:
			flight.slot = flight.newSlot
			flight.newSlot = None

		# Use Istop on new state
		all_vars = inspect.signature(Istop.__init__).parameters
		kwargs_init = {k:self.kwargs_init.get(k, v.default) for k, v in all_vars.items()  if not k in ['self', 'slots', 'flights']}
		istop_model = Istop(self.slots, self.flights, **kwargs_init)

		all_vars = inspect.signature(Istop.run).parameters
		kwargs = {k:kwargs_all.get(k, v.default) for k, v in all_vars.items() if not k in ['self', 'slots', 'flights']}
		istop_model.run(**kwargs)

		self.solution = istop_model.solution
		self.report = istop_model.report


class UDPPFullIstop(mS.ModelStructure):
	"""
	Version in which you pass the cost vectors only. To be used with the 
	FuncApprox local optimiser for instance.
	"""
	requirements = Istop.requirements

	def __init__(self, slots: List[Slot]=None, flights: List[Flight]=None, **kwargs_init):
		self.slots = slots
		self.flights = flights
		self.kwargs_init = kwargs_init

	def run(self, **kwargs_all):
		# TODO: finish that.
		# Compute local UDPP from cost Vect
		# Split kwargs
		all_vars = inspect.signature(UDPPLocal.__init__).parameters
		kwargs_init = {k:self.kwargs_init.get(k, v) for k, v in all_vars.items()}
		udpp_merge_model = UDPPLocal(self.slots, self.flights, **kwargs_init)

		all_vars = inspect.signature(UDPPLocal.run).parameters
		kwargs = {k:kwargs_all.get(k, v) for k, v in all_vars.items()}
		udpp_merge_model.run(**kwargs)

		# Allocate new slot to old ones.
		for flight in self.flights:
			flight.slot = flight.newSlot
			flight.newSlot = None

		# Use Istop on new state
		all_vars = inspect.signature(Istop.__init__).parameters
		kwargs_init = {k:self.kwargs_init.get(k, v) for k, v in all_vars.items()}
		istop_model = Istop(self.slots, self.flights, **kwargs_init)

		all_vars = inspect.signature(Istop.run).parameters
		kwargs = {k:kwargs_all.get(k, v) for k, v in all_vars.items()}
		istop_model.run(**kwargs)


class UDPPLocalFunctionApprox(mS.ModelStructure):
	requirements = list(set(UDPPLocal.requirements + FunctionApprox.requirements))

	def __init__(self, slots: List[Slot]=None, flights: List[Flight]=None, checks=True, 
		cost_func_archetype=None, **kwargs_init):
		self.slots = slots
		self.flights = flights
		self.kwargs_init = kwargs_init
		self.kwargs_init['cost_func_archetype'] = cost_func_archetype

		# Check that all flights have the attributes required by the model
		if checks:
			self.check_requirements()

	def run(self, **kwargs_all):
		# Compute UDPPLocal priorities
		all_vars = inspect.signature(UDPPLocal.__init__).parameters
		kwargs_init = {k:self.kwargs_init.get(k, v.default) for k, v in all_vars.items() if not k in ['self', 'slots', 'flights']}
		#print ('kwargs_init', kwargs_init)
		udpp_local_model = UDPPLocal(self.slots, self.flights, **kwargs_init)

		all_vars = inspect.signature(UDPPLocal.run).parameters
		kwargs = {k:kwargs_all.get(k, v.default) for k, v in all_vars.items() if not k in ['self', 'slots', 'flights']}
		prefs_udpp_local = udpp_local_model.run(**kwargs)

		# Compute function approximation parameters
		all_vars = inspect.signature(FunctionApprox.__init__).parameters
		kwargs_init = {k:self.kwargs_init.get(k, v.default) for k, v in all_vars.items()  if not k in ['self', 'slots', 'flights']}
		function_approx_model = FunctionApprox(self.slots, self.flights, **kwargs_init)

		all_vars = inspect.signature(FunctionApprox.run).parameters
		kwargs = {k:kwargs_all.get(k, v.default) for k, v in all_vars.items() if not k in ['self', 'slots', 'flights']}
		prefs_function_approx = function_approx_model.run(**kwargs)

		merge_dict = {f.name:{**prefs_udpp_local[f.name], **prefs_function_approx[f.name]} for f in self.flights}
		return merge_dict


