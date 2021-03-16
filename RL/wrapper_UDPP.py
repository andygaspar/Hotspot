from collections import OrderedDict

from UDPP import udppModel
from ModelStructure.Costs.costFunctionDict import CostFuns
from Istop import istop
from NNBound import nnBound
from GlobalOptimum import globalOptimum
from libs.tools import print_to_void, clock_time

def allocation_from_df(df, name_slot='new slot'):
	return OrderedDict(df[['flight', name_slot]].set_index('flight').to_dict()[name_slot])

def df_sch_from_flights(schedules, flights):
	list_name = schedules['flight']
	margins = [flights[name].margin_declared for name in list_name]
	costs = [flights[name].cost_declared for name in list_name]
	jumps = [flights[name].jump_declared for name in list_name]
	
	new_df = schedules.copy(deep=True)
	new_df.loc[:, 'margins'] = margins
	new_df.loc[:, 'cost'] = costs
	new_df.loc[:, 'jump'] = jumps
	
	return new_df


class OptimalAllocationComputer:
	def __init__(self, trading_alg='istop'):
		
		# UDPP stage
		with print_to_void():
			self.udpp_model = udppModel.UDPPmodel(df_init=None, costFun=None)
			
			# Trading stage
			if trading_alg=='istop':
				self.model = istop.Istop(df_init=None, costFun=None)
			elif trading_alg=='nnbound':
				self.model = nnBound.NNBoundModel(df_init=None, costFun=None)
			elif trading_alg=='globaloptimum':
				self.model = globalOptimum.GlobalOptimumModel(df_init=None, costFun=None)
			else:
				raise Exception('Unrecognised allocation algorithm:', trading_alg, '. Valid options are:', ['istop', 'nnbound', 'globaloptimum'])

	def compute_optimal_allocation(self, df_sch, costFun):
		df = df_sch.copy(deep=True)

		# with clock_time(message_after='udpp_model reset executed in'):
		self.udpp_model.reset(df, costFun)
		# with clock_time(message_after='udpp_model run executed in'):
		self.udpp_model.run(optimised=True)
		
		# with clock_time(message_after='model reset executed in'):
		self.model.reset(self.udpp_model.solution, costFun)
		# with clock_time(message_after='model run executed in'):
		self.model.run()

		allocation = allocation_from_df(self.model.solution)

		return allocation 

# def compute_optimal_allocation(df_sch, costFun, trading_alg='istop'):
# 	print ('BAH')
# 	with print_to_void():
# 		print ('yolo')
# 		# Copy schedule dataframe and build UDPP model
# 		df = df_sch.copy(deep=True)
# 		udpp_model = udppModel.UDPPmodel(df, costFun)
# 		# Run the model (with true cost functions)
# 		udpp_model.run(optimised=True)

# 		# Build istop model
# 		if trading_alg=='istop':
# 			istop_model = istop.Istop(udpp_model.solution, costFun)
# 			istop_model.run()
# 			allocation = allocation_from_df(istop_model.solution)
# 		elif trading_alg=='nnbound':
# 			nnBound_model = nnBound.NNBoundModel(udpp_model.solution, costFun)
# 			nnBound_model.run()
# 			allocation = allocation_from_df(nnBound_model.solution)
# 		elif trading_alg=='globaloptimum':
# 			globalOptimum_model = globalOptimum.GlobalOptimumModel(udpp_model.solution, costFun)
# 			globalOptimum_model.run()
# 			allocation = allocation_from_df(globalOptimum_model.solution)
# 		else:
# 			raise Exception('UNrecognised allocation algorithm:', trading_alg, '. Valid options are:', ['istop', 'nnbound', 'globaloptimum'])
# 		print ('boum')
# 		return allocation