import gym
import numpy as np

from ModelStructure.ScheduleMaker import scheduleMaker
from ModelStructure.Costs.costFunctionDict import CostFuns

from RL.wrapper_UDPP import allocation_from_df
from RL.wrapper_UDPP import df_sch_from_flights
from RL.wrapper_UDPP import compute_optimal_allocation
from RL.flight import Flight
class SingleGame(gym.Env):
	"""
	Game with iteration where a player (airline)
	modifies its declared flights characteristics (margins and costs)
	in order to minimise its real cost.
	This is using UDPP (with real cost) and ISTOP on top (with declared/fake
	cost functions).

	# UDPATE FOLLOWING
	The state is the set of indices of its flight in the queue. The reward is the
	opposite of the sum of the cost across flights from airline player.
	"""
	def __init__(self, n_f=50, n_a=5, player='A', seed=None, n_first_flights=1,
		dm=2., dj=5, dc=0.1, offset=50., cost_type='jump'):
		super().__init__()

		np.random.seed(seed)

		self.viewer = None
		self.player = player

		scheduleType = scheduleMaker.schedule_types(show=False)

		self.df_sch = scheduleMaker.df_maker(n_f, n_a, distribution=scheduleType[0])

		self.base_allocation = allocation_from_df(self.df_sch, name_slot='slot')

		self.costFun = CostFuns().costFun[cost_type]

		self.build_flights()
		 
		df = self.df_sch[['slot', 'time']]
		self.slot_times = df.set_index('slot').to_dict()['time']
		
		#print (self.slot_times)
		
		self.base_cost = self.cost_of_allocation(self.base_allocation)

		self.n_first_flights = n_first_flights
		
		self.dm = dm
		self.dc = dc
		self.dj = dj
		
		self.history = {}
		
		self.observation_space =  gym.spaces.Discrete(len(self.flight_per_company[self.player]))
		self.action_space = gym.spaces.Discrete(3)#len(self.flight_per_company[self.player]))
		
		#self.observation_space.shape, self.action_space.shape = (, ), (len(self.flight_per_company['A']), 2)
		
		self.offset = offset

	def build_flights(self):
		self.flights = {}
		for i, row in self.df_sch.iterrows():
			self.flights[row['flight']] = Flight(row['eta'],
											   name=row['flight'],
											  margin=row['margins'],
											  cost=row['cost'],
												jump=row['jump'],
											  cost_function=self.costFun,
												)

		self.flight_per_company = {company:list(self.df_sch[self.df_sch['airline']==company]['flight'])
									   for company in self.df_sch['airline'].unique()}

	def cost_of_allocation(self, allocation, only_player=False):
		cost_tot = 0.
		cost_c = {}
		
		#print ('POUET', self.flight_per_company)
		for company, flights in self.flight_per_company.items():
			for flight in flights:
#                 print ('company:', company)
#                 print ('flight:', flight)
				slot = allocation[flight]
				#print ('slot', slot)
				time = self.slot_times[slot]
#                 print ('time', time)
#                 print ('eta:', self.flights[flight].eta)
#                 print ('cost_true:', self.flights[flight].cost)
#                 print ('cost_declared:', self.flights[flight].cost_declared)

				cost = self.flights[flight].cost_f_true(time)
				
				cost_tot += cost
				
				# print ('cost:', cost)

				cost_c[company] = cost_c.get(company, 0.) + cost

		if only_player:
			return cost_c[self.player]
		else:
			return cost_tot , cost_c

	def step(self, action):
		# Apply action (modification of margin and cost) on flights
		self.apply_action(action)

		# Build the df used as input by the optimisers
		df_sch = df_sch_from_flights(self.df_sch, self.flights)
		
		#print (df_sch)

		#print (df_sch)

		# Optimise the allocation using ISTOP and compute back the allocation
		allocation = compute_istop_allocation(df_sch, self.costFun)

		# Compute cost for all companies for player : real cost
		cost_tot, cost_per_c = self.cost_of_allocation(allocation)

		reward = self.offset - cost_per_c[self.player]

		reward_tot = len(self.flights) * self.offset - cost_tot

		done = False

		state = [slot for name, slot in allocation.items() if name in self.flight_per_company[self.player]]

		self.history['reward_tot'] = self.history.get('reward_tot', []) + [reward_tot]
		name = self.flight_per_company[self.player][0]
		self.history['margin_flight'] = self.history.get('margin_flight', []) + [self.flights[name].margin_declared]
		
		return state, reward, done, {} # {'cost_per_c':cost_per_c, 'allocation':allocation}

	def reset(self):
		state = [slot for name, slot in self.base_allocation.items() if name in self.flight_per_company[self.player]]
		return state

	def render(self, mode='human'):
		pass 

	def apply_action(self, action):
		name = self.flight_per_company[self.player][0]
		if action[0]==0:
			dm = -self.dm
		elif action[0]==1:
			dm = 0
		else:
			dm = self.dm
			
		if action[1]==0:
			dj = -self.dj
		elif action[1]==1:
			dj = 0
		else:
			dj = self.dj
				
		new_margin = self.flights[name].margin_declared + dm
		new_cost = self.flights[name].cost_declared# + dc
		new_jump = self.flights[name].jump_declared + dj

		self.flights[name].set_declared_charac(margin=new_margin, cost=new_cost, jump=new_jump)