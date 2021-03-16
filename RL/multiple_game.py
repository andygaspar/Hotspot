import gym
import numpy as np
import itertools
import contextlib
import xpress as xp

from ModelStructure.ScheduleMaker import scheduleMaker
from ModelStructure.Costs.costFunctionDict import CostFuns

from RL.agents import AgentRandomMulti
from RL.wrapper_UDPP import allocation_from_df
from RL.wrapper_UDPP import df_sch_from_flights
from RL.wrapper_UDPP import OptimalAllocationComputer #, compute_optimal_allocation
from RL.flight import Flight
from libs.tools import clock_time, print_to_void


class MultiGame(gym.Env):
	"""
	Game with iteration where a player (airline)
	modifies its declared flights characteristics (margins and costs)
	in order to minimise its real cost.
	This is using UDPP (with real cost) and ISTOP on top (with declared/fake
	cost functions).
	The state is the set of indices of its flight in the queue. The reward is the
	opposite of the sum of the cost across flights from airline player.
	"""
	def __init__(self, n_f=50, n_a=5, player='A', seed=None,
		dm=2., dj=5, dc=0.1, offset=50., cost_type='jump',
		trading_alg='nnbound', new_capacity=5.):
		
		super().__init__()

		np.random.seed(seed)
		
		self.trading_alg = trading_alg

		self.allocation_computer = OptimalAllocationComputer(trading_alg=trading_alg)
		self.allocation_computer_nnbound = OptimalAllocationComputer(trading_alg='nnbound')

		self.viewer = None
		self.player = player

		scheduleType = scheduleMaker.schedule_types(show=False)

		self.df_sch_init = scheduleMaker.df_maker(n_f, n_a, distribution=scheduleType[0], new_capacity=new_capacity)
		self.df_sch = self.df_sch_init.copy()

		self.flight_per_company = self.df_sch[['flight', 'airline']].groupby('airline').agg(list).to_dict()['flight']

		self.base_allocation = allocation_from_df(self.df_sch_init, name_slot='slot')

		self.costFun = CostFuns().costFun[cost_type]

		self.build_flights()

		df = self.df_sch_init[['slot', 'time']]
		self.slot_times = df.set_index('slot').to_dict()['time']

		#print (self.slot_times)

		self.base_cost = self.cost_of_allocation(self.base_allocation)

		self.dm = dm
		self.dc = dc
		self.dj = dj

		self.history = {}

		self.n_f_player = len(self.flight_per_company[self.player])
		
		self.observation_space =  gym.spaces.Discrete(self.n_f_player)
		self.action_space = gym.spaces.Box(0, 2, shape=(self.n_f_player, 2), dtype=int)
		#self.observation_space = gym.spaces.Discrete(len(self.flight_per_company[self.player]))

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
					# print ('company:', company)
					# print ('flight:', flight)
				slot = allocation[flight]
				#print ('slot', slot)
				time = self.slot_times[slot]
					# print ('time', time)
					# print ('eta:', self.flights[flight].eta)
					# print ('cost_true:', self.flights[flight].cost)
					# print ('cost_declared:', self.flights[flight].cost_declared)

				cost = self.flights[flight].cost_f_true(time)

				cost_tot += cost

				# print ('cost:', cost)

				cost_c[company] = cost_c.get(company, 0.) + cost

		if only_player:
			return cost_c[self.player]
		else:
			return cost_tot , cost_c

	def compute_next_state(self):
		# Build the df used as input by the optimisers
		self.df_sch = df_sch_from_flights(self.df_sch_init, self.flights)
		#print (self.df_sch)
		
		# Optimise the allocation using ISTOP or NNbound and compute back the allocation
		allocation = self.allocation_computer.compute_optimal_allocation(self.df_sch, self.costFun)

		# Compute cost for all companies for player : real cost
		cost_tot, cost_per_c = self.cost_of_allocation(allocation)

		reward = self.offset - cost_per_c[self.player]

		reward_tot = len(self.flights) * self.offset - cost_tot

		done = False

		state = [slot for name, slot in allocation.items() if name in self.flight_per_company[self.player]]

		self.history['reward_tot'] = self.history.get('reward_tot', []) + [reward_tot]
		names = self.flight_per_company[self.player]
		self.history['margin_flight'] = self.history.get('margin_flight', []) + [[self.flights[name].margin_declared for name in names]]
		self.history['jump_flight'] = self.history.get('jump_flight', []) + [[self.flights[name].jump_declared for name in names]]
		self.history['reward'] = self.history.get('reward', []) + [reward]
		self.history['reward_per_c'] = self.history.get('reward_per_c', []) + [{name:self.offset-cost for name, cost in cost_per_c.items()}]
		self.history['allocation'] = self.history.get('allocation', []) + [allocation]
		
		return state, reward, done, {} # {'cost_per_c':cost_per_c, 'allocation':allocation}

	def get_player_flight_charac(self, charac_name):
		return list(self.df_sch[charac_name][self.df_sch['airline']=='A'])

	def get_current_margins(self):
		return self.get_player_flight_charac('margins')

	def get_current_jumps(self):
		return self.get_player_flight_charac('jumps')

	def step(self, action):
		# Apply action (modification of margin and cost) on flights
		self.apply_action(action)

		return self.compute_next_state()

	def step_absolute(self, charac):
		
		self.set_charac(charac)
		
		return self.compute_next_state()

	def get_charac(self):
		return np.array([self.get_player_flight_charac('margins'), self.get_player_flight_charac('jumps')])
		# return np.array([(self.flights[name].margin_declared, self.flights[name].jump_declared) for name in self.flight_per_company[self.player]])

	def set_charac(self, charac):
		for i, ac in enumerate(charac):
			name = self.flight_per_company[self.player][i]
			
			new_cost = self.flights[name].cost_declared# + dc

			self.flights[name].set_declared_charac(margin=ac[0], cost=new_cost, jump=ac[1])

	def reset(self):
		# RESET schedules too...
		state = [slot for name, slot in self.base_allocation.items() if name in self.flight_per_company[self.player]]
		
		# Find best NN allocation
		self.best_allocation = self.allocation_computer_nnbound.compute_optimal_allocation(self.df_sch, self.costFun)
		cost_tot, cost_per_c = self.cost_of_allocation(self.best_allocation)
		self.best_reward = self.offset - cost_per_c[self.player]
		self.best_reward_tot = len(self.flights) * self.offset - cost_tot
		
		self.history['reward_tot'] = self.history.get('reward_tot', []) + [self.best_reward_tot]
		names = self.flight_per_company[self.player]
		self.history['margin_flight'] = self.history.get('margin_flight', []) + [[self.flights[name].margin_declared for name in names]]
		self.history['jump_flight'] = self.history.get('jump_flight', []) + [[self.flights[name].jump_declared for name in names]]
		self.history['reward'] = self.history.get('reward', []) + [self.best_reward]
		self.history['reward_per_c'] = self.history.get('reward_p~er_c', []) + [{name:self.offset - cost for name, cost in cost_per_c.items()}]
		self.history['allocation'] = self.history.get('allocation', []) + [self.best_allocation]
		
		return state

	def render(self, mode='human'):
		pass 

	def apply_action(self, action):
		for i, ac in enumerate(action):
			# for each flight
			name = self.flight_per_company[self.player][i]
			
			if ac[0]==0:
				dm = -self.dm
			elif ac[0]==1:
				dm = 0
			else:
				dm = self.dm

			if ac[1]==0:
				dj = -self.dj
			elif ac[1]==1:
				dj = 0
			else:
				dj = self.dj

			new_margin = self.flights[name].margin_declared + dm
			new_cost = self.flights[name].cost_declared# + dc
			new_jump = self.flights[name].jump_declared + dj

			#print (new_jump)

			self.flights[name].set_declared_charac(margin=new_margin, cost=new_cost, jump=new_jump)


class MultiGameJump(gym.Env):
	"""
	Game with iteration where a player (airline)
	modifies its declared flights characteristics (margins and costs)
	in order to minimise its real cost.
	This is using UDPP (with real cost) and ISTOP on top (with declared/fake
	cost functions).
	The state is the set of indices of its flight in the queue. The reward is the
	opposite of the sum of the cost across flights from airline player.
	"""
	def __init__(self, n_f=50, n_a=5, player='A', seed=None,
		dm=2., dj=5, dc=0.1, offset=50., cost_type='jump',
		trading_alg='nnbound', new_capacity=5.):
		
		super().__init__(n_f=n_f, n_a=n_a, player=player, seed=seed,
		dm=dm, dj=dj, dc=dc, offset=offset, cost_type=cost_type,
		trading_alg=trading_alg, new_capacity=new_capacity)

		self.action_space = gym.spaces.Box(0, 2, shape=(self.n_f_player,), dtype=int)
			
	def set_charac(self, charac):
		for i, ac in enumerate(charac):
			name = self.flight_per_company[self.player][i]
			
			new_cost = self.flights[name].cost_declared# + dc
			new_margin = self.flights[name].margin_declared

			self.flights[name].set_declared_charac(margin=new_margin, cost=new_cost, jump=ac)

	def get_charac(self):
		return np.array([self.get_player_flight_charac('jumps')])

	def apply_action(self, action):
		for i, ac in enumerate(action):
			# for each flight
			name = self.flight_per_company[self.player][i]
			
			# if ac[0]==0:
			# 	dm = -self.dm
			# elif ac[0]==1:
			# 	dm = 0
			# else:
			# 	dm = self.dm

			if ac==0:
				dj = -self.dj
			elif ac==1:
				dj = 0
			else:
				dj = self.dj

			new_margin = self.flights[name].margin_declared# + dm
			new_cost = self.flights[name].cost_declared# + dc
			new_jump = self.flights[name].jump_declared + dj

			#print (new_jump)

			self.flights[name].set_declared_charac(margin=new_margin, cost=new_cost, jump=new_jump)


class MultiStochGame(gym.Env):
	"""
	Game with iteration where a player (airline)
	modifies its declared flights characteristics (margins and costs)
	in order to minimise its real cost, but on different allocation/hotstop every time.

	This is using UDPP (with real cost) and ISTOP or nnBound on top (with declared/fake
	cost functions).

	The state is the set of indices of its flight in the queue. The reward is the
	opposite of the sum of the cost across flights from airline player (plus an offset).
	"""
	def __init__(self, n_f=50, n_a=5, player='A', seed=None,
		dm=2., dj=5, dc=0.1, offset=50., cost_type='jump',
		trading_alg='nnbound', n_f_player=2, new_capacity=5.):
		
		super().__init__()

		np.random.seed(seed)
		
		self.trading_alg = trading_alg
		self.allocation_computer = OptimalAllocationComputer(trading_alg=trading_alg)
		self.allocation_computer_nnbound = OptimalAllocationComputer(trading_alg='nnbound')

		self.viewer = None
		self.player = player

		self.costFun = CostFuns().costFun[cost_type]

		self.dm = dm
		self.dc = dc
		self.dj = dj

		self.n_f = n_f
		self.n_a = n_a

		self.history = {}

		self.n_f_player = n_f_player

		self.observation_space =  gym.spaces.Box(0, 50, shape=(self.n_f_player, 3), dtype=int)
		self.action_space = gym.spaces.Box(0, 2, shape=(self.n_f_player, 2), dtype=int)
		
		self.offset = offset

		self.new_capacity = new_capacity

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
		# True cost
		cost_tot = 0.
		cost_c = {}

		#print ('POUET', self.flight_per_company)
		for company, flights in self.flight_per_company.items():
			for flight in flights:
					# print ('company:', company)
					# print ('flight:', flight)
				slot = allocation[flight]
				#print ('slot', slot)
				time = self.slot_times[slot]
					# print ('time', time)
					# print ('eta:', self.flights[flight].eta)
					# print ('cost_true:', self.flights[flight].cost)
					# print ('cost_declared:', self.flights[flight].cost_declared)

				cost = self.flights[flight].cost_f_true(time)

				cost_tot += cost

				# print ('cost:', cost)

				cost_c[company] = cost_c.get(company, 0.) + cost

		if only_player:
			return cost_c[self.player]
		else:
			return cost_tot , cost_c

	def cost_of_allocation_declared(self, allocation, only_player=False):
		# True cost
		cost_tot = 0.
		cost_c = {}

		for company, flights in self.flight_per_company.items():
			for flight in flights:
					# print ('company:', company)
					# print ('flight:', flight)
				slot = allocation[flight]
				#print ('slot', slot)
				time = self.slot_times[slot]
					# print ('time', time)
					# print ('eta:', self.flights[flight].eta)
					# print ('cost_true:', self.flights[flight].cost)
					# print ('cost_declared:', self.flights[flight].cost_declared)

				cost = self.flights[flight].cost_f_declared(time)

				cost_tot += cost

				# print ('cost:', cost)

				cost_c[company] = cost_c.get(company, 0.) + cost

		if only_player:
			return cost_c[self.player]
		else:
			return cost_tot , cost_c

	def make_new_schedules(self):
		# Prepare new hotspot
		scheduleType = scheduleMaker.schedule_types(show=False)

		self.df_sch_init = scheduleMaker.df_maker(self.n_f,
												self.n_a,
												distribution=scheduleType[0],
												new_capacity=self.new_capacity,
												n_flight_first_airline=self.n_f_player)

		self.df_sch = self.df_sch_init.copy()

		self.flight_per_company = self.df_sch[['flight', 'airline']].groupby('airline').agg(list).to_dict()['flight']
		df = self.df_sch_init[['slot', 'time']]
		self.slot_times = df.set_index('slot').to_dict()['time']

		self.build_flights()
		self.player_flights = self.flight_per_company[self.player]

		self.base_allocation = allocation_from_df(self.df_sch_init, name_slot='slot')
		self.base_cost_tot, self.base_cost_per_c = self.cost_of_allocation(self.base_allocation)

		self.best_allocation = self.allocation_computer.compute_optimal_allocation(self.df_sch, self.costFun)
		self.best_cost_tot, self.best_cost_per_c = self.cost_of_allocation(self.best_allocation)

	def compute_reward(self):
		# Build the df used as input by the optimisers
		self.df_sch = df_sch_from_flights(self.df_sch_init, self.flights)
		#print (self.df_sch)
		
		# Optimise the allocation using ISTOP or NNbound and compute back the allocation
		allocation = self.allocation_computer.compute_optimal_allocation(self.df_sch, self.costFun)

		# Compute cost for all companies for player : real cost
		# real cost
		cost_tot, cost_per_c = self.cost_of_allocation(allocation)
		# declared cost
		cost_tot_declared, cost_per_c_declared = self.cost_of_allocation_declared(allocation)
		
		# Reward is relative to cost in intial allocation
		reward = self.offset - (cost_per_c[self.player] - self.best_cost_per_c[self.player])
		reward_fake = self.offset - (cost_per_c_declared[self.player] - self.best_cost_per_c[self.player])

		# print('cost_tot, cost_per_c', cost_tot, cost_per_c)
		# print (self.df_sch)

		reward_tot = len(self.flights) * self.offset - (cost_tot-self.best_cost_tot)

		#state = [slot for name, slot in allocation.items() if name in self.flight_per_company[self.player]]
		
		return reward, reward_tot, cost_tot, cost_per_c, allocation, reward_fake

	def get_player_flight_charac(self, charac_name):
		return list(self.df_sch['margins'][self.df_sch['airline']=='A'])

	def get_current_margins(self):
		return self.get_player_flight_charac('margins')

	def get_current_jumps(self):
		return self.get_player_flight_charac('jumps')

	def get_state(self):
		#state = [slot for name, slot in allocation.items() if name in self.flight_per_company[self.player]]
		state = self.df_sch.set_index('flight').loc[self.player_flights, ['margins', 'jump', 'time']]#, 'slot']]
		return state

	def step(self, action):
		# Apply action (modification of margin and cost) on flights
		#with clock_time(message_after='apply_action executed in'):
		self.apply_action(action)
		
		#with clock_time(message_after='reward_computation executed in'):
			# Compute reward
		reward, reward_tot, cost_tot, cost_per_c, allocation, reward_fake = self.compute_reward()

		# Remember stuff
		names = self.flight_per_company[self.player]
		# self.history['df_sch_init'] = self.history.get('df_sch_init', []) + [self.df_sch_init]
		# self.history['df_sch'] = self.history.get('df_sch', []) + [self.df_sch]
		# self.history['margin_flight_declared'] = self.history.get('margin_flight_declared', []) + [[self.flights[name].margin_declared for name in names]]
		# self.history['jump_flight_declared'] = self.history.get('jump_flight_declared', []) + [[self.flights[name].jump_declared for name in names]]
		
		# self.history['margin_flight_true'] = self.history.get('margin_flight_true', []) + [[self.flights[name].margin for name in names]]
		# self.history['jump_flight_true'] = self.history.get('jump_flight_true', []) + [[self.flights[name].jump for name in names]]

		# # Bare costs
		# self.history['cost_per_c_baseline'] = self.history.get('cost_per_c_baseline', []) + [self.base_cost_per_c]
		# self.history['cost_per_c_init_optimal'] = self.history.get('cost_per_c_init_optimal', []) + [self.best_cost_per_c]
		# self.history['cost_per_c_final_optimal'] = self.history.get('cost_per_c_final_optimal', []) + [cost_per_c]

		# # True reward
		# self.history['reward'] = self.history.get('reward', []) + [reward]
		# self.history['reward_tot'] = self.history.get('reward_tot', []) + [reward_tot]
		# self.history['reward_per_c'] = self.history.get('reward_per_c', []) + [{name:self.offset-cost for name, cost in cost_per_c.items()}]

		
		# self.history['reward_fake'] = self.history.get('reward_fake', []) + [reward_fake]

		# self.history['base_allocation'] = self.history.get('base_allocation', []) + [self.base_allocation]
		# self.history['best_allocation'] = self.history.get('best_allocation', []) + [self.best_allocation]
		# self.history['allocation'] = self.history.get('allocation', []) + [allocation]

		done = False

		# Compute new schedule for next state
		# with clock_time(message_after='make_new_schedules executed in'):	
		self.make_new_schedules()

		state = self.get_state()

		return state, reward, done, {'history':self.history}

	def get_charac(self):
		return np.array([self.get_player_flight_charac('margins'), self.get_player_flight_charac('jumps')])

	def set_charac(self, charac):
		for i, ac in enumerate(charac):
			name = self.flight_per_company[self.player][i]
			
			new_cost = self.flights[name].cost_declared# + dc

			self.flights[name].set_declared_charac(margin=ac[0], cost=new_cost, jump=ac[1])

	def reset(self):
		# Compute new schedule for next state
		self.make_new_schedules()

		state = self.get_state()
		
		return state

	def render(self, mode='human'):
		pass 

	def apply_action(self, action):
		for i, ac in enumerate(action):
			# for each flight
			name = self.flight_per_company[self.player][i]
			
			if ac[0]==0:
				dm = -self.dm
			elif ac[0]==1:
				dm = 0
			else:
				dm = self.dm

			if ac[1]==0:
				dj = -self.dj
			elif ac[1]==1:
				dj = 0
			else:
				dj = self.dj

			new_margin = self.flights[name].margin_declared + dm
			new_cost = self.flights[name].cost_declared# + dc
			new_jump = self.flights[name].jump_declared + dj

			self.flights[name].set_declared_charac(margin=new_margin, cost=new_cost, jump=new_jump)


class MultiStochGameJump(MultiStochGame):
	"""
	Game with iteration where a player (airline)
	modifies its declared flights characteristics (margins and costs)
	in order to minimise its real cost, but on different allocation/hotstop every time.

	This is using UDPP (with real cost) and ISTOP or nnBound on top (with declared/fake
	cost functions).

	The state is the set of indices of its flight in the queue. The reward is the
	opposite of the sum of the cost across flights from airline player (plus an offset).
	"""
	def __init__(self, n_f=50, n_a=5, player='A', seed=None,
		dm=2., dj=5, dc=0.1, offset=50., cost_type='jump',
		trading_alg='nnbound', n_f_player=2, new_capacity=5.):
		
		super().__init__(n_f=n_f, n_a=n_a, player=player, seed=seed,
		dm=dm, dj=dj, dc=dc, offset=offset, cost_type=cost_type,
		trading_alg=trading_alg, new_capacity=new_capacity,
		n_f_player=n_f_player)

		#self.action_space = gym.spaces.Box(0, 2, shape=(self.n_f_player,), dtype=int)
		self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(3)]*self.n_f_player)

		self.n_action = (self.n_f_player, 3)

	def apply_action(self, action):
		for i, ac in enumerate(action):
			# for each flight
			name = self.flight_per_company[self.player][i]
			
			if ac==0:
				dj = -self.dj
			elif ac==1:
				dj = 0
			else:
				dj = self.dj

			new_margin = self.flights[name].margin_declared# + dm
			new_cost = self.flights[name].cost_declared# + dc
			new_jump = self.flights[name].jump_declared + dj

			#print (new_jump)

			self.flights[name].set_declared_charac(margin=new_margin, cost=new_cost, jump=new_jump)


class MultiStochGameMargin(MultiStochGameJump):
	def apply_action(self, action):
		for i, ac in enumerate(action):
			# for each flight
			name = self.flight_per_company[self.player][i]
			
			if ac==0:
				dm = -self.dm
			elif ac==1:
				dm = 0
			else:
				dm = self.dm

			# if ac==0:
			# 	dj = -self.dj
			# elif ac==1:
			# 	dj = 0
			# else:
			# 	dj = self.dj

			new_margin = self.flights[name].margin_declared + dm
			new_cost = self.flights[name].cost_declared# + dc
			new_jump = self.flights[name].jump_declared# + dj

			#print (new_jump)

			self.flights[name].set_declared_charac(margin=new_margin, cost=new_cost, jump=new_jump)


class MultiStochGameMarginFlatSpaces(MultiStochGameMargin):
	# Flat input/output for tf-agent use
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		
		#self.observation_space =  gym.spaces.Box(0, 50, shape=(self.n_f_player, 4), dtype=int)
		self.observation_space =  gym.spaces.Box(0, 100, shape=(self.n_f_player*3, ), dtype=int)
		
		# Flatten actions
		self.action_registery = {i:t for i, t in enumerate(list(itertools.product(*([[0, 1, 2]]*self.n_f_player))))}
		self.action_space = gym.spaces.Box(0,
										   len(self.action_registery)-1,
										   shape=(),
										   dtype=int)
		
	def apply_action(self, action):
		action = self.action_registery[int(action)]
		super().apply_action(action)
		
	def get_state(self):
		state = super().get_state()
		return tuple(np.array(state).flatten())


class MultiStochGameJumpFlatSpaces(MultiStochGameJump):
	# Flat input/output for tf-agents use
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		
		#self.observation_space =  gym.spaces.Box(0, 50, shape=(self.n_f_player, 4), dtype=int)
		self.observation_space =  gym.spaces.Box(0, 100, shape=(self.n_f_player*3, ), dtype=int)
		
		# Flatten actions
		self.action_registery = {i:t for i, t in enumerate(list(itertools.product(*([[0, 1, 2]]*self.n_f_player))))}
		self.action_space = gym.spaces.Box(0,
										   len(self.action_registery)-1,
										   shape=(),
										   dtype=int)
		
	def apply_action(self, action):
		action = self.action_registery[int(action)]
		super().apply_action(action)
		
	def get_state(self):
		state = super().get_state()
		return tuple(np.array(state).flatten())