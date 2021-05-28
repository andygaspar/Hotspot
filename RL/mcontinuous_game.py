import gym
import numpy as np

from copy import copy, deepcopy

from Hotspot.ScheduleMaker import scheduleMaker
from Hotspot.ModelStructure.Costs.costFunctionDict import CostFuns

#from Hotspot.RL.wrapper import df_sch_from_flights
from Hotspot.RL.wrapper import OptimalAllocationComputer, Flight, df_from_flights, allocation_from_flights
from Hotspot.RL.agents import Agent
#from Hotspot.RL.flight import Flight
from Hotspot.libs.uow_tool_belt.general_tools import clock_time
from Hotspot.libs.other_tools import compare_allocations

def linear_function(min_y, max_y):
	def f(x):
		return (max_y-min_y) * x + min_y

	return f

def norm_function(min_y, max_y):
	def f(x):
		return (x - min_y)/(max_y-min_y)

	return f


class ContMGame(gym.Env):
	"""
	Game with iteration where a player (airline)
	modifies its declared flights characteristics (margins and costs)
	in order to minimise its real cost, but on different allocation/hotstop every time.

	This is using UDPP (with real cost) and ISTOP or nnBound on top (with declared/fake
	cost functions).

	The state is the set of indices of its flight in the queue. The reward is the
	opposite of the sum of the cost across flights from airline player (plus an offset).
	"""
	agent_collection = {}

	def __init__(self, n_f=10, n_a=3, players=None, seed=None,
		offset=100., cost_type='jump', min_jump=10, max_jump=100,
		algo='nnbound', n_f_players=[4, 3], new_capacity=5.,
		min_margin=10, max_margin=45, price_jump=0., price_cost=0.,
		price_margin=0., min_cost=0.1, max_cost=2., normed_state=True,
		min_margin_action=None, max_margin_action=None,
		min_jump_action=None, max_jump_action=None):

		super().__init__()

		self.normed_state = normed_state

		self.set_price_jump(price_jump)
		self.set_price_cost(price_cost)
		self.set_price_margin(price_margin)

		self.min_margin = int(min_margin)
		self.max_margin = int(max_margin)
		if not min_margin_action is None:
			self.min_margin_action = int(min_margin_action)
		else:
			self.min_margin_action = int(min_margin)
		if not max_margin_action is None:
			self.max_margin_action = int(max_margin_action)
		else:
			self.max_margin_action = int(max_margin)

		self.min_jump = int(min_jump)
		self.max_jump = int(max_jump)
		if not min_jump_action is None:
			self.min_jump_action = int(min_jump_action)
		else:
			self.min_jump_action = int(min_jump)
		if not max_jump_action is None:
			self.max_jump_action = int(max_jump_action)
		else:
			self.max_jump_action = int(max_jump)

		self.min_cost = int(min_cost)
		self.max_cost = int(max_cost)

		self.norm_margin = norm_function(min_margin, max_margin)
		self.norm_jump = norm_function(min_jump, max_jump)
		min_time = 0.
		max_time = (new_capacity -1)* n_f
		self.norm_time = norm_function(min_time, max_time)

		self.func_jump = linear_function(self.min_jump_action, self.max_jump_action)
		self.func_margin = linear_function(self.min_margin_action, self.max_margin_action)

		np.random.seed(seed)

		self.algo = algo
		self.allocation_computer = OptimalAllocationComputer(algo=algo)
		
		self.viewer = None

		if players is None:
			self.players = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')[:len(n_f_players)]
		else:
			self.players = players

		if len(self.players)==n_a:
			try:
				assert sum(n_f_players)==n_f
			except AssertionError:
				raise Exception ('The sum of flights ({}) is not equal to flights passed as list ({}).'.format(n_f, n_f_players))

		self.costFun = CostFuns().costFun[cost_type]

		self.n_f = n_f
		self.n_a = n_a

		self.history = {}

		self.n_f_players = n_f_players

		self.n_f_players_dict = {player:n_f_players[i] for i, player in enumerate(self.players)}

		self.players_id = {player:i for i, player in enumerate(self.players)}
		self.observation_space =  gym.spaces.Box(0, 100, shape=(sum(self.n_f_players)*3, ))#, dtype=int)

		self.action_space = gym.spaces.Box(0, 1, shape=(sum(self.n_f_players)*2, ))#, dtype=float)

		self.offset = offset

		self.new_capacity = new_capacity

	def game_summary(self):
		print ('Number of flights:', self.n_f)
		print ('Number of airlines:', self.n_a)
		print ('Player names:', self.players)
		print ('Number of flights for each player:', self.n_f_players)
		print ('New capacity of hotspot:', self.new_capacity)
		print ('Min/Max margins:', self.min_margin, self.max_margin)
		print ('Min/Max jumps:', self.min_jump, self.max_jump)
		print ('Trading algorithm:', self.algo)

	def build_agent_from_collection(self, kind='', name='', as_player=None):
		agentClass = self.agent_collection[kind]

		agent = agentClass(kind=kind, gym_env=self, name=name, as_player=as_player)

		return agent

	def build_action_selector_for(self, player):
		return lambda x: self.extract_action_like_from_array(x, player=player)

	def cost_of_allocation(self, allocation, only_for_player=None):
		"""
		Note: true cost s extracted via the cost function of the flights,
		not the delayCostVect (that represents declared costs).
		"""
		# True cost
		cost_tot = 0.
		cost_c = {}

		transferred_cost = 0.

		for company, flights in self.flight_per_company.items():
			for flight in flights:
				slot = allocation[flight]
				time = self.slot_times[slot]

				# Penalty for changing their declared cost function.
				dj = self.flights[flight].jump1 - self.flights[flight].jump1_declared
				dc = self.flights[flight].slope - self.flights[flight].slope_declared
				dm = -(self.flights[flight].margin1 - self.flights[flight].margin1_declared)

				added_cost = dj*self.jump_price + dc*self.cost_price + dm*self.margin_price

				transferred_cost += added_cost

				cost = self.flights[flight].cost_f_true(time) + added_cost

				cost_tot += cost

				cost_c[company] = cost_c.get(company, 0.) + cost

		transferred_cost_per_c = transferred_cost/self.n_f
		for company, flights in self.flight_per_company.items():
			for flight in flights:
				transferred_cost_per_c
				cost_c[company] -= transferred_cost_per_c

		if not only_for_player is None:
			return deepcopy(cost_c[only_for_player]), deepcopy(transferred_cost)
		else:
			return deepcopy(cost_tot), deepcopy(cost_c), deepcopy(transferred_cost)

	def cost_of_allocation_declared(self, allocation, only_for_player=None):
		# True cost
		cost_tot = 0.
		cost_c = {}

		for company, flights in self.flight_per_company.items():
			for flight in flights:
				slot = allocation[flight]

				time = self.slot_times[slot]

				cost = self.flights[flight].cost_f_declared(time)

				cost_tot += cost

				cost_c[company] = cost_c.get(company, 0.) + cost

		if not only_for_player is None:
			return cost_c[only_for_player]
		else:
			return cost_tot , cost_c

	def extract_action_like_from_array(self, ar, player=''):
		"""
		Allows to extract from an action-like array the part related to the player.
		"""
		lim1, lim2 = self.action_selector[player]

		return ar[lim1:lim2]

	def make_new_schedules(self):
		# Prepare new hotspot
		scheduleType = scheduleMaker.schedule_types(show=False)

		# Drawing some flights and slots ar random
		self.slots, hflights = scheduleMaker.slots_flights_maker(self.n_f,
																self.n_a,
																distribution=scheduleType[0],
																new_capacity=self.new_capacity,
																n_flights_first_airlines=self.n_f_players,
																min_margin=self.min_margin,
																max_margin=self.max_margin,
																min_jump=self.min_jump,
																max_jump=self.max_jump)

		self.slot_times = {slot.index:slot.time for slot in self.slots}

		# Wrapping flights
		flights = [Flight(hflight=hflight, cost_function_paras=self.costFun) for hflight in hflights]

		# Compute cost vector for slots
		[flight.compute_cost_vect(self.slots) for flight in flights]
		
		# Create a dict, easier to handle
		self.flights = {flight.name:flight for flight in flights}

		# Another dictionary for companies 
		self.flight_per_company = {}
		for flight in self.flights.values():
			self.flight_per_company[flight.airlineName] = self.flight_per_company.get(flight.airlineName, []) + [flight.name]

		# Keep a snapshot of the intial state
		self.df_sch_base = df_from_flights(self.flights, name_slot='slot')

		#self.build_flights(slots)
		self.players_flights = [flight for player in self.players for flight in self.flight_per_company[player]]

		# Compute cost in base allocation (FPFS)
		self.base_allocation = allocation_from_flights(flights, name_slot='slot')
		self.base_cost_tot, self.base_cost_per_c, _ = self.cost_of_allocation(self.base_allocation)
		
		# Compute costs with truthful margins/jumps
		self.best_allocation = self.allocation_computer.compute_optimal_allocation(self.slots, self.flights)
		self.best_cost_tot, self.best_cost_per_c, _ = self.cost_of_allocation(self.best_allocation)
		
		cost = 0.
		for i, name in enumerate(self.flights.keys()):
			s = self.best_allocation[name]
			cost += self.flights[name].costVect[s]
		
	def compute_reward(self):
		# Optimise the allocation using ISTOP or NNbound and compute back the allocation
		allocation = self.allocation_computer.compute_optimal_allocation(self.slots, self.flights)
		
		# Compute cost for all companies for player : real cost
		# real cost (not: using cost function, not delayCostVect)
		cost_tot, cost_per_c, transferred_cost = self.cost_of_allocation(allocation)
		# declared cost
		cost_tot_declared, cost_per_c_declared = self.cost_of_allocation_declared(allocation)

		# Reward is relative to cost in intial allocation
		rewards = [self.offset - (cost_per_c[player] - self.best_cost_per_c[player]) for player in self.players]
		rewards_fake = {player:self.offset - (cost_per_c_declared[player] - self.best_cost_per_c[player]) for player in self.players}

		reward_tot = len(self.flights) * self.offset - (cost_tot-self.best_cost_tot)

		return rewards, reward_tot, cost_tot, cost_per_c, allocation, rewards_fake, transferred_cost

	def get_player_flight_charac(self, player, charac_name):
		return list(self.df_sch['margins'][self.df_sch['airline']==player])

	def get_current_margins(self, player):
		return self.get_player_flight_charac(player, 'margins')

	def get_current_jumps(self, player):
		return self.get_player_flight_charac(player, 'jumps')

	def get_state(self):
		#state = [slot for name, slot in allocation.items() if name in self.flight_per_company[self.player]]
		#state = self.df_sch.set_index('flight').loc[self.players_flights, ['margins', 'jump', 'time']]#, 'slot']]
		# TODO: in the following, we are using "eta", whereas previously we were using "time", 
		# which corresponds to the FPFS (initial slot). Check that it's ok.
		state = np.array([(self.flights[flight].margin1, self.flights[flight].jump1, self.flights[flight].eta) for flight in self.players_flights])
		#print (state)
		#print ('A', state)
		if self.normed_state:
			# state = np.array([np.array(state['margins'].apply(self.norm_margin)),
			# 		np.array(state['jump'].apply(self.norm_jump)),
			# 		np.array(state['time'].apply(self.norm_time))]).T

			state = np.array([np.vectorize(self.norm_margin)(state.T[0]),
								np.vectorize(self.norm_jump)(state.T[1]),
								np.vectorize(self.norm_time)(state.T[2])]).T

		#print ('B', state)
		state = tuple(np.array(state).flatten())
		return state

	def set_price_jump(self, price):
		self.jump_price = price

	def set_price_cost(self, price):
		self.cost_price = price

	def set_price_margin(self, price):
		self.margin_price = price

	def step(self, action):
		# Apply action (modification of margin and cost) on flights
		self.apply_action(action)

		# Compute reward
		rewards, rewards_tot, cost_tot, cost_per_c, allocation, rewards_fake, transferred_cost = self.compute_reward()

		# Remember stuff
		names = [self.flight_per_company[player] for player in self.players]
		df_sch = df_from_flights(self.flights, name_slot='newSlot')

		done = False

		information = {'allocation':allocation,
						'df_sch':df_sch,
						'df_sch_base':self.df_sch_base.copy(),
						'best_allocation':copy(self.best_allocation),
						'base_allocation':copy(self.base_allocation),
						'transferred_cost':transferred_cost,
						'rewards_tot':rewards_tot,
						'cost_tot':cost_tot,
						'cost_per_c':cost_per_c,
						'rewards_fake':rewards_fake,
						'transferred_cost':transferred_cost,
						'best_cost_per_c':deepcopy(self.best_cost_per_c),
						'rewards':rewards,
						'base_cost_per_c':deepcopy(self.base_cost_per_c)}	

		# Compute new schedule for next state
		self.make_new_schedules()

		state = self.get_state()

		return state, np.array(rewards), done, information

	def get_charac(self, player):
		return np.array([self.get_player_flight_charac(player, 'margins'), self.get_player_flight_charac(player, 'jumps')])

	def set_charac(self, charac):
		for player in self.players:
			for i, ac in enumerate(charac):
				name = self.flight_per_company[player][i]

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
		for i in range(len(self.n_f_players)):
			player = self.players[i]
			if i==0:
				nfp1 = 0
			else:
				nfp1 = self.n_f_players[i-1]
				
			if i==len(self.n_f_players)-1:
				nfp2 = len(self.n_f_players)
			else:
				nfp2 = self.n_f_players[i]

			# this is probably wrong, see other methods below.
			action_p = action[nfp1*2:nfp2*2].reshape(nfp2-nfp1, 2)
			
			for i, ac in enumerate(action_p):
				# for each flight
				name = self.flight_per_company[player][i]

				new_cost = self.flights[name].cost_declared

				scaled_action_margin = self.func_margin(ac[0])
				scaled_action_jump = self.func_jump(ac[1])

				self.flights[name].set_declared_charac(margin=scaled_action_margin, cost=new_cost, jump=scaled_action_jump)

				# Update cost vect for each flight
				self.flights[name].compute_cost_vect(self.slots, declared=True)


class MaxAgentMargin(Agent):
	"""
	Select always the lowest margin possible.
	"""
	def __init__(self, kind='max', gym_env=None, name='', as_player=None):
		super().__init__(kind=kind, name=name)
		if as_player is None:
			as_player = name

		self.high = gym_env.action_space.low
		self.action_selector = gym_env.build_action_selector_for(as_player)

	def action(self, observation):
		action = self.action_selector(self.high)
		return action


class HonestAgentMargin(Agent):
	def __init__(self, kind='honest', gym_env=None, name='', as_player=None):
		super().__init__(kind=kind, name=name)
		if as_player is None:
			as_player = name

		self.n_f = gym_env.n_f_players_dict[as_player]

	def action(self, observation):
		# extract the (real) jumps and pass them down as action
		# TODO: works only for jumps...
		state = np.array(observation).reshape(self.n_f, 3)
		action = state[:, 0]
		#print (self, action)
		return action


class RandomAgentMargin(Agent):
	def __init__(self, kind='random', gym_env=None, name='', as_player=None):
		super().__init__(kind=kind, name=name)
		if as_player is None:
			as_player = name

		self.draw_func = gym_env.action_space.sample
		#self.lims = lims
		self.action_selector = gym_env.build_action_selector_for(as_player)

	def action(self, observation):
		#action = self.draw_func()[self.lims[0]:self.lims[1]]
		action = self.action_selector(self.draw_func())
		#print (self, action)
		return action


class ContMGameMargin(ContMGame):
	agent_collection = {'max':MaxAgentMargin,
						'honest':HonestAgentMargin,
						'random':RandomAgentMargin}

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		if self.normed_state:
			self.observation_space =  gym.spaces.Box(0., 1., shape=(sum(self.n_f_players)*3, ))#, dtype=int)
		else:
			self.observation_space =  gym.spaces.Box(self.min_margin, self.max_margin, shape=(sum(self.n_f_players)*3, ))#, dtype=int)

		self.action_space = gym.spaces.Box(0, 1, shape=(sum(self.n_f_players), ))#, dtype=float)

		self.lims_action = self.n_f_players[:-1]
		self.lims_action = np.cumsum(self.lims_action)
		self.lims_observation = [3 * lim for lim in self.lims_action]

		n = self.action_space.shape[0]
		lims = list(self.lims_action)
		lims = [0] + lims
		lims.append(n)

		self.action_selector = {}
		for player in self.players:
			lim1 = lims[self.players_id[player]]
			lim2 = lims[self.players_id[player]+1]

			self.action_selector[player] = (lim1, lim2)

	def apply_action(self, action):
		for i in range(len(self.n_f_players)):
			player = self.players[i]
			if i==0:
				nfp1 = 0
			else:
				nfp1 = sum(self.n_f_players[:i])
				
			if i==len(self.n_f_players)-1:
				nfp2 = sum(self.n_f_players)#len(self.n_f_players)
			else:
				nfp2 = nfp1+self.n_f_players[i]

			action_p = action[nfp1:nfp2].reshape(nfp2-nfp1, 1)[:, 0]

			for j, ac in enumerate(action_p):
				# for each flight
				name = self.flight_per_company[player][j]

				new_cost = self.flights[name].slope_declared
				new_jump = self.flights[name].jump1_declared

				scaled_action_margin = self.func_margin(ac)
				self.flights[name].set_declared_charac(margin=scaled_action_margin, cost=new_cost, jump=new_jump)

				# Update cost vect for each flight
				self.flights[name].compute_cost_vect(self.slots, declared=True)


class MaxAgentJump(Agent):
	def __init__(self, kind='max', gym_env=None, name='', as_player=None):
		super().__init__(kind=kind, name=name)
		if as_player is None:
			as_player = name

		self.high = gym_env.action_space.high
		self.action_selector = gym_env.build_action_selector_for(as_player)

	def action(self, observation):
		#action = self.high[self.lims[0]:self.lims[1]]
		action = self.action_selector(self.high)
		#print (self, action)
		return action


class HonestAgentJump(Agent):
	def __init__(self, kind='honest', gym_env=None, name='', as_player=None):
		super().__init__(kind=kind, name=name)
		if as_player is None:
			as_player = name

		#self.lims = lims
		#self.n_f = lims[1]-lims[0]
		self.n_f = gym_env.n_f_players_dict[as_player]

	def action(self, observation):
		# extract the (real) jumps and pass them down as action
		# TODO: works only for jumps...
		state = np.array(observation).reshape(self.n_f, 3)
		action = state[:, 1]
		#print (self, action)
		return action


class RandomAgentJump(Agent):
	def __init__(self, kind='random', gym_env=None, name='', as_player=None):
		super().__init__(kind=kind, name=name)
		if as_player is None:
			as_player = name

		self.draw_func = gym_env.action_space.sample
		#self.lims = lims
		self.action_selector = gym_env.build_action_selector_for(as_player)

	def action(self, observation):
		#action = self.draw_func()[self.lims[0]:self.lims[1]]
		action = self.action_selector(self.draw_func())
		#print (self, action)
		return action


class ContMGameJump(ContMGame):
	agent_collection = {'max':MaxAgentJump,
						'honest':HonestAgentJump,
						'random':RandomAgentJump}

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		if self.normed_state:
			self.observation_space =  gym.spaces.Box(0., 1., shape=(sum(self.n_f_players)*3, ))#, dtype=int)
		else:
			self.observation_space =  gym.spaces.Box(self.min_jump, self.max_jump, shape=(sum(self.n_f_players)*3, ))#, dtype=int)

		self.action_space = gym.spaces.Box(0, 1, shape=(sum(self.n_f_players), ))#, dtype=float)

		self.lims_action = self.n_f_players[:-1]
		self.lims_action = np.cumsum(self.lims_action)
		self.lims_observation = [3 * lim for lim in self.lims_action]

		n = self.action_space.shape[0]
		lims = list(self.lims_action)
		lims = [0] + lims
		lims.append(n)

		self.action_selector = {}
		for player in self.players:
			lim1 = lims[self.players_id[player]]
			lim2 = lims[self.players_id[player]+1]

			self.action_selector[player] = (lim1, lim2)

	def apply_action(self, action):
		for i in range(len(self.n_f_players)):
			player = self.players[i]
			if i==0:
				nfp1 = 0
			else:
				nfp1 = sum(self.n_f_players[:i])
				
			if i==len(self.n_f_players)-1:
				nfp2 = sum(self.n_f_players)
			else:
				nfp2 = nfp1+self.n_f_players[i]

			action_p = action[nfp1:nfp2].reshape(nfp2-nfp1, 1)[:, 0]
			
			for j, ac in enumerate(action_p):
				# for each flight
				name = self.flight_per_company[player][j]

				new_slope = self.flights[name].slope_declared
				new_margin = self.flights[name].margin1_declared

				scaled_action_jump = self.func_jump(ac)

				self.flights[name].set_declared_charac(margin=new_margin,
														slope=new_slope,
														jump=scaled_action_jump)

				# Update cost vect for each flight
				self.flights[name].compute_cost_vect(self.slots, declared=True)
				# print ('delayCostVect:', name, self.flights[name].delayCostVect)
				# print ()
				


