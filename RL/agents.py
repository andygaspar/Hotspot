import numpy as np

# class AgentBruteSingle:
# 	def __init__(self, initial_margin=0., dm=2., initial_jump=0., dj=5.):
# 		self.timer = 0
# 		self.initial_margin = initial_margin
# 		self.current_margin = initial_margin
		
# 		self.initial_jump = initial_jump
# 		self.current_jump = initial_jump
		
# 		self.dm = dm
# 		self.dj = dj
		
# 		self.found_zero = False
# 		self.found_max = False
		
# 		self.found_zero2 = False
	
# 	def action(self, state):
# 		if not self.found_max:
# 			if self.current_margin > 0 and not self.found_zero:
# 				self.current_margin = max(0, self.current_margin-self.dm)
# 				return [0, 1]
# 			else:
# 				self.found_zero = True
# 				self.current_margin += self.dm
# 				self.found_max = self.current_margin>30
# 				return [2, 1]
# 		else:
# 			self.found_zero = False
# 			self.found_max = False
# 			if self.current_jump > 0 and not self.found_zero2:
# 				self.current_jump = max(0, self.current_jump-self.dj)
# 				return [1, 0]
# 			else:
# 				self.found_zero2 = True
# 				self.current_jump += self.dj
# 				#self.found_max = self.current_jump>30
# 				return [1, 2]
		
# class AgentRandomMulti:
# 	def __init__(self, initial_margins=None, initial_jumps=None,
# 		dm=2., dj=5., n_f=5):
		
# 		self.timer = 0
# 		self.initial_margins = initial_margins
# 		self.current_margins = initial_margins
		
# 		self.initial_jumps = initial_jumps
# 		self.current_jumps = initial_jumps
		
# 		self.dm = dm
# 		self.dj = dj
		
# 		self.found_zero = False
# 		self.found_max = False
		
# 		self.found_zero2 = False
		
# 		self.n_f = n_f
	
# 	def action(self, state):
# 		action = np.random.choice([0, 1, 2], size=(self.n_f, 2))
# 		#action_jump = np.random.choice([0, 1, 2], size=(n_f, 2))
		
# 		for i, ac in enumerate(action):
# 			# for each flight
# 			#name = self.flight_per_company[self.player][i]
			
# 			if ac[0]==0:
# 				dm = -self.dm
# 			elif ac[0]==1:
# 				dm = 0
# 			else:
# 				dm = self.dm

# 			if ac[1]==0:
# 				dj = -self.dj
# 			elif ac[1]==1:
# 				dj = 0
# 			else:
# 				dj = self.dj

# 			self.current_margins[i] = max(0, self.current_margins[i] + dm)
# 			#new_cost = self.flights[name].cost_declared# + dc
# 			self.current_jumps[i] = max(1, self.current_jumps[i] + dj)
			
# 		return action


class Agent:
	def __init__(self, kind='generic', name='generic'):
		self.kind = kind
		self.name = name

	def load(self, file_name):
		pass

	def save(self, file_name):
		pass

	def action(self, observation):
		pass

	def __repr__(self):
		return 'Agent {} of type {}'.format(self.name, self.kind)


class MaxAgent(Agent):
	def __init__(self, kind='max', gym_env=None, lims=[], name=''):
		super().__init__(kind=kind, name=name)
		self.high = gym_env.action_space.high
		self.lims = lims

	def action(self, observation):
		action = self.high[self.lims[0]:self.lims[1]]
		#print (self, action)
		return action


class HonestAgent(Agent):
	def __init__(self, kind='honest', gym_env=None, lims=[], name=''):
		super().__init__(kind=kind, name=name)
		self.lims = lims
		self.n_f = lims[1]-lims[0]

	def action(self, observation):
		# extract the (real) jumps and pass them down as action
		# TODO: works only for jumps...
		state = np.array(observation).reshape(self.n_f, 3)
		action = state[:, 1]
		#print (self, action)
		return action


class RandomAgent(Agent):
	def __init__(self, kind='random', gym_env=None, lims=[], name=''):
		super().__init__(kind=kind, name=name)
		self.draw_func = gym_env.action_space.sample
		self.lims = lims

	def action(self, observation):
		action = self.draw_func()[self.lims[0]:self.lims[1]]
		#print (self, action)
		return action