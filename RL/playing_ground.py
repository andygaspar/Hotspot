from collections import OrderedDict
import numpy as np
import tensorflow as tf
from tf_agents.policies.policy_saver import PolicySaver

from game_trainer import TFAgentTrainer

def compute_cost_diff(row, player='A'):
	if type(row)==np.float64:
		row = float(row)
	cost_diff = row['cost_per_c'][player] - row['best_cost_per_c'][player]

	return cost_diff

def compute_cost_diff_ratio(row, player='A'):
	if type(row)==np.float64:
		row = float(row)
	cost_diff = (row['cost_per_c'][player] - row['best_cost_per_c'][player])/row['best_cost_per_c'][player]
	return cost_diff

def compute_cost_diff_others(row, player='A'):
	if type(row)==np.float64:
		row = float(row)
	c = np.array([cc for air, cc in row['cost_per_c'].items() if air!=player]).sum()
	cb = np.array([cc for air, cc in row['best_cost_per_c'].items() if air!=player]).sum()
	cost_diff = c-cb
	return cost_diff

def compute_cost_diff_all(row):
	if type(row)==np.float64:
		row = float(row)
	c = np.array([cc for air, cc in row['cost_per_c'].items()]).sum()
	cb = np.array([cc for air, cc in row['best_cost_per_c'].items()]).sum()
	cost_diff = c-cb
	return cost_diff

def compute_cost_diff_ratio_others(row, player='A'):
	if type(row)==np.float64:
		row = float(row)
	c = np.array([cc for air, cc in row['cost_per_c'].items() if air!=player]).sum()
	cb = np.array([cc for air, cc in row['best_cost_per_c'].items() if air!=player]).sum()
	cost_diff = (c-cb)/cb
	return cost_diff

def compute_cost_diff_ratio_all(row):
	if type(row)==np.float64:
		row = float(row)
	c = np.array([cc for air, cc in row['cost_per_c'].items()]).sum()
	cb = np.array([cc for air, cc in row['best_cost_per_c'].items()]).sum()
	cost_diff = (c-cb)/cb
	return cost_diff

class Agent:
	def __init__(self, kind='TF'):
		self.kind = kind


class TFAgentWrap(Agent):
	"""
	Just a wrapper to use easily with training and evaluation.
	"""
	def __init__(self, kind='TF'):
		self.kind = kind

	def wrap_tf_agent(self, tf_agent):
		self.policy = tf_agent.policy

	def load(self, file_name):
		saved_policy = tf.compat.v2.saved_model.load(file_name)
		self.policy = saved_policy

	def save(self, file_name):
		if file_name is None:
			file_name = self.agent_type + '_' + self.game + '_' + str(self.n_f_player) + 'a' + str(self.n_f) + '_policy'

		PolicySaver(self.policy).save(file_name)

	def action(self, observation):
		# TODO: observation wrapper
		action = self.policy.action(time_step)
		action = action.action.numpy()[0]

		return action

	def action_tf(self, time_step):
		# Unwrapped action results
		action = self.policy.action(time_step)

		return action

def merge_actions(actions):
	l = []
	for action in actions:
		l += list(action)

	return np.array(l)

def split_observation(obs):
	n = len(obs)
	lims = list(lims)
	lims = [0] + lims
	lims.append(n)    

	obss = []
	for i in range(len(lims)-1):
		lim = lims[i]
		lim2 = lims[i+1]
		#print (lim, lim2)
		obss.append(obs[lim:lim2])

	return obss

class MaxAgent(agent):
	pass


class HonestAgent(agent):
	pass


class RandomAgent(agent):
	pass


class PlayingGround:
	"""
	Can use single or multi games.
	"""
	def __init__(self, game='jump', multi=False, **kwargs_game):
		self.multi = multi
		self.game = game

		if game=='jump':
			self.gym_env = ContGameJump(**kwargs_game)
		elif game=='margin':
			self.gym_env = ContGameMargin(**kwargs_game)
		elif game=='jump_margin':
			self.gym_env = ContGame(**kwargs_game)
		else:
			raise Exception('Unknown game:', game)
		pass

	def build_tf_trainer(self):
		if not self.multi:
			self.trainer = TFAgentTrainer()
		else:
			self.trainer = TFAgentMultiTrainer()

		self.trainer.build_game_wrapper(self.gym_env)

	def build_tf_agents(self, **kwargs):
		if not self.multi:
			self.trainer.build_agents(**kwargs)
		else:
			self.trainer.build_agent(**kwargs)

	def train(self, **kwargs):
		self.train(**kwargs)

	def wrap_tf_agents_from_trainer(self):
		self.agents = OrderedDict()
		for i, tf_agent in enumerate(self.trainer.tf_agents):
			player = self.gym_env.players[i]
			self.agents[players] = TFAgentWrap(tf_agent)

	def set_agents(self, agents):
		"""
		agents should be a dictionary
		"""
		try:
			assert len(agents)==len(self.gym_env.players)
		except AssertionError:
			raise Exception("You gave several agents but the game is a single one")

		self.agents = agents

	def set_agent(self, agent=None, name=None):
		"""
		To set different agents in different positions
		"""

		self.agents[name] = agent

	def load_tf_agent(self, load_into=None, file_name=None):
		agent = TFAgentWrap()
		agent.load(file_name=file_name)

		if not load_into is None:
			self.set_agent(agent=agent, name=load_into)
		else:
			return agent

	def game_summary(self):
		self.gym_env.game_summary()

	def print_agents(self):
		print ('Agents in simulation:')
		for name, agent in self.agents:
			print ('Agent', name, 'of type', agent.kind)

	def evaluation(self, n_evaluations=100, show_results=True, file_name=None):
		"""Probably obsolete"""
		rewards_eval = []

		for i in range(n_evaluations):
			obs = self.gym_env.reset()
			action = 
			#time_step = self.collect_env.reset()
			#action = self.tf_agent.policy.action(time_step)
			#stuff = self.train_env.step(action)
			rewards_eval.append(stuff.reward.numpy()[0])
		rewards_eval = np.array(rewards_eval)

		if show_results:
			print ('Positive rewards:', 100 * len(rewards_eval[rewards_eval>100])/len(rewards_eval), '%')
			print ('Neutral rewards:', 100 * len(rewards_eval[rewards_eval==100])/len(rewards_eval), '%')
			print ('Negative rewards:', 100 * len(rewards_eval[rewards_eval<100])/len(rewards_eval), '%')
			print ('Average reward:', rewards_eval.mean())
			print ('Reward std:', rewards_eval.std())
			plt.hist(rewards_eval, bins=20)

		# if not file_name is None:
		# 	rewards_eval.to_csv(file_name)

		return rewards_eval

	def compare_with_optimiser(self, n_iter=100):
		# TODO: adapt for sevreal agents
		rewards_agent = []
		rewards_best = []
		for i in range(n_iter):
			if i%10==0:
				print ('i=', i)
			time_step = self.collect_env.reset()

			# Compute best solution on allocation
			f = function_builder(self.collect_env)
			sol = differential_evolution(f, [(0, 100), (0, 100), (0, 100), (0, 100)])
			rewards_best.append(100.-sol['fun'])

			# See reward from action of agent
			action = self.tf_agent.policy.action(time_step)
			stuff = self.collect_env.step(action)
			rewards_agent.append(stuff.reward.numpy()[0])

		rewards_agent = np.array(rewards_agent)
		rewards_best = np.array(rewards_best)

		print ('Agent efficiency:', 100*(1-len(rewards_agent[rewards_best>rewards_agent])/len(rewards_agent)), '%')
		print ('Missed opportunities:', 100*len(rewards_agent[(rewards_best>rewards_agent)&(rewards_agent==100)])/len(rewards_agent[rewards_agent==100]), '%')
		print ('Regrets:', 100*len(rewards_agent[(rewards_best>rewards_agent)&(rewards_best==100)])/len(rewards_agent[rewards_best==100]), '%')
		print ('Average reward (agent/best):', rewards_agent.mean(), rewards_best.mean())

		fig, ax = plt.subplots()
		x = list(range(len(rewards_agent)))
		ax.bar(x, rewards_agent, label='Agent', alpha=0.4)
		ax.bar(x, rewards_best, label='Best', alpha=0.4)
		ax.legend()
		ax.set_xlabel('Reward')
		ax.set_ylabel('Reward')
		#savefig('comparison_best_agent_reward.png')

		return rewards_agent, rewards_best

	def compare_airlines(self, n_iter=100, show_results=True, file_name=None):
		results = {}
		# builder = get_results_builder_multi(self.collect_env)
		# for i in range(n_iter):
		# 	if n_iter>=10 and i%int(n_iter/10)==0:
		# 		print ('i=', i)
		# 	time_step = self.collect_env.reset()
		# 	rew = tf.constant([[100.]*len(self.tf_agents)], dtype=tf.float32)
		# 	step_type = tf.constant([[1]*len(self.tf_agents)], dtype=tf.int32)
		# 	time_step = TimeStep(step_type=step_type,
		# 						  reward=rew,
		# 						  discount=time_step.discount,
		# 						  observation=time_step.observation)

		# 	# ADD lims
		# 	#  Split time step
		# 	sts = split_time_step(time_step, lims=self.lims_triple)

		# 	# Get action based on partial observation
		# 	asts = []
		# 	for j, ts in enumerate(sts):
		# 		asts.append(self.tf_agents[j].policy.action(ts))

		# 	# Merge actions
		# 	action_step = merge_action_steps(asts)

		# 	# # Get next state
		# 	# next_time_step = environment.step(action_step.action)

		# 	#action = self.tf_agent.policy.action(time_step)
		# 	results[i] = builder(action_step.action.numpy()[0])
		# 	#stuff = self.collect_env.step(action)

		results = {}
		for i in range(n_iter):
			if n_iter>=10 and i%int(n_iter/10)==0:
		 		print ('i=', i)

		 	res = self.observe_one_step()

			results[i] = res

		def build_extract_col(i):
			def extract_col(r):
				return r[i]

			return extract_col

		df = pd.DataFrame(results, index=['rewards', 'reward_tot', 'cost_tot', 'cost_per_c', 'allocation', 'reward_fake', 'transferred_cost', 'best_cost_per_c']).T

		players = self.collect_env.pyenv.envs[0].gym.players
		for i in range(len(self.tf_agents)):
			extrator_col = build_extract_col(i)
			df['reward {}'.format(players[i])] = df['rewards'].apply(extrator_col)

		#print (df)
		to_concat = [df]
		for i, player in enumerate(players):
			compute_cost_diff_p = lambda x:compute_cost_diff(x, player=player)
			compute_cost_diff_ratio_p = lambda x:compute_cost_diff_ratio(x, player=player)
			compute_cost_diff_others_p = lambda x:compute_cost_diff_others(x, player=player)
			compute_cost_diff_ratio_others_p = lambda x:compute_cost_diff_ratio_others(x, player=player)

			to_apply = [compute_cost_diff_p,
						compute_cost_diff_ratio_p,
						compute_cost_diff_others_p,
						compute_cost_diff_ratio_others_p,
						compute_cost_diff_all,
						compute_cost_diff_ratio_all]
			dg = df.T.apply(to_apply).T

			dg.columns = ['compute_cost_diff_{}'.format(player),
						'compute_cost_diff_ratio_{}'.format(player),
						'compute_cost_diff_others_{}'.format(player),
						'compute_cost_diff_ratio_others_{}'.format(player),
						'compute_cost_diff_all',
						'compute_cost_diff_ratio_all']

			to_concat.append(dg)
		
		df = pd.concat(to_concat, axis=1)

		# Unravel cost per company
		dg = pd.DataFrame(list(df['cost_per_c']))
		dg.rename(columns={k:'Cost {}'.format(k) for k in dg.columns}, inplace=True)
		dg2 = pd.DataFrame(list(df['best_cost_per_c']))
		dg2.rename(columns={k:'Best cost {}'.format(k) for k in dg2.columns}, inplace=True)
		df = pd.concat([df, dg, dg2], axis=1)
		df.drop(columns=['rewards', 'reward_fake', 'cost_per_c', 'best_cost_per_c', 'allocation'], inplace=True)

		if show_results:
			print ('Average results:')
			print (df.mean())

		if not file_name is None:
			df.to_csv(file_name)

		return df

	def observe_one_step(self, reset=True):
		obs = self.gym_env.reset()
	 	
	 	# TODO: split observation
	 	obss = split_observation(obs, lims=self.lims_triple)

	 	actions = [agent.action(obss[i]) for i, (name, agent) in enumerate(self.agents.items())]

	 	# TODO: merge action
	 	action_merged = merge_actions(actions)

	 	# TODO: put all stuff in info and use step method instead
	 	self.gym_env.apply_action(action)

		res = self.gym_env.compute_reward()

		return res

	def do_plots(self, file_name=None, instantaneous=False):
		try:
			dir_name = '/'.join(file_name.split('/')[:-1])
			os.makedirs(dir_name)
		except OSError as e:
			#if e.errno != errno.EEXIST:
			pass
		except:
			raise

		nc = nice_colors[:]
		c = nice_colors[1]
		nc[1] = nice_colors[2]
		nc[2] = c

		fig, ax = plt.subplots()
		if instantaneous:
			for i, rew in enumerate(self.rewards):
				ax.plot(rew, lw=0.5, label='Instantaneous reward {}'.format(i), alpha=0.6, c=nc[i])

		N = 50	
		y = pd.DataFrame(self.rewards).T.rolling(window=N).mean().iloc[N-1:].values.T
		for i in range(len(y)):
			yy = y[i]
			ax.plot(yy, label='Average 50 runs {}'.format(i), c=nc[i])
		N = 500
		y = pd.DataFrame(self.rewards).T.rolling(window=N).mean().iloc[N-1:].values.T
		for i in range(len(y)):
			yy = y[i]
			ax.plot(yy, label='Average 500 runs {}'.format(i), c=nc[i])
		ax.set_ylabel('Reward')
		ax.set_xlabel('Number of Iterations')
		ax.legend()

		if not file_name is None:
			plt.savefig(file_name, bbox_inches='tight')

	def show_examples(self, n_ex=3):
		#sg = WrapperSingleGame(self.collect_env)
		#gym_env = self.collect_env.pyenv.envs[0].gym
	
		for i in range(n_ex):
			# time_step = self.collect_env.reset()

			# action = self.tf_agent.policy.action(time_step)

			# action = action.action.numpy()[0]

			# gym_env = self.collect_env.pyenv.envs[0].gym
			# #results[i] = builder(action.action.numpy()[0])
			# gym_env.apply_action(action)

			reward, reward_tot, cost_tot, cost_per_c, allocation, reward_fake, cost_true_per_c, transferred_cost = self.observe_one_step()

			print ('initial state:')
			print (gym_env.df_sch_init)
			print ()

			print ('declared state:')
			print (gym_env.df_sch)
			print ()

			print ('Base allocation:')
			print_allocation(gym_env.base_allocation)
			print ()

			print ('Costs in base allocation:')
			print (gym_env.base_cost_per_c)
			print ()

			print ('Best allocation:')
			print_allocation(gym_env.best_allocation)
			print ()

			print ('Costs in best allocation:')
			print (gym_env.best_cost_per_c)
			print ()

			print ('Final allocation:')
			print_allocation(allocation)
			print ()

			print ('Cost of final allocation:')
			print (cost_per_c)
			print ()

			print ('Reward:', reward)
			print ()
			print ()
