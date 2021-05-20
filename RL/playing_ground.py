from collections import OrderedDict
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# from tf_agents.policies.policy_saver import PolicySaver
# from tf_agents.trajectories.time_step import TimeStep

from Hotspot.libs.other_tools import print_allocation
from Hotspot.uow_tool_belt.general_tools import nice_colors

#from Hotspot.RL.continuous_game import ContGame, ContGameJump, ContGameMargin
from Hotspot.RL.mcontinuous_game import ContMGame, ContMGameJump, ContMGameMargin

from Hotspot.RL.game_trainer import TFAgentWrap, TFAgentTrainer, SBAgentTrainer, SBAgentWarp

#from Hotspot.RL.agents import RandomAgent, MaxAgent, HonestAgent

#  Functions used to compute metrics
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

def merge_actions(actions):
	l = []
	for action in actions:
		#print (action)
		if len(np.array(action).shape)>1:
			action = np.array(action).flatten()
		#print (action)
		l += list(action)

	return np.array(l)

def split_observation(obs, lims=[]):
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


class PlayingGround:
	"""
	Can use single or multi games.
	"""
	def __init__(self, game='jump', kind='multi', **kwargs_game):
		#self.multi = kind !='single' # TODO improve that
		self.game = game

		if game=='jump':
			self.gym_env = ContMGameJump(**kwargs_game)
		elif game=='margin':
			self.gym_env = ContMGameMargin(**kwargs_game)
		elif game=='jump_margin':
			self.gym_env = ContMGame(**kwargs_game)
		else:
			raise Exception('Unknown game:', game)

		self.agents = OrderedDict([(player, None) for player in self.gym_env.players])
		self.agents_ids = {name:i for i, name in enumerate(self.agents.keys())}

	def build_tf_trainer(self):
		self.trainer = TFAgentTrainer()

		self.trainer.build_game_wrapper(self.gym_env)

	def build_sb_trainer(self):
		self.trainer = SBAgentTrainer()

		self.trainer.set_env(self.gym_env)

	def build_tf_agents(self, **kwargs):
		self.trainer.build_agents(gym_env=self.gym_env, **kwargs)

	def build_sb_agents(self, players=None, **kwargs):
		if players is None:
			players = self.gym_env.players

		self.trainer.build_agents(players=players, **kwargs)

	def train(self, **kwargs):
		# TODO: support arguments for prepare_buffers for TF
		self.trainer.train_agents(**kwargs)

	def wrap_tf_agents_from_trainer(self):
		self.agents = OrderedDict()
		for i, tf_agent in enumerate(self.trainer.tf_agents):
			player = self.gym_env.players[i]
			self.agents[player] = TFAgentWrap(tf_agent)

	def wrap_sb_agents_from_trainer(self):
		self.agents = OrderedDict()
		for i, model in enumerate(self.trainer.models):
			player = self.gym_env.players[i]
			self.agents[player] = SBAgentWarp(model)

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

	def save_all_agents(self, file_names=[]):
		for i, (name, agent) in enumerate(self.agents.items()):
			agent.save(file_names[i])

	def load_agent_from_collection(self, kind='', load_into=None, **kwargs):
		self.agents[load_into] = self.gym_env.build_agent_from_collection(kind=kind, name=load_into, **kwargs)

	# def load_random_agent(self, load_into=None):
	# 	n = self.gym_env.action_space.shape[0]
	# 	lims = list(self.gym_env.lims_simple)
	# 	lims = [0] + lims
	# 	lims.append(n)

	# 	lim1 = lims[self.agents_ids[load_into]]
	# 	lim2 = lims[self.agents_ids[load_into]+1]
		
	# 	# TODO: random, max, and honest agents should be created
	# 	# by the game itself (gym_env)
	# 	self.agents[load_into] = RandomAgent(gym_env=self.gym_env,
	# 										lims=(lim1, lim2),
	# 										name=load_into)

	# def load_max_agent(self, load_into=None):
	# 	# n = self.gym_env.action_space.shape[0]
	# 	# lims = list(self.gym_env.lims_simple)
	# 	# lims = [0] + lims
	# 	# lims.append(n)

	# 	# lim1 = lims[self.agents_ids[load_into]]
	# 	# lim2 = lims[self.agents_ids[load_into]+1]
		
	# 	self.agents[load_into] = MaxAgent(gym_env=self.gym_env,
	# 										lims=(lim1, lim2),
	# 										name=load_into)

	# def load_honest_agent(self, load_into=None):
	# 	n = self.gym_env.action_space.shape[0]
	# 	lims = list(self.gym_env.lims_simple)
	# 	lims = [0] + lims
	# 	lims.append(n)

	# 	lim1 = lims[self.agents_ids[load_into]]
	# 	lim2 = lims[self.agents_ids[load_into]+1]
		
	# 	self.agents[load_into] = HonestAgent(gym_env=self.gym_env,
	# 										lims=(lim1, lim2),
	# 										name=load_into)

	def game_summary(self):
		self.gym_env.game_summary()

		print ('Agents loaded:')
		for name, agent in self.agents.items():
			print (agent)

	def print_agents(self):
		print ('Agents in simulation:')
		for name, agent in self.agents:
			print ('Agent', name, 'of type', agent.kind)

	def evaluation(self, n_evaluations=100, show_results=True, file_name=None):
		"""Probably obsolete"""
		rewards_eval = []

		for i in range(n_evaluations):
			obs = self.gym_env.reset()
			action = None
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

	def compare_airlines(self, n_iter=100, show_results=True, file_name=None, 
		mets=['rewards', 'rewards_tot', 'cost_tot', 'cost_per_c', 'allocation', 'rewards_fake',
		'best_cost_per_c', 'transferred_cost']):

		def build_extract_col(i):
			def extract_col(r):
				return r[i]

			return extract_col

		results = {}
		for i in range(n_iter):
			if n_iter>=10 and i%int(n_iter/10)==0:
				print ('i=', i)

			res = self.observe_one_step()

			results[i] = [res[met] for met in mets]

		df = pd.DataFrame(results, index=mets).T

		for i, (name, agent) in enumerate(self.agents.items()):
			extrator_col = build_extract_col(i)
			df['reward {}'.format(name)] = df['rewards'].apply(extrator_col)

		to_concat = [df]
		for i, (name, agent) in enumerate(self.agents.items()):
			compute_cost_diff_p = lambda x:compute_cost_diff(x, player=name)
			compute_cost_diff_ratio_p = lambda x:compute_cost_diff_ratio(x, player=name)
			compute_cost_diff_others_p = lambda x:compute_cost_diff_others(x, player=name)
			compute_cost_diff_ratio_others_p = lambda x:compute_cost_diff_ratio_others(x, player=name)

			to_apply = [compute_cost_diff_p,
						compute_cost_diff_ratio_p,
						compute_cost_diff_others_p,
						compute_cost_diff_ratio_others_p,
						compute_cost_diff_all,
						compute_cost_diff_ratio_all]
			dg = df.T.apply(to_apply).T

			dg.columns = ['compute_cost_diff_{}'.format(name),
						'compute_cost_diff_ratio_{}'.format(name),
						'compute_cost_diff_others_{}'.format(name),
						'compute_cost_diff_ratio_others_{}'.format(name),
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
		df.drop(columns=['rewards', 'rewards_fake', 'cost_per_c', 'best_cost_per_c', 'allocation'], inplace=True)

		if show_results:
			print ('Average results:')
			print (df.mean())

		if not file_name is None:
			df.to_csv(file_name)

		return df

	def observe_one_step(self, reset=True):
		obs = self.gym_env.reset()

		obss = split_observation(obs, lims=self.gym_env.lims_observation)

		#print ('obss=', obss)

		actions = [agent.action(obss[i]) for i, (name, agent) in enumerate(self.agents.items())]

		#print ('actions=', actions)

		action_merged = merge_actions(actions)

		#print('action_merged=', action_merged)

		state, rewards, _, information = self.gym_env.step(action_merged)

		return information

	def do_training_plots(self, file_name=None, instantaneous=False):
		try:
			dir_name = '/'.join(file_name.split('/')[:-1])
			os.makedirs(dir_name)
		except OSError as e:
			#if e.errno != errno.EEXIST:
			pass
		except AttributeError as e:
			pass
		except:
			raise

		nc = nice_colors[:]
		c = nice_colors[1]
		nc[1] = nice_colors[2]
		nc[2] = c

		fig, ax = plt.subplots()
		if instantaneous:
			for i, rew in enumerate(self.trainer.rewards):
				ax.plot(rew, lw=0.5, label='Instantaneous reward {}'.format(i), alpha=0.6, c=nc[i])

		N = 50
		y = pd.DataFrame(self.trainer.rewards).T.rolling(window=N).mean().iloc[N-1:].values.T
		for i in range(len(y)):
			yy = y[i]
			ax.plot(yy, label='Average 50 runs {}'.format(i), c=nc[i])
		N = 500
		y = pd.DataFrame(self.trainer.rewards).T.rolling(window=N).mean().iloc[N-1:].values.T
		for i in range(len(y)):
			yy = y[i]
			ax.plot(yy, label='Average 500 runs {}'.format(i), c=nc[i])
		ax.set_ylabel('Reward')
		ax.set_xlabel('Number of Iterations')
		ax.legend()

		if not file_name is None:
			plt.savefig(file_name, bbox_inches='tight')

	def show_examples(self, n_ex=3):
		for i in range(n_ex):
			print ('Example', i)
			print ()

			information = self.observe_one_step()

			print ('initial state:')
			print (information['df_sch_init'])
			print ()

			print ('declared state:')
			print (information['df_sch'])
			print ()

			print ('Base allocation:')
			print_allocation(information['base_allocation'])
			print ()

			print ('Costs in base allocation:')
			print (information['base_cost_per_c'])
			print ()

			print ('Best allocation:')
			print_allocation(information['best_allocation'])
			print ()

			print ('Costs in best allocation:')
			print (information['best_cost_per_c'])
			print ()

			print ('Final allocation:')
			print_allocation(information['allocation'])
			print ()

			print ('Cost of final allocation:')
			print (information['cost_per_c'])
			print ()

			print ('Rewards:', information['rewards'])
			print ()
			print ()
