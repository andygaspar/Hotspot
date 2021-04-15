import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution

import tensorflow as tf
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.environments import suite_gym
from tf_agents.utils import common
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.drivers import dynamic_step_driver
from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.trajectories import trajectory

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
#from tf_agents.environments import suite_pybullet
#from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
#from tf_agents.policies import greedy_policy
#from tf_agents.policies import py_tf_eager_policy
#from tf_agents.policies import random_py_policy
#from tf_agents.replay_buffers import reverb_replay_buffer
#from tf_agents.replay_buffers import reverb_utils
#from tf_agents.train import actor
#from tf_agents.train import learner
#from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
#from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.specs import BoundedTensorSpec

#from RL.multiple_game import MultiStochGameJumpFlatSpaces, MultiStochGameMarginFlatSpaces
from RL.multiple_game import MultiStochGameJumpFlatSpaces, MultiStochGameMarginFlatSpaces
from RL.continuous_game2 import ContGame, ContGameJump, ContGameMargin
from RL.mcontinuous_game import ContMGame, ContMGameJump, ContMGameMargin

from libs.tools import print_allocation
from libs.general_tools3 import nice_colors

# For videos
# import imageio
# import base64
# import IPython

def collect_step(environment, policy, buf):
	time_step = environment.current_time_step()
	action_step = policy.action(time_step)
	#print ('action_step', action_step)
	next_time_step = environment.step(action_step.action)
	#print ('next_time_step=', next_time_step)
	traj = trajectory.from_transition(time_step, action_step, next_time_step)

	# Add trajectory to the replay buf
	buf.add_batch(traj)

def collect_data(env, policy, buf, steps):
	for _ in range(steps):
		collect_step(env, policy, buf)

def collect_step_multi(environment, policies, bufs, lims=[]):
	time_step = environment.current_time_step()
	# rebuild time_step with good reward and step_type shape to avoid other issues
	rew = tf.constant([[100.]*len(bufs)], dtype=tf.float32)
	step_type = tf.constant([[1]*len(bufs)], dtype=tf.int32)
	time_step = TimeStep(step_type=step_type,
						  reward=rew,
						  discount=time_step.discount,
						  observation=time_step.observation)
	# print ('time_step:', time_step)
	# print ()

	# Split tims_step
	sts = split_time_step(time_step, lims=lims)
	# print ('Splitted time step', sts)
	# print ()

	# Get action based on partial observation
	asts = []
	for i, ts in enumerate(sts):
		# print (i, ts)
		asts.append(policies[i].action(ts))

	# Merge actions
	# print ('Action list', asts)
	# print ()
	action_step = merge_action_steps(asts)
	# print ('Merged action_step', action_step)
	# print ()

	# Get next state
	next_time_step = environment.step(action_step.action)

	# Split the new observation
	# print ('Next time step:', next_time_step)
	# print ()
	sts_new = split_time_step(next_time_step, lims=lims)
	# print ('Splitted next time steps:', sts_new)
	# print ()

	# Build trajectories
	trajs = [trajectory.from_transition(sts[i], asts[i], sts_new[i]) for i in range(len(sts))]

	# Add trajectory to the replay buffer
	for i, buf in enumerate(bufs):
		buf.add_batch(trajs[i])

	#print ()

def collect_data_multi(env, policies, bufs, steps, lims=[]):
	for _ in range(steps):
		collect_step_multi(env, policies, bufs, lims=lims)

def split_time_step(ts, lims=[]):
	#print ('TS', ts)
	t = ts.observation
	# print ('t.shape', t.shape)
	# print ()
	n = t.shape[1]
	
	lims = list(lims)
	lims = [0] + lims
	lims.append(n)    
	
	#print ('lims', lims)
	sts = []
	for i in range(len(lims)-1):
		lim = lims[i]
		lim2 = lims[i+1]
		#print ('lim, lim2', lim, lim2)
		# print ('BOUUUUHH, ts.step_type', ts.step_type)
		# print ('BOUUUUHH, ts.step_type.shape', ts.step_type.shape)
		rew = ts.reward[:, i]#tf.Tensor([[]], shape=(1, 2), dtype=float32)
		step_type = ts.step_type[:, i]
		# print ('BOUUUUHH, rew', rew)
		sts.append(TimeStep(step_type=step_type,
							  reward=rew,
							  discount=ts.discount,
							  observation=t[:, lim:lim2]))
		
	return sts

def merge_time_steps(tss):
	ts1 = tss[0]
	
	tt = [ts.observation for ts in tss]
	
	ts = TimeStep(step_type=ts1.step_type,
				  reward=ts1.reward,
				  discount=ts1.discount,
				  observation=tf.concat(tt, axis=1))
	
	return ts

def merge_action_steps(pss):
	ps1 = pss[0]

	tt = [ps.action for ps in pss]

	ps = PolicyStep(state=ps1.state,
				  info=ps1.info,
				  action=tf.concat(tt, axis=1))

	return ps

def split_bounded_tensor_spec(bts, lims=[]):
	"""
	Splits only one dimensional tensors
	"""

	#t = ts.observation
	n = bts.shape[0]
	
	#print (n)

	lims = list(lims)
	lims = [0] + lims
	lims.append(n)    

	btss = []
	for i in range(len(lims)-1):
		lim = lims[i]
		lim2 = lims[i+1]
		#print (lim, lim2)
		btss.append(BoundedTensorSpec(dtype=bts.dtype,
							name=bts.name,
							minimum=bts.minimum,
							maximum=bts.maximum,
							shape=(lim2-lim,))
						   )
	return btss
	
def split_time_step_spec(ts, lims=[]):
	"""
	Splits only one dimensional tensors
	"""

	bts = ts.observation
	n = bts.shape[0]
	
	btss = split_bounded_tensor_spec(bts, lims=lims)
	#rwds = [ts.reward[i:i+1] for i in range(len(lims))]
	#print (n)
	
	tss = []
	for i, bt in enumerate(btss):
		reward_spec = tf.TensorSpec(shape=(),
								dtype=tf.float32,
								name='reward')
		tss.append(TimeStep(step_type=ts.step_type,
							  reward=reward_spec,#rwds[i],#ts.reward,
							  discount=ts.discount,
							  observation=bt)
							)

	return tss


class RewardObserver:
	def __call__(self, trajectory):
		self.reward = trajectory.reward
	
	def result(self):
		return self.reward


class SingleGameFromMulti:
	"""
	"freezes" the environment.
	"""
	def __init__(self, env):
		self.env = env
		self.gym_env = self.env.pyenv.envs[0].gym
		
	def compute_reward(self, action):
		self.env.pyenv.envs[0].gym.apply_action(action)
		reward, reward_tot, cost_tot, cost_per_c, allocation, reward_fake, cost_true_per_c, transferred_cost = self.gym_env.compute_reward()

		return reward
	
	def compute_all(self, action):
		self.env.pyenv.envs[0].gym.apply_action(action)
		reward, reward_tot, cost_tot, cost_per_c, allocation, reward_fake, cost_true_per_c, transferred_cost = self.gym_env.compute_reward()
		
		return reward, reward_tot, cost_tot, cost_per_c, allocation, reward_fake, cost_true_per_c, transferred_cost, self.gym_env.best_cost_per_c

class MultiGameFromMulti:
	"""
	"freezes" the environment.

	for different players.
	"""
	def __init__(self, env):
		self.env = env
		self.gym_env = self.env.pyenv.envs[0].gym
		
	def compute_reward(self, action):
		self.env.pyenv.envs[0].gym.apply_action(action)
		reward, reward_tot, cost_tot, cost_per_c, allocation, reward_fake, transferred_cost = self.gym_env.compute_reward()

		return reward
	
	def compute_all(self, action):
		self.env.pyenv.envs[0].gym.apply_action(action)
		rewards, reward_tot, cost_tot, cost_per_c, allocation, reward_fake, transferred_cost = self.gym_env.compute_reward()
		
		return rewards, reward_tot, cost_tot, cost_per_c, allocation, reward_fake, transferred_cost, self.gym_env.best_cost_per_c


def function_builder(env):
	sg = SingleGameFromMulti(env)
	def f(x):
		return 100-sg.compute_reward(x)
	
	return f

def get_results_builder(env):
	sg = SingleGameFromMulti(env)
	def f(x):
		return sg.compute_all(x)

	return f

def get_results_builder_multi(env):
	sg = MultiGameFromMulti(env)
	def f(x):
		return sg.compute_all(x)

	return f

def compute_cost_diff(row, player='A'):
	#print ('COIN', type(row))
	#print ('AH', row['cost_per_c'])
	#try:
	if type(row)==np.float64:
		row = float(row)
	cost_diff = row['cost_per_c'][player] - row['best_cost_per_c'][player]
	# except Exception as e:
	# 	print ('BANG2', e)
	# 	raise
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
	
class DiscreteGameTrainer:
	def __init__(self):
		pass

	def build_game(self, game='jump', n_f = 10, n_a = 3, n_f_player = 4,
		dm = 20, dj = 20, new_capacity = 5, trading_alg='nnbound'):

		self.game = game
		self.n_f_player = n_f_player
		self.n_f = n_f
		
		if game=='jump':
			self.gym_env = MultiStochGameJumpFlatSpaces(n_f=n_f,
													n_a=n_a,
													seed=None,
													offset=100,
													trading_alg=trading_alg,
													dm=dm,
													dj=dj,
													n_f_player=n_f_player,
													new_capacity=new_capacity)
		elif game=='margin':
			self.gym_env = MultiStochGameMarginFlatSpaces(n_f=n_f,
													n_a=n_a,
													seed=None,
													offset=100,
													trading_alg=trading_alg,
													dm=dm,
													dj=dj,
													n_f_player=n_f_player,
													new_capacity=new_capacity)

		elif game=='margin_jump':
			raise Exception('Not implemented yet')
		else:
			raise Exception('Not implemented')

		py_env = suite_gym.wrap_env(self.gym_env)
		self.train_env = tf_py_environment.TFPyEnvironment(py_env)

		py_env = suite_gym.wrap_env(self.gym_env)
		self.eval_env = tf_py_environment.TFPyEnvironment(py_env)

	def build_agent(self, agent_type='ucb', learning_rate = 0.01, fc_layer_params=(100,),
		boltzmann_temperature=10, epsilon_greedy=None, reward_scale_factor=0.01):

		self.agent_type = agent_type

		if agent_type=='dqn':
			optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

			q_net = q_network.QNetwork(self.train_env.observation_spec(),
										self.train_env.action_spec(),
										fc_layer_params=fc_layer_params)

			self.agent = dqn_agent.DqnAgent(self.train_env.time_step_spec(),
										self.train_env.action_spec(),
										q_network=q_net,
										boltzmann_temperature=boltzmann_temperature,
										epsilon_greedy=epsilon_greedy,
										reward_scale_factor=reward_scale_factor,
										optimizer=optimizer,
										td_errors_loss_fn=common.element_wise_squared_loss,
										train_step_counter=tf.Variable(0),
										)

		elif agent_type=='ucb':
			self.agent = lin_ucb_agent.LinearUCBAgent(time_step_spec=self.train_env.time_step_spec(),
														action_spec=self.train_env.action_spec())

		self.agent.initialize()

	# def compute_avg_return(environment, policy, num_episodes=10, max_time_step_per_episode=1):
	# 	total_return = 0.0
	# 	for _ in range(num_episodes):

	# 		time_step = environment.reset()
	# 		episode_return = 0.0

	# 		i = 0
	# 		while not time_step.is_last() and i<max_time_step_per_episode:
	# 			action_step = policy.action(time_step)
	# 			time_step = environment.step(action_step.action)
	# 			episode_return += time_step.reward
	# 			i += 1
	# 		total_return += episode_return

	# 	avg_return = total_return / num_episodes
	# 	return avg_return.numpy()[0]

	def prepare_buffer(self, replay_buffer_max_length = 50000, initial_collect_steps = 100, batch_size = 1):

		self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=self.agent.collect_data_spec,
																		#data_spec=agent.policy.trajectory_spec,
																		batch_size=self.train_env.batch_size,
																		max_length=replay_buffer_max_length)

		collect_data(self.train_env, self.agent.policy, self.replay_buffer, initial_collect_steps)

		# Dataset generates trajectories with shape [Bx2x...]
		self.dataset = self.replay_buffer.as_dataset(num_parallel_calls=3, 
												sample_batch_size=batch_size, 
												num_steps=2).prefetch(3)

	def save_policy(self, file_name=None):
		if file_name is None:
			file_name = self.agent_type + '_' + self.game + '_' + str(self.n_f_player) + 'a' + str(self.n_f) + '_policy'

		PolicySaver(agent.policy).save('ucb_jump_4a10_policy')

	def train_agent(self, num_iterations = 10000, steps_per_loop = 1, collect_steps_per_iteration = 1,
		n_eval_setp= 500):

		reward_observer = RewardObserver()

		#observers = [replay_buffer.add_batch, reward_observer]#, avg_reward_observer]#, regret_metric]

		iterator = iter(self.dataset)

		# driver = dynamic_step_driver.DynamicStepDriver(
		#                                                 env=environment,
		#                                                 policy=agent.collect_policy,
		#                                                 num_steps=steps_per_loop * batch_size,
		#                                                 observers=observers)

		self.rewards = []
		#avg_reward_values = []

		for i in range(num_iterations):
			 # Collect a few steps using collect_policy and save to the replay buffer.
			collect_data(self.train_env, self.agent.collect_policy, self.replay_buffer, collect_steps_per_iteration)

			# Sample a batch of data from the buffer and update the agent's network.
			experience, unused_info = next(iterator)
			reward_observer(experience)
			self.rewards.append(reward_observer.result().numpy().mean())

			train_loss = self.agent.train(experience).loss

			#step = agent.train_step_counter.numpy()
			
			#driver.run()
			#loss_info = agent.train(replay_buffer.gather_all(), )
			#replay_buffer.clear()
			#reward_values.append(reward_observer.result())
			#avg_reward_values.append(avg_reward_observer.result())
			
			if i%n_eval_setp==0 and i>0:
				print ('i=', i, '; avg reward on last 100 runs:', np.array(self.rewards)[-100:].mean())

	def do_plots(self, instantaneous=False):
		if instantaneous:
			plt.plot(self.rewards, label='Intantaneous reward')
		N = 50
		y = pd.Series(self.rewards).rolling(window=N).mean().iloc[N-1:].values
		plt.plot(y, label='Average 50 runs')
		N = 500
		y = pd.Series(self.rewards).rolling(window=N).mean().iloc[N-1:].values
		plt.plot(y, label='Average 500 runs')
		plt.ylabel('Reward')
		plt.xlabel('Number of Iterations')
		plt.legend()

	def print_specs(self):
		print ('Environment specs:')
		print ('Observation Specs:', train_env.observation_spec())
		print ('Action Specs:',train_env.action_spec())
		print ('Reward Specs:',train_env.reward_spec())
		print ('Time Specs:',train_env.time_step_spec())
		print ()


class ContinuousGameTrainer:
	def __init__(self):
		pass

	def build_game(self, game='jump', **kwargs_game):
		if game=='jump':
			self.gym_env = ContGameJump(**kwargs_game)
		elif game=='margin':
			self.gym_env = ContGameMargin(**kwargs_game)
		elif game=='jump_margin':
			self.gym_env = ContGame(**kwargs_game)
		else:
			raise Exception('Unknown game:', game)

		self.build_wrapper()

	def build_wrapper(self):
		py_env = suite_gym.wrap_env(self.gym_env)
		self.train_env = tf_py_environment.TFPyEnvironment(py_env)

		self.observation_spec, self.action_spec, self.time_step_spec = (spec_utils.get_tensor_specs(self.train_env))

		py_env = suite_gym.wrap_env(self.gym_env)
		self.collect_env = tf_py_environment.TFPyEnvironment(py_env)

	def build_agent(self, critic_learning_rate=3e-4, actor_learning_rate=3e-4, alpha_learning_rate=3e-4,
		target_update_tau=0.005, target_update_period=1, gamma=0.99, reward_scale_factor=1.0,
		actor_fc_layer_params=(256, 256), critic_joint_fc_layer_params=(256, 256)):

		critic_net = critic_network.CriticNetwork((self.observation_spec, self.action_spec),
													observation_fc_layer_params=None,
													action_fc_layer_params=None,
													joint_fc_layer_params=critic_joint_fc_layer_params,
													kernel_initializer='glorot_uniform',
													last_kernel_initializer='glorot_uniform')

		actor_net = actor_distribution_network.ActorDistributionNetwork(self.observation_spec,
																		self.action_spec,
																		fc_layer_params=actor_fc_layer_params,
																		continuous_projection_net=(tanh_normal_projection_network.TanhNormalProjectionNetwork))

		train_step = train_utils.create_train_step()

		self.tf_agent = sac_agent.SacAgent(self.time_step_spec,
											self.action_spec,
											actor_network=actor_net,
											critic_network=critic_net,
											actor_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=actor_learning_rate),
											critic_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=critic_learning_rate),
											alpha_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=alpha_learning_rate),
											target_update_tau=target_update_tau,
											target_update_period=target_update_period,
											td_errors_loss_fn=tf.math.squared_difference,
											gamma=gamma,
											reward_scale_factor=reward_scale_factor,
											train_step_counter=train_step)

		self.tf_agent.initialize()

	def set_price_jump(self, price):
		self.gym_env.set_price_jump(price)
		self.build_wrapper()

	def set_price_cost(self, price):
		self.gym_env.set_price_cost(price)
		self.build_wrapper()

	def set_price_margin(self, price):
		self.gym_env.set_price_margin(price)
		self.build_wrapper()

	def game_summary(self):
		self.collect_env.pyenv.envs[0].gym.game_summary()

	def prepare_buffer(self, replay_buffer_max_length=200, initial_collect_steps=10, batch_size=64):

		self.random_policy = random_tf_policy.RandomTFPolicy(self.collect_env.time_step_spec(), self.collect_env.action_spec())

		self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=self.tf_agent.collect_data_spec,
																		batch_size=self.train_env.batch_size,
																		max_length=replay_buffer_max_length)

		collect_data(self.train_env, self.random_policy, self.replay_buffer, initial_collect_steps)

		# Dataset generates trajectories with shape [Bx2x...]
		self.dataset = self.replay_buffer.as_dataset(num_parallel_calls=3, 
													sample_batch_size=batch_size, 
													num_steps=2).prefetch(3)

	def train_agent(self, num_iterations=4000, n_eval_setp=500):
		# Reset the train step
		self.tf_agent.train_step_counter.assign(0)

		reward_observer = RewardObserver()

		# Evaluate the agent's policy once before training.
		#avg_return = get_eval_metrics()["AverageReturn"]
		#returns = [avg_return]

		iterator = iter(self.dataset)

		collect_steps_per_iteration = 1

		self.rewards = []

		for i in range(num_iterations):
			collect_data(self.train_env,
						self.tf_agent.collect_policy,
						self.replay_buffer,
						collect_steps_per_iteration)

			# Sample a batch of data from the buffer and update the agent's network.
			experience, unused_info = next(iterator)
			reward_observer(experience)
			self.rewards.append(reward_observer.result().numpy().mean())

			train_loss = self.tf_agent.train(experience).loss

			if i%n_eval_setp==0 and i>0:
				print ('i=', i, '; avg reward on last 100 runs:', np.array(self.rewards)[-100:].mean())

	def save_policy(self, file_name=None):
		if file_name is None:
			file_name = self.agent_type + '_' + self.game + '_' + str(self.n_f_player) + 'a' + str(self.n_f) + '_policy'

		PolicySaver(self.tf_agent.policy).save(file_name)

	def load_policy(self, file_name=None):
		saved_policy = tf.compat.v2.saved_model.load(file_name)
		#policy_state = saved_policy.get_initial_state(batch_size=3)
		
		# placeholder for agent
		class DummyAgent:
			pass

		self.tf_agent = DummyAgent()
		self.tf_agent.policy = saved_policy

	def do_plots(self, file_name=None, instantaneous=False):
		fig, ax = plt.subplots()
		if instantaneous:
			ax.plot(self.rewards, lw=0.5, label='Instantaneous reward', alpha=0.6)
		N = 50
		y = pd.Series(self.rewards).rolling(window=N).mean().iloc[N-1:].values
		ax.plot(y, label='Average 50 runs')
		N = 500
		y = pd.Series(self.rewards).rolling(window=N).mean().iloc[N-1:].values
		ax.plot(y, label='Average 500 runs')
		ax.set_ylabel('Reward')
		ax.set_xlabel('Number of Iterations')
		ax.legend()

		if not file_name is None:
			plt.savefig(file_name, bbox_inches='tight')

	def evaluation(self, n_evaluations=100, show_results=True, file_name=None):
		rewards_eval = []

		for i in range(n_evaluations):
			time_step = self.collect_env.reset()
			action = self.tf_agent.policy.action(time_step)
			stuff = self.train_env.step(action)
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
		builder = get_results_builder(self.collect_env)
		for i in range(n_iter):
			if n_iter>=10 and i%int(n_iter/10)==0:
				print ('i=', i)
			time_step = self.collect_env.reset()

			action = self.tf_agent.policy.action(time_step)
			results[i] = builder(action.action.numpy()[0])
			stuff = self.collect_env.step(action)

		df = pd.DataFrame(results, index=['reward',
										'reward_tot',
										'cost_tot',
										'cost_per_c',
										'allocation',
										'reward_fake',
										'cost_true_per_c',
										'transferred_cost',
										'best_cost_per_c']).T
		#print (df)
		dg = df.T.apply([compute_cost_diff, compute_cost_diff_ratio, compute_cost_diff_others, compute_cost_diff_ratio_others]).T
		df = pd.concat([df, dg], axis=1)

		# Unravel cost per company
		dg = pd.DataFrame(list(df['cost_per_c']))
		dg.rename(columns={k:'Cost {}'.format(k) for k in dg.columns}, inplace=True)
		dg2 = pd.DataFrame(list(df['best_cost_per_c']))
		dg2.rename(columns={k:'Best cost {}'.format(k) for k in dg2.columns}, inplace=True)
		dg3 = pd.DataFrame(list(df['cost_true_per_c']))
		dg3.rename(columns={k:'True cost {}'.format(k) for k in dg3.columns}, inplace=True)
		df = pd.concat([df, dg, dg2, dg3], axis=1)
		df.drop(columns=['cost_per_c', 'best_cost_per_c', 'allocation', 'cost_true_per_c'], inplace=True)

		if show_results:
			print ('Average results:')
			print (df.mean())

		if not file_name is None:
			df.to_csv(file_name)

		return df

	def print_specs(self):
		print ('Environment specs:')
		print ('Observation Specs:', self.train_env.observation_spec())
		print ('Action Specs:',self.train_env.action_spec())
		print ('Reward Specs:',self.train_env.reward_spec())
		print ('Time Specs:',self.train_env.time_step_spec())
		print ()


class ContinuousMGameTrainer:
	"""
	Multi player version
	"""
	def __init__(self):
		pass

	def build_game(self, game='jump', **kwargs_game):
		if game=='jump':
			self.gym_env = ContMGameJump(**kwargs_game)
		elif game=='margin':
			self.gym_env = ContMGameMargin(**kwargs_game)
		elif game=='jump_margin':
			self.gym_env = ContMGame(**kwargs_game)
		else:
			raise Exception('Unknown game:', game)

		self.build_wrapper()

	def build_wrapper(self):
		py_env = suite_gym.wrap_env(self.gym_env)
		# print ('POUETTTTTT', py_env.step(np.array((0, 1, 1, 0, 1, 0, 1))))
		# print ('COOOOIIIIN', py_env.step(np.array((0, 1, 1, 0, 1, 0, 1))))
		self.train_env = tf_py_environment.TFPyEnvironment(py_env)

		print ()

		#print ('HASSSSSSS;', self.train_env.current_time_step())

		self.observation_spec, self.action_spec, ts = (spec_utils.get_tensor_specs(self.train_env))

		# print ('self.observation_spec:', self.observation_spec)
		# print ()
		# print ('self.action_spec:', self.action_spec)
		# print ()
		# print ('self.time_step_spec INIT:', ts)

		reward_spec = tf.TensorSpec(shape=(len(self.gym_env.n_f_players), ),
								dtype=tf.float32,
								name='reward')

		self.time_step_spec = TimeStep(step_type=ts.step_type,
										  reward=reward_spec,
										  discount=ts.discount,
										  observation=ts.observation)
		# print ()
		# print ('self.time_step_spec:', self.time_step_spec)
		# print ()
		# print ()

		self.lims_simple = self.gym_env.n_f_players[:-1]
		self.lims_simple = np.cumsum(self.lims_simple)
		#self.lims_double = [2 * lim for lim in self.lims_simple]
		self.lims_triple = [3 * lim for lim in self.lims_simple]
		#print ('lims_simple:', self.lims_simple)
		#print ('lims_triple:', self.lims_triple)

		self.observation_specs = split_bounded_tensor_spec(self.observation_spec, lims=self.lims_triple)
		self.action_specs = split_bounded_tensor_spec(self.action_spec, lims=self.lims_simple)
		#print (self.action_specs)
		self.time_step_specs = split_time_step_spec(self.time_step_spec, lims=self.lims_triple)

		# print ('self.observation_specs:', self.observation_specs)
		# print ('self.action_specs:', self.action_specs)
		# print ('self.time_step_specs:', self.time_step_specs)
		# print ()

		py_env = suite_gym.wrap_env(self.gym_env)
		self.collect_env = tf_py_environment.TFPyEnvironment(py_env)

		# print ('POUET', self.collect_env.time_step_spec())
		# print ('PONO', self.collect_env.action_spec())
		# print ('POUET', self.train_env.time_step_spec())
		# print ('PONO', self.train_env.action_spec())

		# print ()

	def build_agent(self, critic_learning_rate=3e-4, actor_learning_rate=3e-4, alpha_learning_rate=3e-4,
		target_update_tau=0.005, target_update_period=1, gamma=0.99, reward_scale_factor=1.0,
		actor_fc_layer_params=(256, 256), critic_joint_fc_layer_params=(256, 256),
		observation_spec=(), action_spec=(), time_step_spec=()):

		# print ('observation_spec:', observation_spec)
		# print ('action_spec:', action_spec)
		# print ('time_step_spec:', time_step_spec)

		# print ()

		#raise Exception()

		critic_net = critic_network.CriticNetwork((observation_spec, action_spec),
													observation_fc_layer_params=None,
													action_fc_layer_params=None,
													joint_fc_layer_params=critic_joint_fc_layer_params,
													kernel_initializer='glorot_uniform',
													last_kernel_initializer='glorot_uniform')

		actor_net = actor_distribution_network.ActorDistributionNetwork(observation_spec,
																		action_spec,
																		fc_layer_params=actor_fc_layer_params,
																		continuous_projection_net=(tanh_normal_projection_network.TanhNormalProjectionNetwork))

		train_step = train_utils.create_train_step()

		tf_agent = sac_agent.SacAgent(time_step_spec,
										action_spec,
										actor_network=actor_net,
										critic_network=critic_net,
										actor_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=actor_learning_rate),
										critic_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=critic_learning_rate),
										alpha_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=alpha_learning_rate),
										target_update_tau=target_update_tau,
										target_update_period=target_update_period,
										td_errors_loss_fn=tf.math.squared_difference,
										gamma=gamma,
										reward_scale_factor=reward_scale_factor,
										train_step_counter=train_step)

		tf_agent.initialize()

		return tf_agent

	def build_agents(self, critic_learning_rate=3e-4, actor_learning_rate=3e-4, alpha_learning_rate=3e-4,
		target_update_tau=0.005, target_update_period=1, gamma=0.99, reward_scale_factor=1.0,
		actor_fc_layer_params=(256, 256), critic_joint_fc_layer_params=(256, 256)):

		self.tf_agents = [self.build_agent(critic_learning_rate=critic_learning_rate,
									actor_learning_rate=actor_learning_rate,
									alpha_learning_rate=alpha_learning_rate,
									target_update_tau=target_update_tau,
									target_update_period=target_update_period,
									gamma=gamma,
									reward_scale_factor=reward_scale_factor,
									actor_fc_layer_params=actor_fc_layer_params,
									critic_joint_fc_layer_params=critic_joint_fc_layer_params,
									observation_spec=self.observation_specs[i],
									action_spec=self.action_specs[i],
									time_step_spec=self.time_step_specs[i]) for i in range(len(self.gym_env.players))]

	def set_price_jump(self, price):
		self.gym_env.set_price_jump(price)
		self.build_wrapper()

	def set_price_cost(self, price):
		self.gym_env.set_price_cost(price)
		self.build_wrapper()

	def set_price_margin(self, price):
		self.gym_env.set_price_margin(price)
		self.build_wrapper()

	def game_summary(self):
		self.collect_env.pyenv.envs[0].gym.game_summary()

	def prepare_buffers(self, replay_buffer_max_length=200, initial_collect_steps=10, batch_size=64):

		#self.random_policy = random_tf_policy.RandomTFPolicy(self.collect_env.time_step_spec(), self.collect_env.action_spec())
		self.rand_policies = [random_tf_policy.RandomTFPolicy(self.time_step_specs[i],
															self.action_specs[i]) for i in range(len(self.observation_specs))]

		self.replay_buffers = [tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
																				batch_size=self.train_env.batch_size,
																				max_length=replay_buffer_max_length) for agent in self.tf_agents]

		collect_data_multi(self.train_env,
							self.rand_policies,
							self.replay_buffers,
							initial_collect_steps,
							lims=self.lims_triple)

		# Dataset generates trajectories with shape [Bx2x...]
		self.datasets = [buf.as_dataset(num_parallel_calls=3, 
										sample_batch_size=batch_size, 
										num_steps=2).prefetch(3) for buf in self.replay_buffers]

	def train_agents(self, num_iterations=4000, n_eval_setp=500):
		# Reset the train step
		[agent.train_step_counter.assign(0) for agent in self.tf_agents]

		reward_observers = [RewardObserver() for i in range(len(self.tf_agents))]

		# Evaluate the agent's policy once before training.
		#avg_return = get_eval_metrics()["AverageReturn"]
		#returns = [avg_return]

		iterators = [iter(dataset) for dataset in self.datasets]

		collect_steps_per_iteration = 1

		self.rewards = [[], []]

		players = self.collect_env.pyenv.envs[0].gym.players

		for i in range(num_iterations):
			collect_data_multi(self.train_env,
							[agent.collect_policy for agent in self.tf_agents],
							self.replay_buffers,
							collect_steps_per_iteration,
							lims=self.lims_triple)

			# Sample a batch of data from the buffer and update the agent's network.
			for j in range(len(self.tf_agents)):
				iterator = iterators[j]
				experience, unused_info = next(iterator)
				reward_observers[j](experience)
				self.rewards[j].append(reward_observers[j].result().numpy().mean())

				train_loss = self.tf_agents[j].train(experience).loss

			if i%n_eval_setp==0 and i>0:
				print ('i=', i, '; avg rewards on last 100 runs:', [np.array(self.rewards[j])[-100:].mean() for j in range(len(self.tf_agents))])

	def save_policies(self, file_names=None):
		# if file_name is None:
		# 	file_name = self.agent_type + '_' + self.game + '_' + str(self.n_f_player) + 'a' + str(self.n_f) + '_policy'

		for i, agent in enumerate(self.tf_agents):
			PolicySaver(agent.policy).save(file_names[i])

	def load_policies(self, file_names=None):
		saved_policies = [tf.compat.v2.saved_model.load(file_name) for file_name in file_names]
		#policy_state = saved_policy.get_initial_state(batch_size=3)
		
		# placeholder for agent
		class DummyAgent:
			pass

		self.tf_agents = []
		for saved_policy in saved_policies:
			tf_agent = DummyAgent()
			tf_agent.policy = saved_policy
			self.tf_agents.append(tf_agent)

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

	def evaluation(self, n_evaluations=100, show_results=True, file_name=None):
		"""Probably obsolete"""
		rewards_eval = []

		for i in range(n_evaluations):
			time_step = self.collect_env.reset()
			action = self.tf_agent.policy.action(time_step)
			stuff = self.train_env.step(action)
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
		builder = get_results_builder_multi(self.collect_env)
		for i in range(n_iter):
			if n_iter>=10 and i%int(n_iter/10)==0:
				print ('i=', i)
			time_step = self.collect_env.reset()
			rew = tf.constant([[100.]*len(self.tf_agents)], dtype=tf.float32)
			step_type = tf.constant([[1]*len(self.tf_agents)], dtype=tf.int32)
			time_step = TimeStep(step_type=step_type,
								  reward=rew,
								  discount=time_step.discount,
								  observation=time_step.observation)

			# ADD lims
			#  Split time step
			sts = split_time_step(time_step, lims=self.lims_triple)

			# Get action based on partial observation
			asts = []
			for j, ts in enumerate(sts):
				asts.append(self.tf_agents[j].policy.action(ts))

			# Merge actions
			action_step = merge_action_steps(asts)

			# # Get next state
			# next_time_step = environment.step(action_step.action)

			#action = self.tf_agent.policy.action(time_step)
			results[i] = builder(action_step.action.numpy()[0])
			#stuff = self.collect_env.step(action)

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

	def print_specs(self):
		print ('Environment specs:')
		print ('Observation Specs:', self.train_env.observation_spec())
		print ('Action Specs:',self.train_env.action_spec())
		print ('Reward Specs:',self.train_env.reward_spec())
		print ('Time Specs:',self.train_env.time_step_spec())
		print ()



		
		