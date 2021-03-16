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

from RL.multiple_game import MultiStochGameJumpFlatSpaces, MultiStochGameMarginFlatSpaces
from RL.continuous_game2 import ContGame, ContGameJump, ContGameMargin

from libs.tools import print_allocation

# For videos
# import imageio
# import base64
# import IPython

def collect_step(environment, policy, buffer):
	time_step = environment.current_time_step()
	action_step = policy.action(time_step)
	next_time_step = environment.step(action_step.action)
	traj = trajectory.from_transition(time_step, action_step, next_time_step)
	
	#print (traj)

	# Add trajectory to the replay buffer
	buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
	for _ in range(steps):
		collect_step(env, policy, buffer)


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
		reward, reward_tot, cost_tot, cost_per_c, allocation, reward_fake = self.gym_env.compute_reward()

		return reward
	
	def compute_all(self, action):
		self.env.pyenv.envs[0].gym.apply_action(action)
		reward, reward_tot, cost_tot, cost_per_c, allocation, reward_fake = self.gym_env.compute_reward()
		
		return reward, reward_tot, cost_tot, cost_per_c, allocation, reward_fake, self.gym_env.best_cost_per_c

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

def compute_cost_diff(row, player='A'):
	cost_diff = row['cost_per_c'][player] - row['best_cost_per_c'][player]
	return cost_diff

def compute_cost_diff_ratio(row, player='A'):
	cost_diff = (row['cost_per_c'][player] - row['best_cost_per_c'][player])/row['best_cost_per_c'][player]
	return cost_diff

def compute_cost_diff_others(row, player='A'):
	c = np.array([cc for air, cc in row['cost_per_c'].items() if air!=player]).sum()
	cb = np.array([cc for air, cc in row['best_cost_per_c'].items() if air!=player]).sum()
	cost_diff = c-cb
	return cost_diff

def compute_cost_diff_ratio_others(row, player='A'):
	c = np.array([cc for air, cc in row['cost_per_c'].items() if air!=player]).sum()
	cb = np.array([cc for air, cc in row['best_cost_per_c'].items() if air!=player]).sum()
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
		#n_f = 10, n_a = 3, n_f_player = 4,
		#new_capacity = 5, trading_alg='nnbound'):

		if game=='jump':
			gym_env = ContGameJump(**kwargs_game)
		elif game=='margin':
			gym_env = ContGameMargin(**kwargs_game)
		elif game=='jump_margin':
			gym_env = ContGame(**kwargs_game)
		else:
			raise Exception('Unknown game:', game)

		py_env = suite_gym.wrap_env(gym_env)
		self.train_env = tf_py_environment.TFPyEnvironment(py_env)

		self.observation_spec, self.action_spec, self.time_step_spec = (spec_utils.get_tensor_specs(self.train_env))

		py_env = suite_gym.wrap_env(gym_env)
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

		actor_net = actor_distribution_network.ActorDistributionNetwork(
			  self.observation_spec,
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

	def do_plots(self, instantaneous=False):
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

	def evaluation(self, n_evaluations=1000):
		rewards_eval = []

		for i in range(n_evaluations):
			time_step = self.collect_env.reset()
			action = self.tf_agent.policy.action(time_step)
			stuff = self.train_env.step(action)
			rewards_eval.append(stuff.reward.numpy()[0])
		rewards_eval = np.array(rewards_eval)

		print ('Positive rewards:', 100 * len(rewards_eval[rewards_eval>100])/len(rewards_eval), '%')
		print ('Neutral rewards:', 100 * len(rewards_eval[rewards_eval==100])/len(rewards_eval), '%')
		print ('Negative rewards:', 100 * len(rewards_eval[rewards_eval<100])/len(rewards_eval), '%')
		print ('Average reward:', rewards_eval.mean())
		print ('Reward std:', rewards_eval.std())
		plt.hist(rewards_eval, bins=20)

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

	def compare_airlines(self, n_iter=100):
		results = {}
		builder = get_results_builder(self.collect_env)
		for i in range(n_iter):
			if i%int(n_iter/10)==0:
				print ('i=', i)
			time_step = self.collect_env.reset()

			action = self.tf_agent.policy.action(time_step)
			results[i] = builder(action.action.numpy()[0])
			stuff = self.collect_env.step(action)

		df = pd.DataFrame(results, index=['reward', 'reward_tot', 'cost_tot', 'cost_per_c', 'allocation', 'reward_fake', 'best_cost_per_c']).T
		dg = df.T.apply([compute_cost_diff, compute_cost_diff_ratio, compute_cost_diff_others, compute_cost_diff_ratio_others]).T
		df = pd.concat([df, dg], axis=1)

		# Unravel cost per company
		dg = pd.DataFrame(list(df['cost_per_c']))
		dg.rename(columns={k:'Cost {}'.format(k) for k in dg.columns}, inplace=True)
		dg2 = pd.DataFrame(list(df['best_cost_per_c']))
		dg2.rename(columns={k:'Best cost {}'.format(k) for k in dg2.columns}, inplace=True)
		df = pd.concat([df, dg, dg2], axis=1)
		df.drop(columns=['cost_per_c', 'best_cost_per_c', 'allocation'], inplace=True)

		print ('Average results:')
		print (df.mean())

		return df
	
	def print_specs(self):
		print ('Environment specs:')
		print ('Observation Specs:', self.train_env.observation_spec())
		print ('Action Specs:',self.train_env.action_spec())
		print ('Reward Specs:',self.train_env.reward_spec())
		print ('Time Specs:',self.train_env.time_step_spec())
		print ()



		