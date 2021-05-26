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
from tf_agents.networks import actor_distribution_network
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import train_utils
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.specs import BoundedTensorSpec

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

from Hotspot.RL.agents import Agent
#from Hotspot.RL.continuous_game import ContGame, ContGameJump, ContGameMargin
from Hotspot.RL.mcontinuous_game import ContMGame, ContMGameJump, ContMGameMargin

from Hotspot.libs.other_tools import print_allocation
from Hotspot.libs.uow_tool_belt.general_tools import nice_colors

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


class TFAgentWrap(Agent):
	"""
	Just a wrapper to use easily with training and evaluation.
	"""
	def __init__(self, tf_agent=None, kind='TF'):
		self.kind = kind
		if not tf_agent is None:
			self.wrap_tf_agent(tf_agent)

	def wrap_tf_agent(self, tf_agent):
		self.policy = tf_agent.policy
		# for testing
		#self.tf_agent = tf_agent

	def load(self, file_name):
		self.policy = tf.compat.v2.saved_model.load(file_name)

	def save(self, file_name):
		if file_name is None:
			file_name = self.agent_type + '_' + self.game + '_' + str(self.n_f_player) + 'a' + str(self.n_f) + '_policy'

		PolicySaver(self.policy).save(file_name)

	def action(self, observation):
		"""
		observation is an observation out of the gym environment, i.e. a numpy array. 
		It needs to be converted a tf time step.
		"""
		# Converting observation
		rew = tf.constant([100.], dtype=tf.float32)
		step_type = tf.constant([1], dtype=tf.int32)
		discount = tf.constant([0.99], dtype=tf.float32)
		observation_tf = tf.constant(np.array([observation]), dtype=tf.float32)

		time_step = TimeStep(step_type=step_type,
							  reward=rew,
							  discount=discount,
							  observation=observation_tf)
		# Getting action
		action = self.policy.action(time_step)

		# Converting action back to numpy
		action = action.action.numpy()

		return action

	def action_tf(self, time_step):
		# Unwrapped action results
		action = self.policy.action(time_step)

		return action


class TFAgentTrainer:
	"""
	Multi player version
	"""
	def __init__(self):
		pass

	def build_game_wrapper(self, gym_env):
		py_env = suite_gym.wrap_env(gym_env)
		self.train_env = tf_py_environment.TFPyEnvironment(py_env)

		self.observation_spec, self.action_spec, ts = (spec_utils.get_tensor_specs(self.train_env))

		reward_spec = tf.TensorSpec(shape=(len(gym_env.n_f_players), ),
								dtype=tf.float32,
								name='reward')

		self.time_step_spec = TimeStep(step_type=ts.step_type,
										  reward=reward_spec,
										  discount=ts.discount,
										  observation=ts.observation)

		self.lims_action = gym_env.lims_action
		self.lims_observation = gym_env.lims_observation

		self.observation_specs = split_bounded_tensor_spec(self.observation_spec, lims=gym_env.lims_observation)
		self.action_specs = split_bounded_tensor_spec(self.action_spec, lims=gym_env.lims_action)
		self.time_step_specs = split_time_step_spec(self.time_step_spec, lims=gym_env.lims_observation)

		py_env = suite_gym.wrap_env(gym_env)
		self.collect_env = tf_py_environment.TFPyEnvironment(py_env)

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

	def build_agents(self, gym_env=None, critic_learning_rate=3e-4, actor_learning_rate=3e-4, alpha_learning_rate=3e-4,
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
									time_step_spec=self.time_step_specs[i]) for i in range(len(gym_env.players))]

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
							lims=self.lims_observation)

		# Dataset generates trajectories with shape [Bx2x...]
		self.datasets = [buf.as_dataset(num_parallel_calls=3, 
										sample_batch_size=batch_size, 
										num_steps=2).prefetch(3) for buf in self.replay_buffers]

	def train_agents(self, num_iterations=4000, n_eval_setp=500):
		self.prepare_buffers()
		
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
							lims=self.lims_observation)

			# Sample a batch of data from the buffer and update the agent's network.
			for j in range(len(self.tf_agents)):
				iterator = iterators[j]
				experience, unused_info = next(iterator)
				reward_observers[j](experience)
				self.rewards[j].append(reward_observers[j].result().numpy().mean())

				train_loss = self.tf_agents[j].train(experience).loss

			if i%n_eval_setp==0 and i>0:
				print ('i=', i, '; avg rewards on last 100 runs:', [np.array(self.rewards[j])[-100:].mean() for j in range(len(self.tf_agents))])

	def print_specs(self):
		print ('Environment specs:')
		print ('Observation Specs:', self.train_env.observation_spec())
		print ('Action Specs:',self.train_env.action_spec())
		print ('Reward Specs:',self.train_env.reward_spec())
		print ('Time Specs:',self.train_env.time_step_spec())
		print ()


class SBAgentWarp(Agent):
	def __init__(self, model=None, kind='SB'):
		self.kind = kind
		if not model is None:
			self.wrap_sb_agent(model)

	def wrap_sb_agent(self, model):
		self.predict = model.predict

	def action(self, observation):
		action = self.predict(observation)[0]

		# print ('Action from SB agent:', action)
		# print ('Corresponding observation:', observation)

		return action

class SBAgentTrainer:
	# TODO: make multi agent. Have a look at dictionnaries
	# in stable-baselines 3
	def set_env(self, gym_env, n_envs=1):
		if n_envs>1:
			# Not sure this works
			self.gym_env = make_vec_env(gym_env, n_envs=n_envs)
		else:
			self.gym_env = gym_env

	def build_agent(self, policy='MlpPolicy', verbose=0):
		return PPO(policy, self.gym_env, verbose=verbose, n_steps=2)

	def build_agents(self, players, **kwargs):
		self.models = [self.build_agent(**kwargs) for player in players]

	def train_agents(self, num_iterations=4000, n_eval_setp=500):
		# TODO: multi agent
		#print ('learning on {} steps'.format(num_iterations))

		self.models[0].learn(total_timesteps=num_iterations)


		
		