#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np

from RL.game_trainer import ContinuousGameTrainer
from libs.tools import loop, agent_file_name, root_file_name

def wrapper_training(paras={}):
	do_training(**paras)

def wrapper_training_dummy(paras={}):
	print (paras)

def do_training(n_f=10, n_a=3, n_f_player=4, nn=10, n_h=2, game='jump',
	num_iterations=8000, learning_rate=3e-4, num_iterations_validation=1000,
	jump_price=0., version='v1.0'):
	trainer = ContinuousGameTrainer()
	trainer.build_game(game=game,
						n_f=n_f,
						n_a=n_a,
						n_f_player=n_f_player,
						price_jump=jump_price)
	trainer.build_agent(critic_learning_rate = learning_rate,
						actor_learning_rate = learning_rate,
						alpha_learning_rate = learning_rate,
						target_update_tau = 0.005,
						target_update_period = 1,
						gamma = 0.99,
						reward_scale_factor = 100.0,
						actor_fc_layer_params = (nn, nn),
						critic_joint_fc_layer_params = (nn, nn))
	trainer.prepare_buffer(initial_collect_steps=10, batch_size=2048)
	trainer.train_agent(num_iterations=num_iterations, n_eval_setp=500)
	
	#name = 'save_policies/v1.2/nf{}_na{}_nn{}_nfp{}_jp{}_{}'.format(n_f, n_a, nn, n_f_player, jump_price, game)
	name = agent_file_name(n_f_player,
							  nn=nn,
							  n_h=n_h,
							  v=version,
							  nf_tot_game=n_f,
							  n_a=n_a,
							  game_type='single',
							  game=game,
							  jp=jump_price)
	
	trainer.save_policy(name)
	trainer.do_plots(file_name='{}/training.png'.format(name),
					instantaneous=True)

	trainer.compare_airlines(n_iter=num_iterations_validation,
							show_results=False,
							file_name='{}/evaluation.csv'.format(name))

def do_iterations(iterated_paras={}):
	level_sc = list(iterated_paras.keys())
	paras = {k:l[0] for k, l in iterated_paras.items()}
	args = {'paras':paras}

	loop(iterated_paras,
		level_sc,
		paras,
		thing_to_do=wrapper_training,
		**args)

if __name__=='__main__':
	version = 'v1.3'

	parser = argparse.ArgumentParser(description='Mercury batch script', add_help=True)
	
	parser.add_argument('-nf', '--number_flights',
								help='Number of flights in total in hotspot',
								required=False,
								default=None,
								nargs='*')

	parser.add_argument('-na', '--number_airlines',
								help='Number of airlines in hotspot',
								required=False,
								default=None,
								nargs='*')

	parser.add_argument('-nfp', '--number_flights_player',
								help='Number of flights for player',
								required=False,
								default=None,
								nargs='*')

	parser.add_argument('-nit', '--num_iterations',
								help='Number of iterations in training',
								required=False,
								default=None,
								nargs='?')

	parser.add_argument('-nitv', '--num_iterations_validation',
								help='Number of iterations in validation',
								required=False,
								default=None,
								nargs='?')

	parser.add_argument('-nn', '--number_neurons',
								help='Number of neurons in each layer',
								required=False,
								default=None,
								nargs='?')

	parser.add_argument('-nh', '--number_layers',
								help='Number of hidden layers',
								required=False,
								default=None,
								nargs='?')

	parser.add_argument('-jp', '--jump_price',
								help='Price on jump',
								required=False,
								default=None,
								nargs='*')

	args = parser.parse_args()

	print ()

	#print (args)

	iterated_paras = {}

	# default steps
	nf_steps = 5
	na_steps = 1
	nfp_steps = 1

	jp_price_step = 1.

	if not (args.number_flights is None):
		if len(args.number_flights)==1:
			iterated_paras['n_f'] = [int(args.number_flights[0])]
		elif len(args.number_flights)==2:
			iterated_paras['n_f'] = list(range(int(args.number_flights[0]), int(args.number_flights[1])+1, nf_steps))
		else:
			iterated_paras['n_f'] = list(range(int(args.number_flights[0]), int(args.number_flights[1])+1, int(args.number_flights[2])))

	if not (args.number_airlines is None):
		if len(args.number_airlines)==1:
			iterated_paras['n_a'] = [int(args.number_airlines[0])]
		elif len(args.number_airlines)==2:
			iterated_paras['n_a'] = list(range(int(args.number_airlines[0]), int(args.number_airlines[1])+1, na_steps))
		else:
			iterated_paras['n_a'] = list(range(int(args.number_airlines[0]), int(args.number_airlines[1])+1, int(args.number_airlines[2])))

	if not (args.number_flights_player is None):
		if len(args.number_flights_player)==1:
			iterated_paras['n_f_player'] = [int(args.number_flights_player[0])]
		elif len(args.number_flights_player)==2:
			iterated_paras['n_f_player'] = list(range(int(args.number_flights_player[0]), int(args.number_flights_player[1])+1, nfp_steps))
		else:
			iterated_paras['n_f_player'] = list(range(int(args.number_flights_player[0]), int(args.number_flights_player[1])+1, int(args.number_flights_player[2])))

	if not (args.jump_price is None):
		if len(args.jump_price)==1:
			iterated_paras['jump_price'] = [float(args.jump_price[0])]
		elif len(args.jump_price)==2:
			iterated_paras['jump_price'] = list(np.arange(float(args.jump_price[0]), float(args.jump_price[1])+jp_price_step, jp_price_step))
		else:
			iterated_paras['jump_price'] = list(np.arange(float(args.jump_price[0]), float(args.jump_price[1])+float(args.jump_price[2]), float(args.jump_price[2])))

	if not (args.num_iterations is None):
		iterated_paras['num_iterations'] = [int(args.num_iterations)]

	if not (args.num_iterations_validation is None):
		iterated_paras['num_iterations_validation'] = [int(args.num_iterations_validation)]

	if not (args.number_neurons is None):
		iterated_paras['nn'] = [int(args.number_neurons)]

	if not (args.number_layers is None):
		iterated_paras['n_h'] = [int(args.number_layers)]

	iterated_paras['version'] = [version]

	print (iterated_paras)

	do_iterations(iterated_paras=iterated_paras)
	




