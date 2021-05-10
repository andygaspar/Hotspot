#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1,'../..')

import argparse

from Hotspot.RL.game_trainer import ContinuousMGameTrainer
from Hotspot.libs.tools import loop, agent_file_name, root_file_name

def wrapper_training(paras={}):
	do_training(**paras)

def wrapper_training_dummy(paras={}):
	print (paras)

def do_training(n_f=10, n_h=2, n_a=3, n_f_players=[], nn=10, game='jump',
	num_iterations=8000, learning_rate=3e-4, num_iterations_validation=1000,
	version='v1.0', jp=0.):
	n_f_players = list(n_f_players)
	layers = tuple([nn]*n_h)
	trainer = ContinuousMGameTrainer()
	trainer.build_game(game=game, n_f=n_f, n_a=n_a, n_f_players=n_f_players)
	trainer.build_agents(critic_learning_rate = learning_rate,
						actor_learning_rate = learning_rate,
						alpha_learning_rate = learning_rate,
						target_update_tau = 0.005,
						target_update_period = 1,
						gamma = 0.99,
						reward_scale_factor = 100.0,
						actor_fc_layer_params = layers,
						critic_joint_fc_layer_params = layers)
	trainer.prepare_buffers(initial_collect_steps=10, batch_size=2048)
	trainer.train_agents(num_iterations=num_iterations, n_eval_setp=500)

	# name = "save_policies/multi v1.2/nf{}_na{}_nn{}_nfp".format(n_f, n_a, nn)
	# for nfp in n_f_players:
	# 	name += str(nfp) + '_'
		
	# name += game

	name = root_file_name(nn=nn,
						  n_h=n_h,
						  v=version,
						  nf_tot_game=n_f,
						  n_a=n_a,
						  game_type='multi',
						  game=game,
						  n_f_players=n_f_players,
						  jp=jp)

	file_names = [agent_file_name(nfp,
							  nn=nn,
							  n_h=n_h,
							  v=version,
							  nf_tot_game=n_f,
							  n_a=n_a,
							  game_type='multi',
							  game=game,
							  n_f_players=n_f_players,
							  jp=jp) for nfp in n_f_players]

	trainer.save_policies(file_names)
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
	parser = argparse.ArgumentParser(description='Mercury batch script', add_help=True)

	version = 'v1.3'
	
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

	parser.add_argument('-nfp1', '--number_flights_player1',
								help='Number of flights for player 1',
								required=False,
								default=None,
								nargs='*')

	parser.add_argument('-nfp2', '--number_flights_player2',
								help='Number of flights for player 2',
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

	args = parser.parse_args()

	print ()

	print (args)

	iterated_paras = {}

	# default steps
	nf_steps = 5
	na_steps = 1
	nfp_steps = 1

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

	if not (args.number_flights_player1 is None):
		nfp1 = int(args.number_flights_player1[0])
		if len(args.number_flights_player1)==1:
			nfp1s = [nfp1]
		elif len(args.number_flights_player1)==2:
			nfp1_end = int(args.number_flights_player1[1])

			nfp1s = list(range(nfp1, nfp1_end+1, nfp_steps))
		else:
			nfp1_end = int(args.number_flights_player1[1])
			nfp1_step = int(args.number_flights_player1[2])

			nfp1s = list(range(nfp1, nfp1_end+1, nfp1_step))
	else:
		nfp1s = [4]

	if not (args.number_flights_player2 is None):
		nfp2 = int(args.number_flights_player2[0])
		if len(args.number_flights_player2)==1:
			nfp2s = [nfp2]
		elif len(args.number_flights_player2)==2:
			nfp2_end = int(args.number_flights_player2[1])

			nfp2s = list(range(nfp2, nfp2_end+1, nfp_steps))
		else:
			nfp2_end = int(args.number_flights_player2[1])
			nfp2_step = int(args.number_flights_player2[2])

			nfps2 = list(range(nfp2, nfp2_end+1, nfp2_step))

	else:
		nfp2s = [3]

	iterated_paras['n_f_players'] = [(nf1, nf2) for nf1 in nfp1s for nf2 in nfp2s]

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
	




