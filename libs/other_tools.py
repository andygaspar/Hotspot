import sys
import contextlib
from pathlib import Path
import numpy as np
import pandas as pd


def print_allocation(allocation):
	# al_sorted = OrderedDict(sorted(allocation.items(), key=lambda t: t[1]))
	s = ''
	for flight, slot in allocation.items():
		s += str(slot) + ' -> ' + str(flight) + ' ; '

	print(s)


def compare_allocations(allocation1, allocation2):
	print('Comparison between allocations:')
	s = ''
	for i, (slot1, name1) in enumerate(allocation1.items()):
		name2 = allocation2[slot1]
		if name1 != name2:
			s += 'Slot {} : {} -> {}\n'.format(slot1, name1, name2)
		# print ('Slot', slot, ':', name1, '->', name2)
	if len(s) > 0:
		print(s)
	else:
		print('Allocations are the same!')


def compare_allocations_costs(allocation1, allocation2, flights, cost_vect):
	comparison = {}
	comparison['initial_cost'] = compute_cost(flights, allocation1, cost_vect)
	comparison['final_cost'] = compute_cost(flights, allocation2, cost_vect)

	return comparison


def generate_comparison(allocation1, allocation2, airlines, cost_vect):
	results = {'airline': [], 'num flights': [], 'initial costs': [], 'final costs': [], 'reduction %': []}
	for air, flights in airlines.items():
		c = compare_allocations_costs(allocation1, allocation2, flights, cost_vect)
		results['airline'].append(air)
		results['num flights'].append(len(flights))
		results['initial costs'].append(c['initial_cost'])
		results['final costs'].append(c['final_cost'])
		results['reduction %'].append(-100 * (c['final_cost'] - c['initial_cost']) / c['initial_cost']
									  if c['initial_cost'] > 0 else float('inf'))

	results = pd.DataFrame(results)

	results.loc['total', ['num flights', 'initial costs', 'final costs']] = results.sum(axis=0)[
		['num flights', 'initial costs', 'final costs']]
	results.loc['total', 'airline'] = 'total'

	results.loc['total', 'reduction %'] = -100 * (
			results.loc['total', 'final costs'] - results.loc['total', 'initial costs']) / results.loc[
											  'total', 'initial costs']

	return results.reset_index()


# TODO: make it work for any version
def agent_file_name(nfp, nn=128, n_h=2, v='v1.0', nf_tot_game=10, n_a=3,
					game_type='single', game='jump', n_f_players=[], jp=0.,
					root_dir=None):
	file_name = root_file_name(nfp=nfp, nn=nn, n_h=n_h, v=v, nf_tot_game=nf_tot_game, n_a=n_a,
							   game_type=game_type, game=game, n_f_players=n_f_players, jp=jp,
							   root_dir=root_dir)

	if game_type == 'multi':
		file_name = '{}/nfp{}'.format(file_name, nfp)

	return file_name


def root_file_name(nfp=None, nn=128, n_h=2, v='v1.0', nf_tot_game=10, n_a=3,
				   game_type='single', game='jump', n_f_players=[], jp=0.,
				   root_dir=None):
	if root_dir is None:
		root_dir = Path(__file__).resolve().parent.parent.parent / 'saved_policies'

	if game_type == 'single':
		file_name = str(
			root_dir / "{}/nf{}_na{}_nn{}_nh{}_nfp{}_jp{}_{}".format(v, nf_tot_game, n_a, nn, n_h, nfp, jp, game))
	elif game_type == 'multi':
		assert len(n_f_players) > 1

		file_name = str(root_dir / "multi {}/nf{}_na{}_nn{}_nh{}_jp{}_nfp".format(v, nf_tot_game, n_a, nn, n_h, jp))
		for nfp in n_f_players:
			file_name += str(nfp) + '_'

		file_name += game
	else:
		raise Exception('Unrecognised game_type:', game_type)

	return file_name


def compute_cost(flights, allocation, cost_vect):
	cost = sum([cost_vect[f][allocation[f].index] for f in flights])

	return cost


@contextlib.contextmanager
def write_on_file(name_file=None):
	if name_file!=None:
		with open(name_file, 'w') as f:
			save_stdout = sys.stdout
			sys.stdout = f
			yield
			sys.stdout = save_stdout
	else:
		stdout_backup = sys.stdout
		sys.stdout = None
		yield
		sys.stdout = stdout_backup


def get_first_matching_element(iterable, default = None, condition = lambda x: True):
	"""
	Returns the first item in the `iterable` that
	satisfies the `condition`.

	If the condition is not given, returns the first item of
	the iterable.

	If the `default` argument is given and the iterable is empty,
	or if it has no items matching the condition, the `default` argument
	is returned if it matches the condition.

	The `default` argument being None is the same as it not being given.

	Raises `StopIteration` if no item satisfying the condition is found
	and default is not given or doesn't satisfy the condition.

	>>> first( (1,2,3), condition=lambda x: x % 2 == 0)
	2
	>>> first(range(3, 100))
	3
	>>> first( () )
	Traceback (most recent call last):
	...
	StopIteration
	>>> first([], default=1)
	1
	>>> first([], default=1, condition=lambda x: x % 2 == 0)
	Traceback (most recent call last):
	...
	StopIteration
	>>> first([1,3,5], default=1, condition=lambda x: x % 2 == 0)
	Traceback (most recent call last):
	...
	StopIteration
	"""

	try:
		return next(x for x in iterable if condition(x))
	except StopIteration:
		if default is not None:# and condition(default):
			return default
		else:
			raise


def remove_nan_coupled_lists(list1, list2):
	list1 = np.array(list1)
	list2 = np.array(list2)
	mask = ~pd.isnull(list1) & ~pd.isnull(list2)
	list1 = list1[mask]
	list2 = list2[mask]

	return list1, list2


def sort_lists(list1, list2, remove_nan=False):
	"""
	Sort ith respect to values in list1
	"""
	if remove_nan:
		list1, list2 = remove_nan_coupled_lists(list1, list2)
	return zip(*sorted(zip(list1, list2), key=lambda pair: pair[0]))


def loop(a, level, parass, ret={}, thing_to_do=None, **args):
	"""
	Typical usage:

	paras = {'pouet':2, 'pouic':4}

	a = {'pouet':[0, 1], 'pouic':[10, 11]}

	args = {'paras':paras}

	def yep(paras={}):
		return paras['pouet'] + paras['pouic']

	level = ['pouet', 'pouic']
	loop(a, level, paras, thing_to_do=yep, **args)
	"""

	if level == []:
		return thing_to_do(**args)
	else:
		assert level[0] in a.keys()
		for i in a[level[0]]:
			print(level[0], '=', i)
			if not level[0] in parass.keys():
				raise Exception('Trying to update a key that does not exist.')

			parass.update({level[0]: i})

			ret[i] = loop(a, level[1:], parass, ret={}, thing_to_do=thing_to_do, **args)

	return ret
