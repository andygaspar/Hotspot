from pathlib import Path

def print_allocation(allocation):
	s = ''
	for slot, name in allocation.items():
		
		s += str(name) + ' -> ' + str(slot) + ' ; '
		
	print (s)

# TODO: make it work for any version
def agent_file_name(nfp, nn=128, n_h=2, v='v1.0', nf_tot_game=10, n_a=3,
					game_type='single', game='jump', n_f_players=[], jp=0.,
					root_dir=None):
	

	file_name = root_file_name(nfp=nfp, nn=nn, n_h=n_h, v=v, nf_tot_game=nf_tot_game, n_a=n_a,
								game_type=game_type, game=game, n_f_players=n_f_players, jp=jp,
								root_dir=root_dir)

	if game_type=='multi':
		file_name = '{}/nfp{}'.format(file_name, nfp)
		
	return file_name

def root_file_name(nfp=None, nn=128, n_h=2, v='v1.0', nf_tot_game=10, n_a=3,
					game_type='single', game='jump', n_f_players=[], jp=0.,
					root_dir=None):
	
	if root_dir is None:
		root_dir = Path(__file__).resolve().parent.parent.parent / 'saved_policies'

	if game_type=='single':
		file_name = str(root_dir / "{}/nf{}_na{}_nn{}_nh{}_nfp{}_jp{}_{}".format(v, nf_tot_game, n_a, nn, n_h, nfp, jp, game))
	elif game_type=='multi':
		assert len(n_f_players)>1
		
		file_name = str(root_dir / "multi {}/nf{}_na{}_nn{}_nh{}_jp{}_nfp".format(v, nf_tot_game, n_a, nn, n_h, jp))
		for nfp in n_f_players:
			file_name += str(nfp) + '_'

		file_name += game
	else:
		raise Exception('Unrecognised game_type:', game_type)
		
	return file_name