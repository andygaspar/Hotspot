import sys
import contextlib
import os
import datetime

from scipy.optimize import minimize_scalar, curve_fit
import statsmodels.distributions.empirical_distribution as edf
from scipy.interpolate import interp1d
from numpy import *
import numpy as np
from numpy.random import randint, choice
from scipy.signal import argrelextrema
from scipy.stats import pearsonr
import networkx as nx
import imp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
#from mpl_toolkits.basemap import Basemap
from paramiko import SSHClient
from scp import SCPClient
import string
import multiprocessing as mp
from contextlib import contextmanager
from sshtunnel import SSHTunnelForwarder, open_tunnel
from sqlalchemy import create_engine
from itertools import permutations, islice
from math import radians, cos, sin, asin, sqrt, atan2
import pandas as pd
from mpmath import polyroots
from collections import OrderedDict
import pickle


R = 6373

"""
MYLIB VERSION
Still building... python 3 version of my previous general_tools module.
"""

legend_location={'ur':1, 'ul':2, 'll':3, 'lr':4, 'r':5, 'cl':6, 'cr':7, 'lc':8, 'uc':9, 'c':10}

#my_conv = { FIELD_TYPE.LONG: int, FIELD_TYPE.FLOAT: float, FIELD_TYPE.DOUBLE: float }

_colors=('Blue','BlueViolet','Brown','CadetBlue','Crimson','DarkMagenta','DarkRed','DeepPink','Gold','Green','OrangeRed')

def hex_to_rgb(value):
	value = value.lstrip('#')
	lv = len(value)
	rgb_255 = tuple(int(value[i:i+lv//3], 16) for i in range(0, lv, lv//3))
	return tuple(a/255. for a in rgb_255)

nice_colors = ['#348ABD',  '#7A68A6',  '#A60628',  '#467821',  '#CF4457',  '#188487',  '#E24A33']
nice_colors = [hex_to_rgb(v) for v in nice_colors]

def percentile_custom(n):
	def percentile_(x):
		return percentile(x, n)
	percentile_.__name__ = 'percentile_%s' % n
	return percentile_

def inverted_edf(x):
	"""
	Return eh inverted edf interpolated function from the values in x

	Input parameters:
	x: an array of values
	
	Return:
	The linear interpolation of the inversed edf of x

	Example of use:    
	x=[34, 21, 113, 153, 421, 235, 134, 21, 1, 43, 1234, 52, 235]
	iedf_x=inverted_edf(x)
	"""
	if len(set(x))>1:
		x_edf = edf.ECDF(x)
		
		slope_changes_x = sorted(set(x))
		
		edf_values_at_slope_changes_x = [ x_edf(item) for item in slope_changes_x]
		
		inverted_edf = interp1d(edf_values_at_slope_changes_x, slope_changes_x)
		
		return inverted_edf
	
	else:
		return None

def haversine(lon1, lat1, lon2, lat2):
	"""
	Calculate the great circle distance between two points 
	on the earth (specified in decimal degrees)
	----------
	Parameters
	----------
	lon1, lat1 : coordinates point 1 in degrees
	lon2, lat2 : coordinates point 2 in degrees
	-------
	Return
	------
	Distance between points in km
	"""
	# convert decimal degrees to radians 
	lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
	# haversine formula 
	dlon = lon2 - lon1 
	dlat = lat2 - lat1 
	a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
	#c = 2 * asin(sqrt(a)) 
	c= 2 * atan2(sqrt(a), sqrt(1-a))
	km = R * c
	return km

def intermetidate_point(lon1, lat1, lon2, lat2, fraction):
	d = haversine(lon1, lat1, lon2, lat2)

	# convert decimal degrees to radians 
	lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
	ad = d / R #angular distance

	a = sin((1-fraction)*ad) / sin(ad)
	b = sin(fraction*ad) / sin(ad)
	x = a * cos(lat1) * cos(lon1) + b * cos(lat2) * cos(lon2)
	y = a * cos(lat1) * sin(lon1) + b * cos(lat2) * sin(lon2)
	z = a * sin(lat1) + b * sin(lat2)

	lat = atan2(z, sqrt(x**2+y**2))
	lon = atan2(y, x)

	return (lon*180/math.pi, lat*180/math.pi)

def proportional(v1,v2,p1,p2,p):
	if p1==p2:
		v=v1
	else:
		v = v1 + (p-p1)*(v2-v1)/(p2-p1)
		
	return v

def create_dir(name):
	# TODO: modernise (see loading)
	if not os.path.exists(name):
		os.makedirs(name)

def loading(func):   #decorator
	"""
	This is useful when you want to compute something, but you might already have
	it on the disk, in which case you want to load it from it.

	How to use: when declaring a reading func, use:

	@loading
	def my_func(...):
		...

	Then when calling my_func, you can use four additional keywords:
	path: where is the file should be/is saved. Can be a string 
		or a list of two elements [rep, file]. Default 'pouet'
	save: if you want to actually save the data on the disk. Default True.
	force: if True, ignore the data on the disk, if present. Default False
	verbose_load: tune verbosity for this decorator. Default False.

	TODO: modernise.
	"""
	def wrapper(*args, **kwargs):
		if 'path' in kwargs.keys():
			if type(kwargs['path'])==type('p'):
				path=kwargs['path']
				rep=''
			elif type(kwargs['path'])==type(['p','p']):
				assert len(kwargs['path'])==2
				path= kwargs['path'][0] + '/' + kwargs['path'][1]
				rep=kwargs['path'][0]
		else:
			path = 'pouet' # A changer (mettre le nom de la fonction). TODO
		if 'force' in kwargs.keys():
			force = kwargs['force']
		else:
			force = False
		if 'save' in kwargs.keys():
			save = kwargs['save']
		else:
			save = True
		if 'verbose_load' in kwargs.keys():
			verbose_load = kwargs['verbose_load']
		else:
			verbose_load = False
		kwargs.pop('path', None)
		kwargs.pop('save', None)
		kwargs.pop('force', None)
		kwargs.pop('verbose_load', None)
		
		if os.path.exists(path) and not force:
			if verbose_load:
				print ('Loading from disk.')
			with open(path, 'rb') as f:
				something = pickle.load(f)
			return something
		else:
			if verbose_load:
				print ('Computing from scratch.')
			something = func(*args,**kwargs)
			if save:
				if rep!='':
					try:
						os.mkdir(rep)
					except:
						pass
				with open(path, 'wb') as f:
					pickle.dump(something,f)
				if verbose_load:
					print ('Saved in ', path)
			return something
	return wrapper

class DummyFile:
	def write(self, x): pass

	def flush(self): pass

@contextlib.contextmanager
def silence(silent):
	if silent:
		save_stdout = sys.stdout
		sys.stdout = DummyFile()
	try:
		yield
	except:
		raise
	finally:
		if silent:
			sys.stdout = save_stdout

@contextlib.contextmanager
def clock_time(message_before='', 
	message_after='executed in', print_function=print,
	oneline=False):

	if oneline:
		print_function(message_before, end="\r")
	else:
		print_function(message_before)
	start = datetime.datetime.now()
	yield
	elapsed = datetime.datetime.now() - start

	if oneline:
		message = ' '.join([message_before, message_after, str(elapsed)])
	else:
		message = ' '.join([message_after, str(elapsed)])
		
	print_function (message)

def timeit(f, listt):
	start=datetime.datetime.now()
	[f(l) for l in listt]
	elapsed = datetime.datetime.now() - start
	print ('Typical time of execution:', elapsed.total_seconds()/float(len(listt)))

def counter(i, end, start=0, message=''):
	sys.stdout.write('\r' + message + str(int(100*(abs(i-start)+1)/float(abs(end-start)))) + '%')
	sys.stdout.flush() 
	if i==end-1:
		print()

@contextlib.contextmanager
def write_on_file(name_file):
	if name_file!=None:
		with open(name_file, 'w') as f:
			save_stdout = sys.stdout
			sys.stdout = f
			yield
			sys.stdout = save_stdout
	else:
		yield

@contextlib.contextmanager
def logging(filename, mode='w'):
	"""
	Filename can be a stream, in which case it is return as it is.
	"""
	import _io
	if type(filename)==_io.TextIOWrapper:
		yield filename
	else:
		if filename is None:
			yield None
		else:
			f = open(filename, mode)
			yield f
			f.close()

def yes(question):
	ans = ''
	while not ans in ['Y','y','yes','Yes','N','n','No','no']:
		ans = input(question + ' (y/n)\n')
	return ans in ['Y','y','yes','Yes']

def recursive_minimization(f, bounds, n=100, depth=0, max_depth=5, target=-1, tol=0.001):
	#print ('Depth=', depth)
	x = arange(bounds[0], bounds[1], (bounds[1] - bounds[0])/float(n))
	values = array([f(xx) for xx in x])
	# print ('bounds', bounds)
	# print ('values', values)
	possible_minima_idx = argrelextrema(values, less, mode = 'wrap')[0]
	#print ('possible minima', [x[i] for i in possible_minima_idx])
	r_mins = [(x[max(0, i-2)] + bounds[0])/2. for i in possible_minima_idx]
	r_maxs = [(x[min(i+2, len(values)-1)] + bounds[1])/2. for i in possible_minima_idx]
	r_inits = [x[i] for i in possible_minima_idx]
	i = 0
	res = {'message':'No solution found.'}
	while i<len(r_mins) and res['message'] == 'No solution found.':
		#print ('i=', i)
		if f(r_mins[i])<f(r_inits[i])<f(r_maxs[i]):
			res = minimize_scalar(f, tol = tol, bracket = [r_mins[i], r_inits[i], r_maxs[i]])#, bounds=(10**(-8.), 1.))
		else:
			res = minimize_scalar(f, tol = tol, bracket = [r_mins[i], r_maxs[i]])
		#res = minimize_scalar(f, tol = 0.000001, bracket = [r_mins[i], r_inits[i], r_maxs[i]])#, bounds=(10**(-8.), 1.))
		#print (res)
		#print ((res['fun'],target),tol)
		if (res['fun']-target)<tol:
			res['message'] = 'Solution found.'
			#print ('FOUND')
			#return res
		else:
			if depth<max_depth:
				res = recursive_minimization(f, (r_mins[i], r_inits[i], r_maxs[i]), depth = depth +1, target = target, tol = tol)
			else:
				res = {'message':'No solution found.'}

		i+=1

		#if res['message'] == 'No solution found.'

	return res

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

	if level==[]:
		return thing_to_do(**args)#(paras, G)
		# return thing_to_do(**parass)#(paras, G)
	else:
		assert level[0] in a.keys()
		for i in a[level[0]]:
			print (level[0], '=', i)
			#parass.update(level[0], i)
			if not level[0] in parass.keys():
				raise Exception('Trying to update a key that does not exist.')

			parass.update({level[0]:i})
			
			#print (parass['sb']['r_div'], parass['firms']['r_s'])
			ret[i] = loop(a, level[1:], parass, ret={}, thing_to_do=thing_to_do, **args)

	return ret

def add_dicts(d1, d2):
	"""
	Method to add the values of both dictionnary.
	"""

	dd = {}
	for k, v in d1.items():
		dd[k] = v + d2.get(k, 0.)

	for k, v in d2.items():
		if not k in d1.keys():
			dd[k] = v

	return dd

# def minimize_hard_scalar_function(f, x_guess = 1., n_tries = 100, brackets = [0., 10.], target = -1, tol = 0.001):
#     # res ={'message':'No solution found.'}
#     # k = 0
#     # i_best = -1
#     # bounds = brackets
#     # r_mins, r_maxs = [], []
#     # #print (quick_view_function(f, arange(0.00001, 1., 0.01)))
#     # while k<n_tries and res['message'] == 'No solution found.':
#     #     r_mins, r_inits, r_maxs, bounds = self.quick_guess(f, bounds = bounds, tentative = k, last_bests = (r_mins, r_maxs))

#     #     kk=0
#     #     while kk<len(r_mins) and res['message'] == 'No solution found.':
#     #         res = minimize_scalar(f, tol = tol, bracket = [r_mins[kk], r_maxs[kk]])#, bounds=(10**(-8.), 1.))

#     #         if res['fun']<=1.:
#     #             res['message'] = 'Solution found.'
#     #         else:
#     #             res['message'] = 'No solution found.'
#     #             last_rmin = r_mins[0]
#     #             last_rmax = r_maxs[0]


#     #         print (res['message'])

#     #         kk+=1
#     #     k+=1

#     return recursive_minimization(f, brackets)

def shift_spatial_network(G, shiftx=0., shifty=0., key_coords='coords'):
	for n in G.nodes():
		G.node[n][key_coords][0] += shiftx
		G.node[n][key_coords][1] += shifty

	return G

def center_spatial_network(G, center=(0., 0.), key_coords='coords'):
	coin = [G.node[n][key_coords][0] for n in G.nodes()]
	pouet = [G.node[n][key_coords][1] for n in G.nodes()]
	min_x, max_x = min(coin), max(coin)
	min_y, max_y = min(pouet), max(pouet)

	current_center = array((min_x+max_x)/2., (min_y+max_y)/2.)

	shiftx, shifty = array(center) - current_center

	return shift_spatial_network(G, shiftx=shiftx, shifty=shifty, key_coords=key_coords)

def build_triangular(N, side=1., eps=10e-6):
	"""
	build a triangular lattice in a rectangle with N nodes along the abscissa (so 4*N**2 in total)
	
	Parameters
	==========
		N: int,
			number of nodes along the abscissa
		side: float
			size of a side (in meters, cm, etc.)
		eps: float,
			small number for the detection of overlapping nodes.

	Returns
	=======
		G: Networkx Graph.

	"""
	
	
	G = nx.Graph()   
	a = side/float(N+0.5) - eps
	n = 0
	j = 0
	while j*sqrt(3.)*a <= side:
		i=0
		while i*a <= side:
			G.add_node(n, coords=[i*a, j*sqrt(3.)*a]) #LUCA: node capacity added.
			n+=1
			if i*a + a/2 < side and  j*sqrt(3.)*a + (sqrt(3.)/2.)*a < side:
				G.add_node(n, coords=[i*a + a/2., j*sqrt(3.)*a + (sqrt(3.)/2.)*a]) #LUCA: node capacity added.
				n+=1
			i+=1
		j+=1
			
	for n in G.nodes():
		for m in G.nodes():
			if n!=m and abs(sqrt((G.node[n]['coords'][0] - G.node[m]['coords'][0])**2\
			+ (G.node[n]['coords'][1]- G.node[m]['coords'][1])**2) - a) <eps:
				G.add_edge(n,m)
	print(len(G.nodes()))
	
	return G
	
class Paras(dict):
	"""
	Class Paras
	===========
	Custom dictionnary used to update parameters in a controlled way.
	This class is useful in case of multiple iterations of simulations
	with sweeping parameters and more or less complex interdependances
	between variables.
	In case of simple utilisation with a single iteration or no sweeping,
	a simple dictionary is enough.
	The update process is based on the attribute 'update_priority', 'to_update'.
	The first one is a list of keys. First entries should be updated before updating 
	later ones.
	The second is a dictionary. Each value is a tuple (f, args) where f is function
	and args is a list of keys that the function takes as arguments. The function
	returns the value of the corresponding key. 
	Notes
	-----
	'update_priority' and 'to_update' could be merged in an sorted dictionary.

	Taken from ELSA ABM.
	"""
	
	def __init__(self, dic):
		for k,v in dic.items():
			self[k]=v
		self.to_update={}

	def update_with_levels(self, name_para, new_value):
		"""
		Updates the value with key name_para to new_value.
		Parameters
		----------
		name_para : string
			label of the parameter to be updated
		new_value : object
			new value of entry name_para of the dictionary.
		Notes
		-----
		Changed in 2.9.4: self.update_priority instead of update_priority.
		NOTE: WAS CALLED "update" PREVIOUSLY
		"""
		
		self[name_para] = new_value
		# Everything before level_of_priority_required should not be updated, given the para being updated.
		lvl = self.levels.get(name_para, len(self.update_priority)) #level_of_priority_required
		#print name_para, 'being updated'
		#print 'level of priority:', lvl, (lvl==len(update_priority))*'(no update)'
		for j in range(lvl, len(self.update_priority)):
			k = self.update_priority[j]
			(f, args) = self.to_update[k]
			vals = [self[a] for a in args] 
			self[k] = f(*vals)

	def analyse_dependance(self):
		"""
		Detect the first level of priority hit by a dependance in each parameter.
		Those who don't need any kind of update are not in the dictionnary.
		This should be used once when the 'update_priority' and 'to_update' are 
		finished.
		It computes the attribute 'levels', which is a dictionnary, whose values are 
		the parameters. The values are indices relative to update_priority at which 
		the update should begin when the parameter corresponding to key is changed. 
		"""

		# print 'Analysing dependances of the parameter with priorities', self.update_priority
		self.levels = {}
		for i, k in enumerate(self.update_priority):
			(f, args) = self.to_update[k]
			for arg in args:
				if arg not in self.levels.keys():
					self.levels[arg] = i

def read_paras(paras_file=None, post_process=None):
	"""
	Reads parameter file for a single simulation.    
	"""

	if paras_file is None:
		import my_paras as paras_mod
	else:
		paras_mod = imp.load_source("paras", paras_file)

	paras = paras_mod.paras

	if post_process!=None:
		paras = post_process_paras(paras)

	return paras

def sort_lists(list1, list2, remove_nan=False):
	"""
	Sort ith respect to values in list1
	"""
	if remove_nan:
		list1, list2 = remove_nan_coupled_lists(list1, list2)
	return zip(*sorted(zip(list1, list2), key = lambda pair: pair[0]))

def remove_nan_coupled_lists(list1, list2):
	list1 = array(list1)
	list2 = array(list2)
	mask = ~pd.isnull(list1) & ~pd.isnull(list2)
	list1 = list1[mask]
	list2 = list2[mask]

	return list1, list2

def fit(x, y, first_point=0, last_point=-1, f_fit=None,
		p0=None, bounds=(-inf, inf)):
	"""
	Simple function for linear fit.

	Parameters
	==========
	x: list or numpy array
		x-coordinate of points to fit
	y: list or numpy array
		y-coordinate of points to fit
	first_point: int, optional
		index of first point to consider
	last_point: int, optional
		index of last point to consider
	f_fit: function
		function for it. Should have the signature (x, a, b, ...), where 
		x is the variable and a, b etc the coefficients to be regressed.
		If None, a linear function a + b*x is used.
	p_0: iterable, optional
		array of initial guess for the coefficients
	condition: boolean, optional,
		If True, x and y are standarised before fit.

	Return
	======
	popt: numpy array
		of optimal coefficients
	pcov: numpy array
		covariance matrix of regression
	f_fit_opt: function
		function with optimal coefficients

	Notes
	=====
	nan values are discarded.
	Can plot output with 
	plot(x, y)
	plot(x, vectorize(f_fit_opt)(x))

	or:
	plot(x, y)
	x_cont = linspace(min(x), max(x), 100)
	plot(x_cont, vectorize(f_fit_opt)(x_cont))
	
	"""
	if f_fit is None:
		def f_fit(x, a, b):
			return a + b*x

	x, y = sort_lists(x, y, remove_nan=True)
	x, y = array(x), array(y)

	popt, pcov = curve_fit(f_fit,
							x[first_point:last_point],
							y[first_point:last_point],
							p0=p0,
							bounds=bounds)

	def f_fit_opt(x):
		return f_fit(x,*popt)

	return popt, pcov, f_fit_opt

def r_squared(y, y_fit):
	"""
	Returns the coefficient of determination of a fit.

	Parameters
	==========
	y: iterable,
		initial points
	y_fit: iteratble,
		fitted points

	Returns
	=======
	r: float
		coefficient of determination

	Notes
	=====
	Discard nan values.

	"""

	y_fit, y = remove_nan_coupled_lists(y_fit, y)

	y_bar = y.mean()
	SS_tot = sum((y-y_bar)**2)
	SS_res = sum((y-y_fit)**2)
	return 1. - SS_res/SS_tot

def bootstrap_test(sample1, sample2, k = 1000, p_value = 0.05, two_tailed = True):
	"""
	Test the null hypothesis that the two samples are independent from each other 
	thanks to pearson coefficients (two-tailed).
	Note that we keep nan values during the resampling (and eliminate them to compute 
	the pearson coefficient).

	TODO: one tail test!
	
	Remember: if True, the difference is NOT significant (the two samples are from
	the same distribution with confidence > 1-p_value).

	THIS IS PROBABLY USELESS, USE P-VALUE GIVEN BY pearsonr!
	"""
	# eliminate all entries which have a nan in one of the sample. 
	sample1_bis, sample2_bis = zip(*[zz for zz in zip(sample1, sample2) if not pd.isnull(zz[0]) and not pd.isnull(zz[1])])
	r_sample = pearsonr(sample1_bis, sample2_bis)[0]

	sample1_bis = array(sample1_bis)
	sample2_bis = array(sample2_bis)
	
	n = len(sample1_bis)
	try:
		assert n == len(sample2_bis)
	except AssertationError:
		raise Exception("Samples must have same sizes.")

	r_resample = zeros(k)
	for i in range(k):
		s1_rand = sample1_bis[randint(0, n, n)] # Resampling with the same size
		s2_rand = sample2_bis[randint(0, n, n)] 
		s1_rand_bis, s2_rand_bis = zip(*[zz for zz in zip(s1_rand, s2_rand) if not pd.isnull(zz[0]) and not pd.isnull(zz[1])])
		r_resample[i] = pearsonr(s1_rand_bis, s2_rand_bis)[0]
		
	ci = percentile(r_resample, [100.*p_value/2., 100.*(1.-p_value/2.)])
	
	#print ("Percentiles:"), ci
	
	return  ci[0]<r_sample<ci[1], ci

def bootstrap_mean_test(sample1, sample2, k=1000, p_value=0.05, two_tailed=True, replace=False):
	"""
	Test the null hypothesis that both samples have the same mean with confidence 1-p_value (two-tailed).
	
	Remember: if True, the test is passed, meaning that both samples have the same mean.
	In other word, the difference is significant if False is returned.
	"""

	if not two_tailed:
		raise Exception('One-tail test not implemented')
	# Shift both distributions by the overall  mean
	mean_total = array(list(sample1) + list(sample2)).mean()
	sample1_shifted = array(sample1) - mean_total + array(sample1).mean()
	sample2_shifted = array(sample2) - mean_total + array(sample2).mean()

	# Draw some means from shifted samples, with replacement	
	resampled_means1 = array([choice(sample1_shifted, size=len(sample1), replace=replace).mean() for i in range(k)])
	resampled_means2 = array([choice(sample2_shifted, size=len(sample2), replace=replace).mean() for i in range(k)])

	# Compare the resampled means with the empirical mean
	diff_mean_resampled = resampled_means1 - resampled_means2
	diff_mean_empirical = array(sample1).mean() - array(sample2).mean()

	ci = percentile(diff_mean_resampled, [100.*p_value/2., 100.*(1.-p_value/2.)])

	return ci[0]<diff_mean_empirical<ci[1], ci

def permutation_test(sample1, sample2, k=1000, p_value=0.05, two_tailed=True, low_mem=False):
	"""
	Permutation test: better than bootstrap in general.

	Remember: if True, the difference is NOT significant (the two means are from
	the same distribution with confidence > 1-p_value).

	If low_mem = True, computation will be slower (x2) but will used much less memory.
	"""
	if not two_tailed:
		raise Exception('One-tailed test not implemented yet')

	diff = array(sample1) - array(sample2)
	diff_mean_empirical = diff.mean()
	
	if not low_mem:
		draw = choice([1., -1], (k, len(diff)))

		diff_resampled = draw * diff
		dist_mean = diff_resampled.mean(axis=1)
	else:
		dist_mean = zeros(k)
		for kk in range(k):
			draw_b = choice([1., -1], len(sample1))

			diff_resampled_b = draw_b * diff

			dist_mean[kk] = diff_resampled_b.mean()

	ci = percentile(dist_mean, [100.*p_value/2., 100.*(1.-p_value/2.)])

	return ci[0]<=diff_mean_empirical<=ci[1], ci

def permutation_test_only_diff(diff_sample, k=1000, p_value=0.05, two_tailed=True, low_mem=False):
	"""
	Use in case you have only access to the differences in the 
	two sequences, not the sequences themselves.
	"""
	if not two_tailed:
		raise Exception('One-tailed test not implemented yet')

	diff = array(diff_sample)
	diff_mean_empirical = diff.mean()
	
	if not low_mem:
		draw = choice([1., -1], (k, len(diff)))

		diff_resampled = draw * diff
		dist_mean = diff_resampled.mean(axis=1)
	else:
		dist_mean = zeros(k)
		for kk in range(k):
			draw_b = choice([1., -1], len(diff))

			diff_resampled_b = draw_b * diff

			dist_mean[kk] = diff_resampled_b.mean()

	ci = percentile(dist_mean, [100.*p_value/2., 100.*(1.-p_value/2.)])

	return ci[0]<=diff_mean_empirical<=ci[1], ci

def human_format(num, latex=True):
	"""
	Formatting function for big numbers.
	"""
	num = float('{:.3g}'.format(num))
	magnitude = 0
	while abs(num) >= 1000:
		magnitude += 1
		num /= 1000.0
	if latex:
		return '${}{}$'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])
	else:
		return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

def human_format_for_ticks(x, pos):
	'The two args are the value and tick position'
	return human_format(x)

def draw_zonemap(x_min,y_min,x_max,y_max,res, continents_color='white', lake_color='white', sea_color='white',\
	 lw=0.8, draw_mer_par=True, color_coast_lines='#6D5F47', color_countries='#6D5F47'):
	m = Basemap(projection='gall',lon_0=0.,llcrnrlon=y_min,llcrnrlat=x_min,urcrnrlon=y_max,urcrnrlat=x_max,resolution=res)
	m.drawmapboundary(fill_color=sea_color) #set a background colour
	m.fillcontinents(color=continents_color, lake_color=lake_color)  # #85A6D9')
	m.drawcoastlines(color=color_coast_lines, linewidth=lw)
	m.drawcountries(color=color_countries, linewidth=lw)
	if draw_mer_par:
		m.drawmeridians(arange(-180, 180, 5), color='#bbbbbb')
		m.drawparallels(arange(-90, 90, 5), color='#bbbbbb')
	return m

def simple_color_map_function(color1, color2, min_value=0., max_value=1.):
	"""
	Build a function for simple linear interpolation between two colors.
	"""
	if type(color1) in [str]:
		color1 = hex_to_rgb(color1)
	if type(color2) in [str]:
		color2 = hex_to_rgb(color2)

	def f(value):
		norm_value = (float(max_value)-float(value))/(float(max_value)-float(min_value))
		avg = average(array([color1, color2]), axis=0, weights=[norm_value, 1. - norm_value])
		return clip(avg, 0., 1.)
	return f

def simple_colormap_object(cmap_f, min_value=0., max_value=1., k=100):
	"""
	This is designed to used in conjunction with the above function

	To make the color map legend, do:
	cbar = fig.colorbar(sm) # use the right-hand side with subplots, ax=axes[:])
	cbar.ax.set_ylabel('Statistical significance')
	"""
	cmap, norm = mcolors.from_levels_and_colors(linspace(min_value, max_value, k+1),
												[cmap_f(hue) for hue in linspace(min_value, max_value, k)])
	#sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
	sm.set_array([])

	return sm

def _pre_compute_positions(G, color_nodes=nice_colors[1], first_col_map_node=nice_colors[1], second_col_map_node=nice_colors[2], limits=(0,0,0,0), size_nodes=1., size_edges=2., nodes=[], edges=True, 
	multilayer=False, key_word_weight='weight', z_order_nodes=6, diff_edges=False, coords_in_minutes=True, enlarge_limits=0.,
	shift_multilayer=0.1, colors_multilayer=nice_colors, lower_value_color='min', highest_value_color='max'):
	
	"""
	Used for plotting different maps.
	NOTHING IS PROJECTED HERE.

	Coordinates should be lat, lon, not the reverse.
	"""

	if coords_in_minutes:
		unit = 60.
	else:
		unit = 1.

	def width_double(x, minn, maxx, min_final=0.5, max_final=2.):
		#return scale*x/maxx#+0.02#0.2
		return max_final*(x-minn)/(maxx-minn) + min_final

	def width(x, maxx, scale=0.5):
		return scale*x/maxx#+0.02#0.2
	
	restrict_nodes = True
	if nodes == []:
		nodes = G.nodes()
	if limits==(0, 0, 0, 0):
		limits = (min([G.node[n]['coord'][0]/unit for n in nodes]) - enlarge_limits,
				min([G.node[n]['coord'][1]/unit for n in nodes]) - enlarge_limits,
				max([G.node[n]['coord'][0]/unit for n in nodes]) + enlarge_limits,
				max([G.node[n]['coord'][1]/unit for n in nodes]) + enlarge_limits)
		restrict_nodes = False

	if nodes==[]:
		if restrict_nodes:
			# Restrict nodes to geometrical extent of the zone.
			nodes = [n for n in G.nodes() if limits[0]-0.2<=G.node[n]['coord'][0]/unit<=limits[2]+0.2 and limits[1]-0.2<=G.node[n]['coord'][1]/unit<=limits[3]+0.2]
		else:
			nodes = G.nodes()

	if type(color_nodes)==dict:
		color_nodes = [color_nodes[n] for n in nodes]
	elif type(color_nodes) in [unicode, str]:
		if color_nodes=='strength':
			values = [G.degree(n, weight=key_word_weight) for n in nodes]
			if lower_value_color=='min':
				min_value = min(values)
			else:
				min_value = lower_value_color
			if highest_value_color=='max':
				max_value = max(values)
			else:
				max_value = highest_value_color
			map_col = simple_color_map_function(first_col_map_node, second_col_map_node, min_value, max_value)
			color_nodes = [map_col(G.degree(n, weight=key_word_weight)) for n in nodes]
		elif color_nodes=='degree':
			values=[G.degree(n) for n in nodes]
			if lower_value_color=='min':
				min_value = min(values)
			else:
				min_value = lower_value_color
			if highest_value_color=='max':
				max_value = max(values)
			else:
				max_value = highest_value_color
			map_col = simple_color_map_function(first_col_map_node, second_col_map_node, min_value, max_value)
			color_nodes = [map_col(G.degree(n)) for n in nodes]
		elif type(color_nodes) in [str, unicode]:
			values = [G.node[n][color_nodes] for n in nodes]
			if lower_value_color=='min':
				min_value = min(values)
			else:
				min_value = lower_value_color
			if highest_value_color=='max':
				max_value = max(values)
			else:
				max_value = highest_value_color
			map_col = simple_color_map_function(first_col_map_node, second_col_map_node, min_value, max_value)
			color_nodes = [map_col(G.node[n][color_nodes]) for n in nodes]
		else:
			Exception("The following size function is not implemented:" + color_nodes)

	if type(z_order_nodes)==dict:
		z_order_nodes = [z_order_nodes[n] for n in nodes]

	if type(size_nodes) in [int, float]:
		size_nodes = [size_nodes for n in nodes]
	elif size_nodes==[]:
		size_nodes = [1 for n in nodes]
	elif type(size_nodes)==tuple:
		if size_nodes[0]=='strength':
			size_nodes=[G.degree(n, weight=key_word_weight)*size_nodes[1] for n in nodes]
		elif size_nodes[0]=='degree':
			size_nodes=[G.degree(n)*size_nodes[1] for n in nodes]
		elif type(size_nodes[0]) in [str, unicode]:
			size_nodes = [G.node[n][size_nodes[0]]*size_nodes[1] for n in nodes]
		else:
			Exception("The following size function is not implemented:" + size_nodes)
	elif type(size_nodes)==dict:
		size_nodes=[size_nodes[n] for n in nodes]
		
	edges_positions, edges_width, edges_colors = [], [], []
	if edges:
		if not multilayer:
			if key_word_weight!=None:
				max_wei = max([abs(G[e[0]][e[1]][key_word_weight]) for e in G.edges() if e[0] in nodes and e[1] in nodes])
				min_wei = min([abs(G[e[0]][e[1]][key_word_weight]) for e in G.edges() if e[0] in nodes and e[1] in nodes])
			for e in G.edges():
				if e[0] in nodes and e[1] in nodes:
					if diff_edges:
						color = nice_colors[2] if G[e[0]][e[1]][key_word_weight]>0 else nice_colors[0]
					else:
						color = 'k'
					#xe1,ye1 = m(G.node[e[0]]['coord'][1]/unit,G.node[e[0]]['coord'][0]/unit)
					xe1, ye1 = (G.node[e[0]]['coord'][0]/unit, G.node[e[0]]['coord'][1]/unit)
					#xe2,ye2 = m(G.node[e[1]]['coord'][1]/unit,G.node[e[1]]['coord'][0]/unit)
					xe2,ye2 = (G.node[e[1]]['coord'][0]/unit, G.node[e[1]]['coord'][1]/unit)
					edges_positions.append(([xe1, xe2], [ye1, ye2]))
					if key_word_weight!=None:
						if type(size_edges)==tuple:
							edges_width.append(width_double(G[e[0]][e[1]][key_word_weight], min_wei, max_wei, min_final=size_edges[0], max_final=size_edges[1]))
						else:
							edges_width.append(width(G[e[0]][e[1]][key_word_weight], max_wei, scale=size_edges))
					else:
						edges_width.append(size_edges)
					edges_colors.append(color)
		else:
			color_multi = {l:colors_multilayer[i%len(colors_multilayer)] for i, l in enumerate(G.layers)}
			if key_word_weight!=None:
				pouet = [abs(G[n1][n2][l][key_word_weight]) for n1, n2, l in G.edges(layer='all') if n1 in nodes and n2 in nodes]
				max_wei = max(pouet)
				min_wei = min(pouet)
			for n1, n2 in G.edges():
				for i, l in enumerate(G[n1][n2].keys()):
					if n1 in nodes and n2 in nodes:
						color = color_multi[l]
						xe1, ye1 = (G.node[n1]['coord'][0]/unit, G.node[n1]['coord'][1]/unit)
						xe2, ye2 = (G.node[n2]['coord'][0]/unit, G.node[n2]['coord'][1]/unit)
						o = array((xe1, ye1))
						d = array((xe2, ye2))
						if len(G[n1][n2].keys())>1:
							n = array((-ye2+ye1, xe2-xe1))# normal to edge
							n = n/linalg.norm(n)
							shift = 2.*shift_multilayer * i / (len(G[n1][n2].keys())-1.) - shift_multilayer
							o += n * shift
							d += n * shift

						edges_positions.append(([o[0], d[0]], [o[1], d[1]]))

						if key_word_weight!=None:
							if type(size_edges)==tuple:
								edges_width.append(width_double(G[n1][n2][l][key_word_weight], min_wei, max_wei, min_final=size_edges[0], max_final=size_edges[1]))
							else:
								edges_width.append(width(G[n1][n2][l][key_word_weight], max_wei, scale=size_edges))
						else:
							edges_width.append(size_edges)
						edges_colors.append(color)
		
	return nodes, limits, color_nodes, z_order_nodes, size_nodes, edges_positions, edges_width, edges_colors

def map_of_net(G, color_nodes=nice_colors[1], first_col_map_node=nice_colors[1], second_col_map_node=nice_colors[2], limits=(0,0,0,0), title='', size_nodes=1., size_edges=2., nodes=[], zone_geo=[], edges=True, fmt='svg', dpi=100, \
	save_file=None, show=True, figsize=(9,6), continents_color='white', sea_color='white', key_word_weight=None, z_order_nodes=6, diff_edges=False, lw_map=0.8,\
	draw_mer_par=True, ax=None, coords_in_minutes=True, use_basemap=False, split_nodes_by=0., enlarge_limits=0., color_coast_lines='#6D5F47', color_countries='#6D5F47',\
	shift_multilayer=0.1, colors_multilayer=nice_colors, multilayer=False, bbox_inches='tight', alpha_edges=1., antialiased=False, node_edgecolor='w', alpha_nodes=1.,\
	node_contour_width=1., lower_value_color='min', delete_fig=False, highest_value_color='max'):

	"""
	limites is (lat_min, lon_min, lat_max, lon_max)
	"""
	
	nodes, \
	(x_min, y_min, x_max, y_max), \
	color_nodes, \
	z_order_nodes, \
	size_nodes, \
	edges_positions, \
	edge_width, \
	edges_colors = _pre_compute_positions(G, 
										color_nodes=color_nodes, 
										first_col_map_node=first_col_map_node,
										second_col_map_node=second_col_map_node,
										limits=limits, 
										size_nodes=size_nodes, 
										nodes=nodes, 
										edges=edges, 
										key_word_weight=key_word_weight, 
										z_order_nodes=z_order_nodes, 
										diff_edges=diff_edges, 
										coords_in_minutes=coords_in_minutes, 
										size_edges=size_edges,
										enlarge_limits=enlarge_limits,
										shift_multilayer=shift_multilayer,
										colors_multilayer=colors_multilayer,
										multilayer=multilayer,
										lower_value_color=lower_value_color,
										highest_value_color=highest_value_color)

	if ax==None:
		fig = plt.figure(figsize=figsize)
		ax = fig.add_subplot(111)
		#ax = plt.subplot()
		ax.set_aspect(figsize[1]/figsize[0])
		ax.set_title(title)
	
	# Make basemap
	if not use_basemap:
		def m(a, b):
			return b, a
		ax.set_xlim((x_min, x_max))
		ax.set_ylim((y_min, y_max))
	else:
		m = draw_zonemap(x_min, y_min, x_max, y_max, 
			'i', 
			sea_color=sea_color, 
			continents_color=continents_color, 
			lake_color=sea_color,
			lw=lw_map, 
			draw_mer_par=draw_mer_par,
			color_coast_lines=color_coast_lines,
			color_countries=color_countries)
	# Convert coordinates
	if split_nodes_by>0.:
		x, y = split_coords(G, nodes, r=split_nodes_by)
	else:
		x, y = zip(*[G.node[n]['coord'] for n in nodes])

	x, y = m(y,x)

	# Draw nodes
	if node_edgecolor=='same':
		node_edgecolor = color_nodes
	try:
		ax.scatter(x, y, marker='o', zorder=z_order_nodes, s=size_nodes, c=color_nodes, lw=node_contour_width, edgecolor=node_edgecolor, alpha=alpha_nodes)#,cmap=my_cmap)
	except:
		print ('color_nodes')
		print (color_nodes)
		raise

	# Draw edges
	for i, (xxs, yys) in enumerate(edges_positions):
		xxs, yys = m(yys, xxs)
		lw = edge_width[i]
		cl = edges_colors[i]
		ax.plot(xxs, yys, '-', lw=lw, color=cl, zorder=4, alpha=alpha_edges, antialiased=antialiased)

	if zone_geo!=[]:
		patch = PolygonPatch(adapt_shape_to_map(zone_geo, m), facecolor='grey', edgecolor='grey', alpha=0.08, zorder=3)#edgecolor='grey', alpha=0.08,zorder=3)
		ax.add_patch(patch)

	if save_file!=None:
		plt.savefig(save_file + '.' + fmt, dpi=dpi, bbox_inches=bbox_inches)
		print ('Figure saved as', save_file + '.' + fmt)
	if show:
		plt.show()

	if delete_fig:
		plt.close("all")
	else:
		return ax

#########################################################################################

"""
These three functions allow to convert a nested dictionary into a multi-indexed dataframe.
Typical usage:
df = recursive_concat(results)
"""
def make_df_with_3_levels(user_dict):
	frames = []
	user_ids = []
	for user_id, d in user_dict.iteritems():
		user_ids.append(user_id)
		frames.append(pd.DataFrame.from_dict(d, orient='index'))

	return pd.concat(frames, keys=user_ids)

def dict_depth(d, depth=0):
	if not isinstance(d, dict) or not d:
		return depth
	return max(dict_depth(v, depth+1) for k, v in d.iteritems())

def recursive_concat(dic):
	if dict_depth(dic)==3:
		return make_df_with_3_levels(dic)
	else:
		frames = []
		user_ids = []
		for user_id, dic2 in dic.items():
			user_ids.append(user_id)
			frames.append(recursive_concat(dic2))
		
		return pd.concat(frames, keys=user_ids)

# Alternative when the Dictionaries as only three levels (two keys and one value)

def make_df_with_3_levels_quick(user_dict, rename_column_dict=None, rename_indexes=None):
	df = pd.Panel(user_dict).to_frame().unstack().T

	if rename_indexes:
		df.index.set_names(rename_indexes, inplace=True)

	df.rename(columns=rename_column_dict, inplace=True)

	return df

# Inverse
def make_nested_dict_from_df(df):
	return df.groupby(level=0).apply(lambda df: df.xs(df.name).to_dict()).to_dict()


#########################################################################################

def show_dic(dic):
	for k, v in dic.items():
		if type(v) in [float, np.float64]:
			print (k, ":", '{:3.2f}'.format(v))
		else:
			print (k, ":", v)

def alphabet(length):
	# Generate 'A', B', then 'AA', 'BB', etc.
	letters = list(string.ascii_uppercase)
	while len(letters)<length:
		letters.extend([i+b for i in letters for b in letters])

	return letters[:length]

# This is for parallel computation (independent runs).
# Typical use: 
# Take a function do((A, B, C))
# take a list of inputs, e.g. inputs = [(a1, b1, c1), (a2, b2, c2), (a3, b3, c3)]
# Compute result with: results_list = parmap(do, inputs)
# UNTESTED YET FOR PYTHON3

# def spawn(f):
#     def fun(pipe, x):
#         pipe.send(f(x))
#         pipe.close()
#     return fun

# def parmap(f,X):
#     pipe = [Pipe() for x in X]
#     proc = [Process(target=spawn(f), args=(c,x)) for x,(p,c) in izip(X, pipe)]
#     [p.start() for p in proc]
#     [p.join() for p in proc]
#     return [p.recv() for (p,c) in pipe]

# New stuff 
def parallelize_old(f, inputs, nprocs=None):
	"""
	Parameters
	==========
	f: function,
		e.g. with calling signature (A, B, C)
	inputs: iterable,
		list of inputs for function, e.g. [(a1, b1, c1), (a2, b2, c2), (a3, b3, c3)]
	nprocs: int,
		number of processes to create. If None, max number of processes.
	"""

	pool = mp.Pool(nprocs)

	results = pool.starmap(f, inputs)

	return results

from itertools import repeat

def parallelize(f, args=None, kwargs=None, nprocs=None):
	"""
	Parameters
	==========
	f: function,
		e.g. with calling signature (A, B, C=None, D=None)
	args: iterable,
		list of inputs for function, e.g. [(a1, b1), (a2, b2), (a3, b3)]
	kwargs: iterable,
		list of inputs for function, e.g. [{'C':20, 'D':25}, {'C':30, 'D':40}]
	nprocs: int,
		number of processes to create. If None, max number of processes.
	"""
	
	if kwargs is None:
		kwargs = [{}]*len(args)

	if args is None:
		args = [()]*len(kwargs)

	pool = mp.Pool(nprocs)
	
	results = starmap_with_kwargs(pool, f, args, kwargs)

	return results

def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
	# taken from https://stackoverflow.com/questions/45718523/pass-kwargs-to-starmap-while-using-pool-in-python

	args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)

	return pool.starmap(apply_args_and_kwargs, args_for_starmap)

def apply_args_and_kwargs(fn, args, kwargs):
	return fn(*args, **kwargs)

def fun(f, q_in, q_out):
	while True:
		i, x = q_in.get()
		if i is None:
			break
		q_out.put((i, f(*x)))

def function_with_queues(f,q_in, q_out):
	while True:
		data = q_in.get()
		i = data[0]
		data = data[1]
		if data == None:
			break
		res = f(*data)
		q_out.put((i,res))

def parmap(f, X, nprocs=mp.cpu_count()):
	"""
	Typical use:
	inputs = [(a, ) for a in aa]
	results = parmap(f, inputs)
 
	where f(a) for instance.
	"""
	#Create input and output queues
	q_in = mp.Queue()
	q_out = mp.Queue()

	#Launch processes with queu in an out
	proc = [mp.Process(target=function_with_queues, args=(f, q_in, q_out))
			for _ in range(nprocs)]

	[p.start() for p in proc]
	
	#Producer of input
	[q_in.put((i,item)) for i,item in enumerate(X)] #Put i so that we can keep track of order to sort at return
	[q_in.put((None,None)) for _ in range(nprocs)] #Put None at the end so that the consummer finish

	#Close the queues
	q_in.close()
	q_in.join_thread()

	#Join the processes (wait for them to finish)
	[p.join() for p in proc]
	

	#Read results
	res=[]
	while not q_out.empty():
		res = res+[q_out.get()]

	#Close out queue
	q_out.close()
	q_out.join_thread()

	#Return results ordered
	return [x for i, x in sorted(res)]

def parmap2(f, X, nprocs=mp.cpu_count()):
	"""
	Typical use:
	inputs = [(a, ) for a in aa]
	results = parmap(f, inputs)
	where f(a) for instance.
	"""

	q_in = mp.Queue(1)
	q_out = mp.Queue()

	proc = [mp.Process(target=fun, args=(f, q_in, q_out))
			for _ in range(nprocs)]
	for p in proc:
		p.daemon = True
		p.start()

	sent = [q_in.put((i, x)) for i, x in enumerate(X)]
	[q_in.put((None, None)) for _ in range(nprocs)]
	res = [q_out.get() for _ in range(len(sent))]

	[p.join() for p in proc]

	return [x for i, x in sorted(res)]

def spread_integer(n, n_cap):
	X = []
	n_still = n
	while n_still>0.:
		X += [int(ceil(n_still/(n_cap-len(X))))]
		
		n_still -= X[-1]

	return X

@contextmanager
def ssh_client_connection(connection=None, ssh_hostname=None, ssh_username=None, ssh_password=None, ssh_pkey=None,
						ssh_key_password=''):
	"""
	If connection is given, simply returns connection.

	Note: This is incompatible with ssh_tunnel_connection!
	The latter uses ssh_tunnel, whereas this one uses
	paramiko
	"""

	kill_connection = False
	if connection is None:
		kill_connection = True
		connection = SSHClient()
		connection.load_system_host_keys()

		if not 'ssh_password' is None:
			connection.connect(hostname=ssh_hostname,
								username=ssh_username,
								password=ssh_password)
		else:
			connection.connect(hostname=ssh_hostname,
								username=ssh_username,
								key_filename=ssh_pkey,
								passphrase=ssh_key_password)

	try:
		yield connection
	finally:
		#print("closing connection")
		if kill_connection:
			connection.close()         
			
def ssh_copy(f,t,ssh):
	with SCPClient(ssh.get_transport()) as scp:
		scp.put(f,t)

def ssh_tunnel_connection(ssh_parameters, hostname, port=22, allow_agent=False, debug_level=0):
	if 'ssh_password' in ssh_parameters.keys():
		ssh_tunnel = open_tunnel(ssh_parameters.get('ssh_hostname'),
										ssh_username=ssh_parameters.get('ssh_username'),
										ssh_password=ssh_parameters.get('ssh_password'),
										allow_agent=allow_agent,
										remote_bind_address=(hostname, port),
										debug_level=debug_level,
										set_keepalive=20.
										)
	else:
		ssh_tunnel = open_tunnel(ssh_parameters.get('ssh_hostname'),
										ssh_username=ssh_parameters.get('ssh_username'),
										ssh_pkey=ssh_parameters.get('ssh_pkey'),
										ssh_private_key_password=ssh_parameters.get('ssh_key_password',''),
										allow_agent=allow_agent,
										remote_bind_address=(hostname, port),
										debug_level=debug_level,
										set_keepalive=20.
										)
	ssh_tunnel.start()

	return ssh_tunnel

@contextmanager
def mysql_server(engine=None, hostname=None, port=None, username=None, password=None, database=None,
	connector='mysqldb', ssh_parameters=None, allow_agent=False, ssh_tunnel=None,
	debug_level = 0):
	"""
	If engine is given, yield simply engine.
	"""

	kill_engine = False
	if engine is None:
		kill_engine = True
		if ssh_tunnel is None and ssh_parameters is None:
			engine = create_engine('mysql+' + connector + '://' + username + ':' + password + '@' + hostname + '/' + database)
		else:
			if ssh_tunnel is None:
				ssh_tunnel = ssh_tunnel_connection(ssh_parameters,hostname,port,allow_agent,debug_level)
				engine = create_engine('mysql+' + connector + '://' + username + ':' + password + '@127.0.0.1:%s/' % ssh_tunnel.local_bind_port + database)

	try:
		yield {'engine':engine, 'ssh_tunnel':ssh_tunnel}
	finally:
		if kill_engine:
			engine.dispose()
			if not ssh_tunnel is None:
				ssh_tunnel.stop()

# Two way dict
class TwoWayDict(dict):
	def add(self, item1, item2):
		# Preferred methods
		# Remove any previous connections with these values
		if item1 in self:
			del self[item1]
		if item2 in self:
			del self[item2]
		dict.__setitem__(self, item1, item2)
		dict.__setitem__(self, item2, item1)

	def __setitem__(self, key, value):
		# Remove any previous connections with these values
		if key in self:
			del self[key]
		if value in self:
			del self[value]
		dict.__setitem__(self, key, value)
		dict.__setitem__(self, value, key)

	def __delitem__(self, key):
		dict.__delitem__(self, self[key])
		dict.__delitem__(self, key)

	def __len__(self):
		"""Returns the number of connections"""
		return dict.__len__(self) // 2


def build_step_multi_valued_function(df, name_min_col='delay_min_minutes', name_max_col='delay_max_minutes',
						add_lower_bound=None, add_upper_bound=None, value_lower_bound=0.,
					   value_upper_bound=99999., columns=None):
	if columns is None:
		columns = [col for col in df.columns if not col in [name_min_col, name_max_col]]
	
	# build bins
	mins = sorted(df[name_min_col])
	if not add_lower_bound is None:
		mins = [add_lower_bound] + mins
		
	maxs = sorted(df[name_max_col])
	if not add_upper_bound is None:
		maxs += [add_upper_bound]
		
	# Values
	if not add_lower_bound is None:
		values = {i+1:{k:v for k, v in dict(row).items() if k in columns} for i, row in df.iterrows()}
		values[0] = {col:value_lower_bound for col in columns}
	else:
		values = {i:{k:v for k, v in dict(row).items() if k in columns} for i, row in df.iterrows()}
			
	if not add_upper_bound is None:
		values[max(values.keys())+1] = {col:value_upper_bound for col in columns}

	def f(x, col):
		it = (i for i, v in enumerate(mins) if x  < v)
		idx = max(0, next(it, len(values)) - 1)

		return values[idx][col]
	
	return f

def build_step_bivariate_function(df,
	name_min_col1='flight_type_distance_gcd_km_min',
	name_max_col1='flight_type_distance_gcd_km_max',
	name_min_col2='delay_min_minutes',
	name_max_col2='delay_max_minutes',
	value_column='compensation',
	add_lower_bound1=None, add_upper_bound1=None,
	value_lower_bound1=0., value_upper_bound1=99999.,
	add_lower_bound2=None, add_upper_bound2=None,
	value_lower_bound2=0., value_upper_bound2=99999.):

	if value_lower_bound1>value_upper_bound1:
		value_upper_bound1 = value_lower_bound1 
	if value_lower_bound2>value_upper_bound2:
		value_upper_bound2 = value_lower_bound2
	
	# build bins
	values = OrderedDict()
	mins1 = sorted(list(set(df[name_min_col1])))
	if not add_lower_bound1 is None:
		mins1 = [add_lower_bound1] + mins1
		
	for v1 in mins1:
		#print ('v1=', v1)
		values[v1] = OrderedDict()
		
		#print (mask1)
		mins2 = sorted(list(set(df[name_min_col2])))
		if not add_lower_bound2 is None:
			mins2 = [add_lower_bound2] + mins2
			
		#print (mins2)
			
		for i, v2 in enumerate(mins2):
			if (not add_lower_bound2 is None) and i==0:
				values[v1][v2] = value_lower_bound2
			else:
				#print ('v2=', v2)
				mask = (df[name_min_col1]==v1) & (df[name_min_col2]==v2)
				#print (mask)
				dff = df[mask]
				if len(dff)>0:
					values[v1][v2] = dff.iloc[0][value_column]
				
	#print (values)

	def f(x1, x2):
		# print ('x1, x2', x1, x2)
		#print ([type(a) for a in list(values.keys())])
		# print (list(values.keys()))
		it = (i for i, v in enumerate(values.keys()) if x1 < v)
		# print (list(((i, v) for i, v in enumerate(values.keys()) if x1 < v)))
		idx1 = max(0, next(it, len(values.keys())) - 1)
		# print ('idx1=', idx1)
		v1 = list(values.keys())[idx1]
		
		# print ('v1=', v1)
		# print (values[v1])
		# print (list(values[v1].keys()))
		
		it2 = (i for i, v in enumerate(values[v1].keys()) if x2 < v)
		#print ([i for i, v in enumerate(values[v1].keys()) if x2 < v])
		#print ('aoin', next(it, len(values[v1].keys())))
		coin = next(it2, len(values[v1].keys()))
		#print ('coin', coin)
		idx2 = max(0, coin-1) # max(0, coin - 2)
		# if idx2<0:
		# 	raise Exception('No value below:', list(values[v1].keys())[0])
		#print ('idx2=', idx2)
		v2 = list(values[v1].keys())[idx2]

		#print (values[v1][v2])
	
		return values[v1][v2]
	
	return f

def sort_paired_values(x, y):
	return list(zip(*sorted(zip(x, y), key=lambda x:x[0])))

def inv_mu_sig_lognorm(mu_p, sig_p):
	"""
	Given the desired mu_p and sig_p, and mean and std of a lognorm
	distribution, returns mu and sig of the underlying normal 
	distribution.
	Note: mu_p should be strictly positive.
	"""

	A = 1+sig_p**2/mu_p**2

	mu = log(mu_p) - 0.5 * log(A)
	sig = sqrt(log(A))

	return mu, sig

def inv_s_scale_lognorm(mu_p, sig_p):
	"""
	Idem but direcly for scipy function
	"""
	mu, sig = inv_mu_sig_lognorm(mu_p, sig_p)

	s = sig
	scale = exp(mu)

	return s, scale

def sol_Bs(sig_p, M_p):
	t = (sig_p/M_p)**2
	
	sols = polyroots([1,1,-t,t])
	sols = [float(p.real) for p in sols if p.imag==0. and p>1.]
	
	if len(sols)>0:
		sol = sorted(sols)[0]
	else:
		sol=nan
	return sol

def A_B(sig_p, M_p):
	X = sol_Bs(sig_p, M_p)
	B = X**2
	A = M_p/(1.-X)
	
	return A, B
	
def inv_s_loc_scale_lognorm(mu_p, sig_p, M_p):
	"""
	Given mean, sig, and median desired for the lognormal distribution, 
	returns the s, loc, and scale parameters for the scipy distribution.
	"""

	A, B = A_B(sig_p, M_p)
	
	s = sqrt(log(B))
	scale = A
	loc = - A * sqrt(B) + mu_p
	
	return s, loc, scale

def partial_corr(C, corr_type='pearson'):
	"""
	Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
	for the remaining variables in C.
	Parameters
	----------
	C : array-like, shape (n, p)
		Array with the different variables. Each column of C is taken as a variable
	Returns
	-------
	P : array-like, shape (p, p)
		P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
		for the remaining variables in C.
	"""
	if corr_type=='pearson':
		corr_func = stats.pearsonr
	elif corr_type=='spearman':
		corr_func = stats.spearmanr
	else:
		raise Exception()
	
	C = np.asarray(C)
	p = C.shape[1]
	P_corr = np.zeros((p, p), dtype=np.float)
	for i in range(p):
		P_corr[i, i] = 1
		for j in range(i+1, p):
			idx = np.ones(p, dtype=np.bool)
			idx[i] = False
			idx[j] = False
			beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
			beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

			res_j = C[:, j] - C[:, idx].dot( beta_i)
			res_i = C[:, i] - C[:, idx].dot(beta_j)
			
			corr = corr_func(res_i, res_j)[0]
			P_corr[i, j] = corr
			P_corr[j, i] = corr
		
	return P_corr

def weight_avg(df, by=None, weight=None, stats=['mean']):
	dfs = {}
	df = df.select_dtypes(include='number')
	cols = [col for col in df.columns if not col in [weight, by]]

	if type(by)==list:
		pp = [weight] + by
	else:
		pp = [weight, by]
		
	for stat in stats:
		if stat in ['mean', 'avg']:
			dg = df[cols].mul(df[weight], axis=0)
			dg[by] = df[by]
			dh = dg.groupby(by).sum()
			dh = dh.mul(1./df[pp].groupby(by).sum()[weight], axis=0)
			dfs[stat] = dh
		elif stat=='std':
			# Compute average first
			dg = df[cols].mul(df[weight], axis=0)
			dg[by] = df[by]
			dh = dg.groupby(by).sum()
			dh_avg = dh.mul(1./df[pp].groupby(by).sum()[weight], axis=0)
			
			dg = (df[cols]**2).mul(df[weight], axis=0)
			dg[by] = df[by]
			dh = dg.groupby(by).sum()
			dh = np.sqrt((dh.mul(1./df[pp].groupby(by).sum()[weight], axis=0) - dh_avg**2))
			dfs[stat] = dh
		else:
			#print ('Ignoring unknown', stat, 'statistics')
			pass
		
	return pd.concat(dfs).unstack(level=0)