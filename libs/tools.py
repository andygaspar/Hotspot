import sys
from collections import OrderedDict
import contextlib
import datetime

class TwoWayDict(OrderedDict):
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

def print_allocation(allocation):
	s = ''
	for slot, name in allocation.items():
		
		s += str(name) + ' -> ' + str(slot) + ' ; '
		
	print (s)

@contextlib.contextmanager
def print_to_void():
    stdout_backup = sys.stdout
    sys.stdout = None
    yield
    sys.stdout = stdout_backup

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