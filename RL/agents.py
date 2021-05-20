import numpy as np

class Agent:
	def __init__(self, kind='generic', name='generic'):
		self.kind = kind
		self.name = name

	def load(self, file_name):
		pass

	def save(self, file_name):
		pass

	def action(self, observation):
		pass

	def __repr__(self):
		return 'Agent {} of type {}'.format(self.name, self.kind)
