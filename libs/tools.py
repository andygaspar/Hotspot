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