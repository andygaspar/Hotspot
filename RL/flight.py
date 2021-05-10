class Flight:
	def __init__(self, eta, name=None, margin=None, cost=None, cost_function=None,
		jump=None):
		
		self.eta = eta
		self.name = name
		#self.slot_indices_dict = slot_indices_dict
		self.set_true_charac(margin=margin,
							cost=cost,
							jump=jump)
		
		self.set_declared_charac(margin=margin,
								cost=cost,
								jump=jump)
		
		self.set_cost_function(cost_function)
		
	def set_true_charac(self, margin=None, cost=None, jump=None):
		self.margin = max(0., margin)
		self.cost = max(0., cost)
		self.jump = max(0., jump)
		
	def set_declared_charac(self, margin=None, cost=None, jump=None):
		self.margin_declared = max(0., margin)
		self.cost_declared = max(0., cost)
		self.jump_declared = max(0., jump)
		
	def set_cost_function(self, cost_function):
		class DummyFlight:
			pass
		
		class DummySlot:
			pass
		
		# Real cost function
		def f(time):
			slot = DummySlot()
			slot.time = time
			#self.index = self.slot_indices_dict[time]
			return cost_function(self, slot)
		
		self.cost_f_true = f 
		
		# Declared cost function
		declared_flight = DummyFlight()
		declared_flight.eta = self.eta
		declared_flight.margin = self.margin_declared
		declared_flight.cost = self.cost_declared
		declared_flight.jump = self.jump_declared
		
		def ff(time):
			slot = DummySlot()
			slot.time = time
			#self.index = self.slot_indices_dict[time]
			return cost_function(declared_flight, slot)
		
		self.cost_f_declared = ff