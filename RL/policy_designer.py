from scipy.optimize import dual_annealing

class PolicyDesigner:
	"""
	Knows real cost of airlines
	"""
	def __init__(self, trainer=None):
		self.trainer = trainer

	def train_with_prices(self, jump_price=None, cost_price=None, margin_price=None,
		num_iterations=10000, n_eval_setp=500):
		"""
		Run RL with price of jump p (per unit of jump)
		"""

		if not jump_price is None:
			self.trainer.game.set_jump_price(jump_price)

		if not cost_price is None:
			self.trainer.game.set_cost_price(cost_price)

		if not margin_price is None:
			self.trainer.game.set_margin_price(margin_price)

		self.trainer.train_agent(num_iterations=num_iterations,
							n_eval_setp=n_eval_setp)

	def find_best_prices(self, n_iter_evaluation=100):
		### Only jump price for now
		def f(x):
			self.train_with_prices(jump_price=x)

			df = self.trainer.compare_airlines(n_iter=n_iter_evaluation)

			a = abs(df.mean()['compute_cost_diff_ratio_others'])

			return a

		dual_annealing(f, [(0., 100.)])


		

		





