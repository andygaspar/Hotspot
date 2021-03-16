import gym
import numpy as np

from copy import copy

from ModelStructure.ScheduleMaker import scheduleMaker
from ModelStructure.Costs.costFunctionDict import CostFuns

from RL.wrapper_UDPP import allocation_from_df
from RL.wrapper_UDPP import df_sch_from_flights
from RL.wrapper_UDPP import OptimalAllocationComputer #, compute_optimal_allocation
from RL.flight import Flight
from libs.tools import clock_time

def linear_function(min_y, max_y):
    def f(x):
        return (max_y-min_y) * x + min_y

    return f

class ContGame(gym.Env):
    """
    Game with iteration where a player (airline)
    modifies its declared flights characteristics (margins and costs)
    in order to minimise its real cost, but on different allocation/hotstop every time.

    This is using UDPP (with real cost) and ISTOP or nnBound on top (with declared/fake
    cost functions).

    The state is the set of indices of its flight in the queue. The reward is the
    opposite of the sum of the cost across flights from airline player (plus an offset).
    """
    def __init__(self, n_f=10, n_a=3, player='A', seed=None,
        offset=100., cost_type='jump', min_jump=10, max_jump=100,
        trading_alg='nnbound', n_f_player=4, new_capacity=5.,
        min_margin=10, max_margin=45, price_jump=0., price_cost=0.,
        price_margin=0.):

        super().__init__()

        self.set_price_jump(price_jump)
        self.set_price_cost(price_cost)
        self.set_price_margin(price_margin)

        self.min_margin = int(min_margin)
        self.max_margin = int(max_margin)

        self.func_margin = linear_function(min_margin, max_margin)

        self.min_jump = int(min_jump)
        self.max_jump = int(max_jump)

        self.func_jump = linear_function(min_jump, max_jump)

        np.random.seed(seed)

        self.trading_alg = trading_alg
        self.allocation_computer = OptimalAllocationComputer(trading_alg=trading_alg)
        self.allocation_computer_nnbound = OptimalAllocationComputer(trading_alg='nnbound')

        self.viewer = None
        self.player = player

        self.costFun = CostFuns().costFun[cost_type]

        self.n_f = n_f
        self.n_a = n_a

        self.history = {}

        self.n_f_player = n_f_player

        #self.observation_space =  gym.spaces.Box(0, 100, shape=(self.n_f_player, 3))#, dtype=float)
        self.observation_space =  gym.spaces.Box(0, 100, shape=(self.n_f_player*3, ))#, dtype=int)

        #self.action_space = gym.spaces.Box(0, 1, shape=(self.n_f_player, 2))#, dtype=float)
        self.action_space = gym.spaces.Box(0, 1, shape=(self.n_f_player*2, ))#, dtype=float)

        self.offset = offset

        self.new_capacity = new_capacity

    def game_summary(self):
        print ('Number of flights:', self.n_f)
        print ('Number of airlines:', self.n_a)
        print ('Player name:', self.player)
        print ('Number of flights for player:', self.n_f_player)
        print ('New capacity of hotspot:', self.new_capacity)
        print ('Min/Max margins:', self.min_margin, self.max_margin)
        print ('Min/Max jumps:', self.min_jump, self.max_jump)
        print ('Trading algorithm:', self.trading_alg)

    def build_flights(self):
        self.flights = {}
        for i, row in self.df_sch.iterrows():
            self.flights[row['flight']] = Flight(row['eta'],
                                                name=row['flight'],
                                                margin=row['margins'],
                                                cost=row['cost'],
                                                jump=row['jump'],
                                                cost_function=self.costFun,
                                                )

        self.flight_per_company = {company:list(self.df_sch[self.df_sch['airline']==company]['flight'])
                                       for company in self.df_sch['airline'].unique()}

    def cost_of_allocation(self, allocation, only_player=False):
        # True cost
        cost_tot = 0.
        cost_c = {}

        for company, flights in self.flight_per_company.items():
            for flight in flights:
                    # print ('company:', company)
                    # print ('flight:', flight)
                slot = allocation[flight]
                #print ('slot', slot)
                time = self.slot_times[slot]

                #print ('A', self.flights[flight].jump)
                #print ('B', self.flights[flight].jump_declared)

                # Penalty for changing their declared cost function.
                dj = self.flights[flight].jump - self.flights[flight].jump_declared
                dc = self.flights[flight].cost - self.flights[flight].cost_declared
                dm = -(self.flights[flight].margin - self.flights[flight].margin_declared)

                add_cost = dj*self.jump_price + dc*self.cost_price + dm*self.margin_price

                cost = self.flights[flight].cost_f_true(time)# + add_cost

                cost_tot += cost

                # print ('cost:', cost)

                cost_c[company] = cost_c.get(company, 0.) + cost

        #print ()

        if only_player:
            return cost_c[self.player]
        else:
            return cost_tot , cost_c

    def cost_of_allocation_declared(self, allocation, only_player=False):
        # True cost
        cost_tot = 0.
        cost_c = {}

        for company, flights in self.flight_per_company.items():
            for flight in flights:
                slot = allocation[flight]

                time = self.slot_times[slot]

                cost = self.flights[flight].cost_f_declared(time)

                cost_tot += cost

                cost_c[company] = cost_c.get(company, 0.) + cost

        if only_player:
            return cost_c[self.player]
        else:
            return cost_tot , cost_c

    def make_new_schedules(self):
        # Prepare new hotspot
        scheduleType = scheduleMaker.schedule_types(show=False)

        self.df_sch_init = scheduleMaker.df_maker(self.n_f,
                                                self.n_a,
                                                distribution=scheduleType[0],
                                                new_capacity=self.new_capacity,
                                                n_flight_first_airline=self.n_f_player,
                                                min_margin=self.min_margin,
                                                max_margin=self.max_margin,
                                                min_jump=self.min_jump,
                                                max_jump=self.max_jump)

        self.df_sch = self.df_sch_init.copy()

        self.flight_per_company = self.df_sch[['flight', 'airline']].groupby('airline').agg(list).to_dict()['flight']
        df = self.df_sch_init[['slot', 'time']]
        self.slot_times = df.set_index('slot').to_dict()['time']

        self.build_flights()
        self.player_flights = self.flight_per_company[self.player]

        self.base_allocation = allocation_from_df(self.df_sch_init, name_slot='slot')
        self.base_cost_tot, self.base_cost_per_c = self.cost_of_allocation(self.base_allocation)

        self.best_allocation = self.allocation_computer.compute_optimal_allocation(self.df_sch, self.costFun)
        self.best_cost_tot, self.best_cost_per_c = self.cost_of_allocation(self.best_allocation)

    def compute_reward(self):
        # Build the df used as input by the optimisers
        self.df_sch = df_sch_from_flights(self.df_sch_init, self.flights)
        #print (self.df_sch)

        # Optimise the allocation using ISTOP or NNbound and compute back the allocation
        allocation = self.allocation_computer.compute_optimal_allocation(self.df_sch, self.costFun)

        # Compute cost for all companies for player : real cost
        # real cost
        cost_tot, cost_per_c = self.cost_of_allocation(allocation)
        # declared cost
        cost_tot_declared, cost_per_c_declared = self.cost_of_allocation_declared(allocation)

        # Reward is relative to cost in intial allocation
        reward = self.offset - (cost_per_c[self.player] - self.best_cost_per_c[self.player])
        reward_fake = self.offset - (cost_per_c_declared[self.player] - self.best_cost_per_c[self.player])

        # print('cost_tot, cost_per_c', cost_tot, cost_per_c)
        # print (self.df_sch)

        reward_tot = len(self.flights) * self.offset - (cost_tot-self.best_cost_tot)

        #state = [slot for name, slot in allocation.items() if name in self.flight_per_company[self.player]]

        return reward, reward_tot, cost_tot, cost_per_c, allocation, reward_fake

    def get_player_flight_charac(self, charac_name):
        return list(self.df_sch['margins'][self.df_sch['airline']=='A'])

    def get_current_margins(self):
        return self.get_player_flight_charac('margins')

    def get_current_jumps(self):
        return self.get_player_flight_charac('jumps')

    def get_state(self):
        #state = [slot for name, slot in allocation.items() if name in self.flight_per_company[self.player]]
        state = self.df_sch.set_index('flight').loc[self.player_flights, ['margins', 'jump', 'time']]#, 'slot']]
        state = tuple(np.array(state).flatten())
        return state

    def set_price_jump(self, price):
        self.jump_price = price

    def set_price_cost(self, price):
        self.cost_price = price

    def set_price_margin(self, price):
        self.margin_price = price

    def step(self, action):
        # Apply action (modification of margin and cost) on flights
        self.apply_action(action)

        # Compute reward
        reward, reward_tot, cost_tot, cost_per_c, allocation, reward_fake = self.compute_reward()

        # Remember stuff
        names = self.flight_per_company[self.player]

        done = False

        information = {'allocation':allocation,
                                'df_sch':self.df_sch.copy(),
                                'df_sch_init':self.df_sch_init.copy(),
                                'best_allocation':copy(self.best_allocation),
                                'base_allocation':copy(self.base_allocation)}

        # Compute new schedule for next state
        self.make_new_schedules()

        state = self.get_state()

        return state, reward, done, information

    def get_charac(self):
        return np.array([self.get_player_flight_charac('margins'), self.get_player_flight_charac('jumps')])

    def set_charac(self, charac):
        for i, ac in enumerate(charac):
            name = self.flight_per_company[self.player][i]

            new_cost = self.flights[name].cost_declared# + dc

            self.flights[name].set_declared_charac(margin=ac[0], cost=new_cost, jump=ac[1])

    def reset(self):
        # Compute new schedule for next state
        self.make_new_schedules()

        state = self.get_state()

        return state

    def render(self, mode='human'):
        pass 

    def apply_action(self, action):
        # Unflatten action
        action = action.reshape((self.n_f_player, 2))
        for i, ac in enumerate(action):
            # for each flight
            name = self.flight_per_company[self.player][i]

            new_cost = self.flights[name].cost_declared

            scaled_action_margin = self.func_margin(ac[0])
            scaled_action_jump = self.func_jump(ac[1])

            self.flights[name].set_declared_charac(margin=scaled_action_margin, cost=new_cost, jump=scaled_action_jump)


class ContGameMargin(ContGame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.observation_space =  gym.spaces.Box(self.min_margin, self.max_margin, shape=(self.n_f_player*3, ))#, dtype=int)

        self.action_space = gym.spaces.Box(0, 1, shape=(self.n_f_player, ))#, dtype=float)

    def apply_action(self, action):
        for i, ac in enumerate(action):
            # for each flight
            name = self.flight_per_company[self.player][i]

            new_cost = self.flights[name].cost_declared
            new_jump = self.flights[name].jump_declared

            scaled_action_margin = self.func_margin(ac)
            self.flights[name].set_declared_charac(margin=scaled_action_margin, cost=new_cost, jump=new_jump)


class ContGameJump(ContGame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.observation_space =  gym.spaces.Box(self.min_jump, self.max_jump, shape=(self.n_f_player*3, ))#, dtype=int)

        self.action_space = gym.spaces.Box(0, 1, shape=(self.n_f_player, ))#, dtype=float)

    def apply_action(self, action):
        for i, ac in enumerate(action):
            # for each flight
            name = self.flight_per_company[self.player][i]

            new_cost = self.flights[name].cost_declared
            new_margin = self.flights[name].margin_declared

            scaled_action_jump = self.func_jump(ac)
            self.flights[name].set_declared_charac(margin=new_margin, cost=new_cost, jump=scaled_action_jump)