# from Skier.AIPlayer import AIPlayer
from Enviroment.phisical_properties import PhysicalProperties
from Skier.skier import Skier
from Training.network import Network
import numpy as np
from Training.replayMemory import ReplayMemory


class AITrainer(Skier):

    def __init__(self, reward: float, reward_out: float,
                 sample_size: int, capacity: int,
                 gamma: float, eps_min: float, eps_decay: float,
                 double_q_interval: int = 20):

        super().__init__()
        self.rewardOut = reward_out
        self.reward = reward
        self.model_network = Network(4)
        self.target_network = Network(4)
        self.state = None
        self.final_state = -np.ones(4)
        self.action = None
        self.replayMemory = ReplayMemory(sample_size, capacity)
        self.gamma = gamma
        self.eps_greedy_value = 1.
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.double_q_interval = double_q_interval
        self.double_q_counter = 0

    def get_random_action(self) -> int:
        self.action = np.random.choice([0, 1, 2])
        return self.action

    def get_action(self, state: PhysicalProperties) -> int:
        self.state = self.get_state(state)
        if np.random.rand() > self.eps_greedy_value:
            self.action = self.model_network.get_action(self.state)
        else:
            self.action = self.get_random_action()
        return self.convert_action(self.action)

    def update_eps(self, iteration: int):
        self.eps_greedy_value = self.eps_min + (1 - self.eps_min) * np.exp(- self.eps_decay * iteration)

    def train_model_network(self):
        if self.replayMemory.size < self.replayMemory.sampleSize:
            return
        for i in range(2):
            self.model_network.update_weights(self.replayMemory.get_sample(), self.gamma, self.target_network)
        self.double_q_counter += 1

        if self.double_q_interval == 0:
            return
        if self.double_q_counter % self.double_q_interval == 0:
            self.update_target_network()

    def update_target_network(self):
        self.target_network.take_weights(self.model_network)

    def end(self):
        self.replayMemory.add_record(self.state, self.action, self.final_state,
                                     self.reward, done=True)
        self.train_model_network()

    def out(self):
        self.replayMemory.add_record(self.state, self.action, self.final_state,
                                     self.rewardOut, done=True)
        self.train_model_network()

    def gate_done(self, next_state: PhysicalProperties):
        next_state = self.get_state(next_state)
        self.replayMemory.add_record(self.state, self.action, next_state,
                                     self.reward, done=False)

    def add_record(self, next_state: PhysicalProperties, done: bool):
        next_state = self.get_state(next_state)
        self.replayMemory.add_record(self.state, self.action, next_state, reward=0, done=done)

    def get_state(self, state: PhysicalProperties):
        return np.array([state.position.x, state.position.y, state.v[0], state.v[1]])

    def convert_action(self, action):
        if action == 0:
            return -1
        if action == 1:
            return 0
        if action == 2:
            return 1
