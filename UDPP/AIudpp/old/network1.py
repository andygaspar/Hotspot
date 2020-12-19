import copy
from typing import List

import torch
from torch import nn, optim

from UDPP.AirlineAndFlightAndSlot.udppAirline import UDPPairline
from UDPP.udppModel import UDPPmodel


class Net(torch.nn.Module):
    def __init__(self, inputDimension):
        super(Net, self).__init__()
        self.inputDimension = inputDimension
        self.fc1 = torch.nn.Linear(self.inputDimension, 64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 6)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, X):
        hidden = self.fc1(X)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output


class AirNetwork:

    def __init__(self, inputDimension, batchSize):

        self.inputDimension = inputDimension
        self.batchSize = batchSize
        self.lr = 1e-3
        self.lambdaL2 = 1e-4
        self.epochs = 200
        self.width = 64
        self.loss = 1e4
        self.bestLoss = 1e4
        self.bestWaits = None

        self.network = nn.Sequential(
            nn.Linear(self.inputDimension, self.width),
            nn.LeakyReLU(),
            nn.Linear(self.width, self.width*2),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.width*2, self.width*4),
            nn.LeakyReLU(),
            nn.Linear(self.width * 4, 12),
            # nn.Dropout(p=0.2)
            # nn.LeakyReLU(),
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

        torch.cuda.current_device()
        print(torch.cuda.is_available())
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr, weight_decay=self.lambdaL2)

    def train(self, numFlights, batchSize, inputs, outputs,
              airlines: List[UDPPairline], UDPPmodels: List[UDPPmodel]):
        criterion = torch.nn.MSELoss()
        self.network.train()
        for e in range(self.epochs):
            self.optimizer.zero_grad()
            X = torch.tensor(inputs, requires_grad=True).to(self.device) \
                .reshape(self.batchSize, self.inputDimension).type(dtype=torch.float32)

            Y = self.network(X)


            Y_test = torch.tensor(outputs).to(self.device).type(dtype=torch.float32)

            loss = criterion(Y, Y_test)
            loss.backward()
            self.optimizer.step()
            finalCosts = []

            if e == self.epochs-1:
                self.loss = loss.item()

        if self.loss < self.bestLoss:
            self.bestLoss = self.loss
            self.bestWaits = copy.deepcopy(self.network.state_dict())

    @staticmethod
    def my_loss(Y, rewards):
        sumTens = torch.sum(Y, 1)
        loss = torch.mean((sumTens - rewards)**2)
        return loss

    def prioritisation(self, input_list: List[float]):

        X = torch.tensor(input_list, requires_grad=False). \
            to(self.device).reshape(1, self.inputDimension).type(dtype=torch.float32)

        self.network.eval()
        with torch.no_grad():
            priorities = self.network(X).flatten().cpu().numpy()

        return priorities

    def save_weights(self, filename: str = "netWeights"):
        torch.save(self.bestWaits, filename + '.pt')

    def load_weights(self, file: str = "netWeights.pt"):
        self.network.load_state_dict(torch.load(file))
