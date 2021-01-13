from typing import List

import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn, optim
from IPython import display

from Programma.UDPP import udppModel
from Programma.UDPP.AIudpp.trainAuxFuns1 import run_UDPP_local
from Programma.UDPP.AirlineAndFlightAndSlot.udppAirline import UDPPairline
from Programma.UDPP.AirlineAndFlightAndSlot.udppSlot import UDPPslot
from Programma.UDPP.udppModel import UDPPmodel
from data.dfMaker import df_maker
from Programma.ModelStructure.modelStructure import ModelStructure
from Programma.ModelStructure.Costs.costFunctionDict import CostFuns
from Programma.UDPP.Local.manageMflights import manage_Mflights


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
        self.epochs = 50
        self.width = 64
        self.loss = 0

        self.network = nn.Sequential(
            nn.Linear(self.inputDimension, self.width),
            nn.LeakyReLU(),
            nn.Linear(self.width, self.width*2),
            nn.LeakyReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(self.width*2, self.width*3),
            nn.LeakyReLU(),
            nn.Linear(self.width * 3, 6),
            # nn.Dropout(p=0.2)
            # nn.LeakyReLU(),
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

        # torch.cuda.current_device()
        print(torch.cuda.is_available())
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr, weight_decay=self.lambdaL2)

    def prioritisation(self, input_list: List[float]):

        X = torch.tensor(input_list, requires_grad=False). \
            to(self.device).reshape(1, self.inputDimension).type(dtype=torch.float32)

        self.network.eval()
        with torch.no_grad():
            priorities = self.network(X).flatten().cpu().numpy()

        return priorities

    def train(self, numFlights, batchSize, inputs, initialCosts: np.array,
              airlines: List[UDPPairline], UDPPmodels: List[UDPPmodel]):
        criterion = torch.nn.MSELoss()
        finalCosts = []
        self.network.train()
        for e in range(self.epochs):
            self.optimizer.zero_grad()
            X = torch.tensor(inputs, requires_grad=True).to(self.device) \
                .reshape(self.batchSize, self.inputDimension).type(dtype=torch.float32)

            Y = self.network(X)

            priorities = Y.flatten().cpu().detach().numpy().reshape(batchSize, numFlights)

            for i in range(batchSize):
                run_UDPP_local(priorities[i], airlines[i], UDPPmodels[i].slots)
                finalCosts.append(UDPPmodels[i].compute_costs(airlines[i].flights, "final"))

            rewards = torch.tensor((initialCosts - finalCosts) / initialCosts)

            loss = self.my_loss(Y, rewards)
            loss.backward()
            self.optimizer.step()
            finalCosts = []

            if e == self.epochs-1:
                self.loss = loss.item()

    @staticmethod
    def my_loss(Y, rewards):
        sumTens = torch.sum(Y, 1)
        loss = torch.mean((sumTens - rewards)**2)
        return loss
