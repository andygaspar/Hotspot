import numpy as np
import pandas as pd
from Programma.Istop.Solution import offer as ol


# kind of duplication of function in flightList
def get_flight(i, flights):
    for f in flights:
        if i == f.slot:
            return f


class Solution:

    def __init__(self, model):
        self.update_flights_status(model)
        self.df = self.make_solution_df(model)
        self.airline_balance = self.make_airline_balance(model)
        self.offers = ol.OffersList(model)

    @staticmethod
    def make_solution_df(model):
        cols = ["slot", "flight", "airline", "gdp_schedule", "old_schedule", "initial eta", "cost", "old cost", "priority"]
        df = pd.DataFrame(columns=cols)
        for j in model.slotIndexes:
            for i in model.slotIndexes:
                if model.solutionArray[i, j] != 0:
                    flight = get_flight(i, model.flights)
                    row = dict(zip(cols,
                                   [j, flight.name, flight.airline.name, model.slotTimeGrid[j], flight.fpfs,
                                    flight.eta, flight.cost * model.delays[i, j],
                                    flight.cost * model.delays[i, i], flight.priority]))
                    df = df.append(row, ignore_index=True)
        return df

    @staticmethod
    def make_airline_balance(model):
        old_balance = np.zeros(model.airlines.shape[0])
        new_balance = np.zeros(model.airlines.shape[0])
        for airline in model.airlines:
            for flight in airline.flights:
                old_balance[airline.index] += flight.cost * model.delays[flight.slot, flight.slot]
                new_balance[airline.index] += flight.cost * model.delays[flight.slot, flight.newSlot]
        return pd.DataFrame({"airline": model.airlines, "new balance": new_balance, "old balance": old_balance})

    @staticmethod
    def update_flights_status(model):
        for j in model.slotIndexes:
            for i in model.slotIndexes:
                if model.solutionArray[i, j] != 0:
                    flight = get_flight(i, model.flights)
                    flight.newSlot = j
                    flight.new_arrival = model.slotTimeGrid[j]

    def __repr__(self):
        return self.df

    def __str__(self):
        return self.df.to_string()



