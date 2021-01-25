import numpy as np
import pandas as pd
from typing import Union, List, Callable
from itertools import product
from ModelStructure.Slot import slotList as sl
from ModelStructure.Airline import airline as air
from ModelStructure.Flight import flightList as fll
from ModelStructure.Airline import airlineList as airList


class ModelStructure:

    def __init__(self, df_init: pd.DataFrame, costFun: Union[Callable, List[Callable]], airline_ctor=air.Airline):

        self.df = df_init

        self.slots = sl.make_slots_list(self.df)

        self.airlines = airList.make_airlines_list(self.df, self.slots, airline_ctor)

        self.numAirlines = len(np.unique(self.df["airline"]))

        self.flights = fll.make_flight_list(self)

        self.set_flights_cost_functions(costFun)

        self.set_flights_cost_vect()

        self.numFlights = len(self.flights)

        self.initialTotalCosts = self.compute_costs(self.flights, "initial")

        self.airDict = dict(zip([airline.name for airline in self.airlines], range(len(self.airlines))))

        self.scheduleMatrix = self.make_schedule_matrix()

        self.emptySlots = self.df[self.df["flight"] == "Empty"]["slot"].to_numpy()

        # self.mipSolution = None

        # self.solutionArray = None

        self.solution = None

        self.report = None


    @staticmethod
    def compute_costs(flights, which):
        if which == "initial":
            return sum([flight.costFun(flight, flight.slot) for flight in flights])
        if which == "final":
            return sum([flight.costFun(flight, flight.newSlot) for flight in flights])

    @staticmethod
    def compute_delays(flights, which):
        if which == "initial":
            return sum([flight.slot.time-flight.eta for flight in flights])
        if which == "final":
            return sum([flight.newSlot.time-flight.eta for flight in flights])

    def make_schedule_matrix(self):
        arr = []
        for flight in self.flights:
            arr.append([flight.slot.time] + [flight.eta] + flight.costVect)
        return np.array(arr)

    def __str__(self):
        return str(self.airlines)

    def __repr__(self):
        return str(self.airlines)

    def print_schedule(self):
        print(self.df)

    def print_performance(self):
        # print(self.solution)
        print(self.report)

    def get_flight_by_slot(self, slot: sl.Slot):
        for flight in self.flights:
            if flight.slot == slot:
                return flight

    def get_flight_from_name(self, f_name):
        for flight in self.flights:
            if flight.name == f_name:
                return flight

    def set_flights_cost_functions(self, costFun):
        if isinstance(costFun, Callable):
            for flight in self.flights:
                flight.set_cost_fun(costFun)
        else:
            i = 0
            for flight in self.flights:
                flight.set_cost_fun(costFun[i])
                i += 1

    def set_flights_cost_vect(self):
        for flight in self.flights:
            flight.costVect = [flight.costFun(flight, slot) for slot in self.slots]


