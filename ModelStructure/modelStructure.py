import numpy as np
import pandas as pd
from typing import Union, List, Callable
from itertools import product
from ModelStructure.Airline.airline import Airline
from ModelStructure.Slot import slot as sl
from ModelStructure.Flight.flight import Flight


class ModelStructure:

    def __init__(self, flights: List[Flight], air_ctor=Airline):

        self.flights = flights

        self.slots = [flight.slot for flight in self.flights]

        self.airlines, self.airDict = self.make_airlines(air_ctor)

        self.numAirlines = len(self.airlines)

        self.set_flights_attributes()

        self.set_flights_cost_vect()

        self.numFlights = len(self.flights)

        self.initialTotalCosts = self.compute_costs(self.flights, "initial")

        self.scheduleMatrix = self.make_schedule_matrix()

        self.emptySlots = []

        self.solution = None

        self.report = None

        self.df = self.make_df()

    @staticmethod
    def compute_costs(flights, which):
        if which == "initial":
            return sum([flight.costFun(flight, flight.slot) for flight in flights])
        if which == "final":
            return sum([flight.costFun(flight, flight.newSlot) for flight in flights])

    @staticmethod
    def compute_delays(flights, which):
        if which == "initial":
            return sum([flight.slot.time - flight.eta for flight in flights])
        if which == "final":
            return sum([flight.newSlot.time - flight.eta for flight in flights])

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
        print(self.report)

    def get_flight_by_slot(self, slot: sl.Slot):
        for flight in self.flights:
            if flight.slot == slot:
                return flight

    def get_flight_from_name(self, f_name):
        for flight in self.flights:
            if flight.name == f_name:
                return flight

    def set_flights_cost_vect(self):
        for flight in self.flights:
            flight.costVect = []
            for slot in self.slots:
                flight.costVect.append(flight.costFun(slot))


    def make_slots(self):
        pass

    def make_airlines(self, air_ctor):

        air_flight_dict = {}
        for flight in self.flights:
            if flight.airlineName not in air_flight_dict.keys():
                air_flight_dict[flight.airlineName] = [flight]
            else:
                air_flight_dict[flight.airlineName].append(flight)

        air_names = list(air_flight_dict.keys())
        airlines = [air_ctor(air_names[i], i, air_flight_dict[air_names[i]]) for i in range(len(air_flight_dict))]
        air_dict = dict(zip(airlines, range(len(airlines))))

        return airlines, air_dict

    def make_df(self):
        slot_index = [flight.slot.index for flight in self.flights]
        flights = [flight.name for flight in self.flights]
        airlines = [flight.airlineName for flight in self.flights]
        slot_time = [flight.slot.time for flight in self.flights]

        return pd.DataFrame({"slot": slot_index, "flight": flights, "airline": airlines, "time": slot_time})

    def set_flights_attributes(self):
        for flight in self.flights:
            flight.set_eta_slot(self.slots)
            flight.set_compatible_slots(self.slots)
            flight.set_not_compatible_slots(self.slots)

    def get_new_flight_list(self):
        new_flight_list = []
        for flight in self.flights:
            new_flight = Flight(*flight.get_attributes())
            new_flight.slot = flight.newSlot
            new_flight.newSlot = None
            new_flight_list.append(new_flight)

        return sorted(new_flight_list, key=lambda f: f.slot)
