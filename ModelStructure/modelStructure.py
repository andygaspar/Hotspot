import numpy as np
import pandas as pd
from typing import Union, List, Callable
from itertools import product
from Hotspot.ModelStructure.Airline.airline import Airline
from Hotspot.ModelStructure.Slot import slot as sl
from Hotspot.ModelStructure.Flight.flight import Flight

import matplotlib.pyplot as plt

from Hotspot.ModelStructure.Slot.slot import Slot


class ModelStructure:

    requirements = []

    def __init__(self, slots: List[Slot] = None, flights: List[Flight] = None,
        air_ctor=Airline, checks=False):

        if not flights is None:
            self.slots = slots

            self.flights = [flight for flight in flights if flight is not None]

            self.set_flight_index()

            # self.set_cost_vect()

            self.airlines, self.airDict = self.make_airlines(air_ctor)

            self.numAirlines = len(self.airlines)

            self.set_flights_attributes()

            self.numFlights = len(self.flights)

            if 'costVect' in self.requirements or 'delayCostVect' in self.requirements:
                self.initialTotalCosts = self.compute_costs(self.flights, "initial")

                self.scheduleMatrix = self.set_schedule_matrix()

            self.emptySlots = []

            self.solution = None

            self.report = None

            self.df = self.make_df()

            # Check that all flights have the attributes required by the model
            if checks:
                self.check_requirements()

    def check_requirements(self):
        reqs_ok = True
        for flight in self.flights:
            req_ok = False
            for req in self.requirements:
                try:
                    # If req is an iterable, then all attributes in the 
                    # list must be owned by the flight object
                    iterator = iter(req)
                    req_attr_ok = True
                    for attr in req:
                        req_attr_ok = req_attr_ok and hasattr(flight, attr)
                    req_ok = req_ok or req_attr_ok
                except TypeError:
                    req_ok = req_ok or hasattr(flight, attr)

            reqs_ok = reqs_ok and req_ok

        try:
            assert reqs_ok
        except:
            raise Exception("Not all flights have the necessary requirements for this model.\n \
                            Check the model requirements by inspecting model.requirements.")

    @staticmethod
    def compute_costs(flights, which):
        if which == "initial":
            return sum([flight.cost_fun(flight.slot) for flight in flights])
        if which == "final":
            return sum([flight.cost_fun(flight.newSlot) for flight in flights])

    @staticmethod
    def compute_costs_list(flights, which):
        if which == "initial":
            return [flight.cost_fun(flight.slot) for flight in flights]
        if which == "final":
            return [flight.cost_fun(flight.newSlot) for flight in flights]

    @staticmethod
    def compute_delays(flights, which):
        if which == "initial":
            return sum([flight.slot.time - flight.eta for flight in flights])
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

    def print_new_schedule(self):
        print(self.solution)

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

    def get_new_flight_list(self):
        """
        Creates a new list of flight objects with newSlot as slot,
        except if the former is None, in which case the new objects
        have the same slot attribute than the old one.
        """
        new_flight_list = []
        for flight in self.flights:
            new_flight = Flight(**flight.get_attributes())
            if not flight.newSlot is None:

                new_flight.slot = flight.newSlot
                new_flight.newSlot = None
            new_flight_list.append(new_flight)

        return sorted(new_flight_list, key=lambda f: f.slot)

    def set_flight_index(self):
        for i in range(len(self.flights)):
            self.flights[i].index = i

    def set_flights_attributes(self):
        for flight in self.flights:
            flight.set_eta_slot(self.slots)
            flight.set_compatible_slots(self.slots)
            flight.set_not_compatible_slots(self.slots)

    def set_delay_vect(self):
        for flight in self.flights:
            flight.delayVect = np.array(
                [0 if slot.time < flight.eta else slot.time - flight.eta for slot in self.slots])

    def set_schedule_matrix(self):
        arr = []
        for flight in self.flights:
            arr.append([flight.slot.time] + [flight.eta] + list(flight.costVect))
        return np.array(arr)

    def make_airlines(self, air_ctor):
        air_flight_dict = {}
        for flight in self.flights:
            if flight.airlineName not in air_flight_dict.keys():
                air_flight_dict[flight.airlineName] = [flight]
            else:
                air_flight_dict[flight.airlineName].append(flight)

        air_names = list(air_flight_dict.keys())
        airlines = [air_ctor(air_names[i], air_flight_dict[air_names[i]]) for i in range(len(air_flight_dict))]
        air_dict = dict(zip(airlines, range(len(airlines))))

        return airlines, air_dict

    def make_df(self):
        slot_index = [flight.slot.index for flight in self.flights]
        flights = [flight.name for flight in self.flights]
        airlines = [flight.airlineName for flight in self.flights]
        slot_time = [flight.slot.time for flight in self.flights]
        eta = [flight.eta for flight in self.flights]
        airline = [flight.airlineName for flight in self.flights]

        return pd.DataFrame({"slot": slot_index, "flight": flights, "airline": airlines, "time": slot_time,
                             "eta": eta})

    def update_flights(self):
        [flight.update_slot() for flight in self.flights]


def make_slot_and_flight(slot_time: float, slot_index: int,
                         flight_name: str = None, airline_name: str = None, eta: float = None,
                         delay_cost_vect: np.array = None, udpp_priority=None, tna=None,
                         slope: float = None, margin_1: float = None, jump_1: float = None, margin_2: float = None,
                         jump_2: float = None,
                         empty_slot=False, cost_vect=None):

    slot = Slot(slot_index, slot_time)
    if not empty_slot:
        flight = Flight(slot=slot,
                        flight_name=flight_name,
                        airline_name=airline_name,
                        eta=eta,
                        delay_cost_vect=delay_cost_vect,
                        cost_vect=cost_vect,
                        udpp_priority=udpp_priority,
                        tna=tna,
                        slope=slope,
                        margin_1=margin_1,
                        jump_1=jump_1,
                        margin_2=margin_2,
                        jump_2=jump_2)
    else:
        flight = None
    return slot, flight


