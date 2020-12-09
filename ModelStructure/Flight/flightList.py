import numpy as np


def make_flight_list(model):
    flight_list = []
    flight_slots = model.df[model.df["flight"] != "Empty"]["slot"]

    for i in flight_slots:
        flight_list.append(get_flight(i, model.airlines))

    return np.array(flight_list)


def get_flight(i, airlines):
    for a in airlines:
        for f in a.flights:
            if i == f.slot.index:
                return f


def assign_flight_num(flight_list):
    from Programma.ModelStructure.Flight.flight import Flight
    flight: Flight
    i = 0
    for flight in flight_list:
        flight.set_num(i)
        i += 1
