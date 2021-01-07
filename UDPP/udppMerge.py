import numpy as np


def sort_flights_by_time(flights):
    time_list = [f.newSlot.time for f in flights]
    sorted_indexes = np.argsort(time_list)
    return np.array([flights[i] for i in sorted_indexes])


def getFirstCompatibleFlight(slot, sorted_flights, slots):
    for flight in sorted_flights:
        if flight.eta <= slot.time:
            return flight


def UDPPmerge(flights, slots):
    sorted_flights = list(sort_flights_by_time(flights))
    print([f.newSlot for f in sorted_flights])
    i = 0
    while len(sorted_flights) > 0:
        if sorted_flights[0].eta <= slots[i].time:
            sorted_flights[0].newSlot = slots[i]
            sorted_flights.pop(0)

        else:
            flight = getFirstCompatibleFlight(slots[i], sorted_flights, slots)
            flight.newSlot = slots[i]
            sorted_flights.remove(flight)
        i += 1




    # for f in sorted_flights:
    #     if f.eta <= slots[i].time:
    #         f.newSlot = slots[i]