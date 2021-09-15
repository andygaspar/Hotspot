import copy
from typing import List

from ...ModelStructure.Airline.airline import Airline
from ...ModelStructure.Slot.slot import Slot


def get_slot_from_time(slot_list: List[Slot], time: float):
    i = 0
    while i < len(slot_list) and time >= slot_list[i].time:
        i += 1
    return slot_list[i-1]


def udpp_local(airline: Airline, slots: List[Slot]):
    local_slots = list(copy.copy(airline.AUslots))
    for flight in airline.flights:
        if flight.udppPriority == "P":
            flight.newSlot = get_slot_from_time(slots, flight.tna)
            local_slots.remove(get_slot_from_time(local_slots, flight.tna))

    for flight in airline.flights:
        print("\n\n")
        if flight.udppPriority == "N":
            print("udpp prioity number", flight.udppPriorityNumber)
            print("local slots", local_slots)
            print("flight", flight.name, "airline", flight.airline)
            flight.newSlot = local_slots[flight.udppPriorityNumber]
        print("\n\n")