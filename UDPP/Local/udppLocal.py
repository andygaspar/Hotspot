from typing import Union, Callable, List

from Hotspot.UDPP.AirlineAndFlightAndSlot.udppFlight import UDPPflight
from Hotspot.UDPP.AirlineAndFlightAndSlot.udppAirline import UDPPairline
from Hotspot.ModelStructure.Slot.slot import Slot
from Hotspot.UDPP.AirlineAndFlightAndSlot.udppSlot import UDPPslot
from Hotspot.ModelStructure.modelStructure import ModelStructure
from Hotspot.ModelStructure.Solution import solution
from Hotspot.UDPP.Local.manageMflights import manage_Mflights
from Hotspot.UDPP.Local.manageNflights import manage_Nflights
from Hotspot.UDPP.Local.mangePflights import manage_Pflights


def make_slot_list(flights: List[UDPPflight]):
    return [flight for flight in flights if flight.priority != "B"]


def udpp_local(airline: UDPPairline, slots: List[Slot]):

    slotList: List[UDPPslot]
    Pflights: List[UDPPflight]
    Mflights: List[UDPPflight]

    slotList = [UDPPslot(flight.slot, None, flight.localNum) for flight in airline.flights if flight.priority != "B"]
    Pflights = [flight for flight in airline.flights if flight.priorityValue == "P"]
    manage_Pflights(Pflights, slotList, slots)

    Mflights = [flight for flight in airline.flights if flight.priorityValue == "M"]
    manage_Mflights(Mflights, slotList)

    Nflights = [flight for flight in airline.flights if flight.priorityValue == "N"]
    manage_Nflights(Nflights, slotList)



    for flight in airline.flights:
        flight.UDPPlocalSolution = flight.newSlot

