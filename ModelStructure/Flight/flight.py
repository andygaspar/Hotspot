import numpy as np
from typing import List, Callable

from Istop.Preferences import preference
from ModelStructure.Slot.slot import Slot


class Flight:

    def __init__(self, slot: Slot, flight_name: str, airline_name: str,
                 eta: float, delay_cost_vect=np.array,
                 udpp_priority: str = None, udpp_priority_number: int = None, tna: float = None,
                 slope: float = None, margin_1: float = None, jump_1: float = None,
                 margin_2: float = None, jump_2: float = None):

        self.slot = slot

        self.name = flight_name

        self.airlineName = airline_name

        self.eta = eta

        # attribute  handled by ModelStructure

        self.airline = None

        self.etaSlot = None

        self.costVect = None

        self.delayCostVect = delay_cost_vect

        self.delayVect = None

        self.compatibleSlots = None

        self.notCompatibleSlots = None

        self.localNum = None

        self.newSlot = None

        # UDPP attributes

        self.udppPriority = udpp_priority

        self.udppPriorityNumber = udpp_priority_number

        self.tna = tna

        # ISTOP attributes  *************

        self.slope = slope

        self.margin1 = margin_1

        self.jump1 = jump_1

        self.margin2 = margin_2

        self.jump2 = jump_2

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if type(other) == str:
            return self.name == other
        return self.name == other.name

    def set_local_num(self, i):
        self.localNum = i

    def delay(self, slot: Slot):
        return slot.time - self.eta

    def set_compatible_slots(self, slots: List[Slot]):
        compatible_slots = []
        for slot in slots:
            if slot.time >= self.eta:
                compatible_slots.append(slot)
        self.compatibleSlots = compatible_slots

    def set_not_compatible_slots(self, slots):
        not_compatible_slots = []
        for slot in slots:
            if slot not in self.compatibleSlots:
                not_compatible_slots.append(slot)
        self.notCompatibleSlots = not_compatible_slots

    def set_eta_slot(self, slots):
        i = 0
        while slots[i].time < self.eta:
            i += 1
        self.etaSlot = slots[i]

    def get_attributes(self):
        return self.slot, self.name, self.airlineName, self.eta, self.delayCostVect, \
               self.udppPriority, self.udppPriorityNumber, self.tna, \
               self.slope, self.margin1, self.jump1, self.margin2, self.jump2

    def cost_fun(self, slot):
        return self.costVect[slot.index]
