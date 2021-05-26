import numpy as np
from typing import List, Callable

from Hotspot.Istop.Preferences import preference
from Hotspot.ModelStructure.Slot.slot import Slot


class Flight:

    def __init__(self, slot: Slot, flight_name: str, airline_name: str,
                 eta: float, delay_cost_vect=np.array, cost_vect: np.array = None,
                 udpp_priority: str = None, udpp_priority_number: int = None, tna: float = None,
                 slope: float = None, margin_1: float = None, jump_1: float = None,
                 margin_2: float = None, jump_2: float = None):

        self.index = None

        self.slot = slot

        self.name = flight_name

        self.airlineName = airline_name

        self.eta = eta

        # attribute  handled by ModelStructure

        self.airline = None

        self.etaSlot = None

        self.costVect = cost_vect

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
        d = {'slot':self.slot,
            'flight_name':self.name,
            'airline_name':self.airlineName,
            'eta':self.eta,
            'delay_cost_vect':self.delayCostVect,
            'cost_vect':self.costVect,
            'udpp_priority':self.udppPriority,
            'udpp_priority_number':self.udppPriorityNumber,
            'tna':self.tna,
            'slope':self.slope,
            'margin_1':self.margin1,
            'jump_1':self.jump1,
            'margin_2':self.margin2,
            'jump_2':self.jump2}

        return d

    def cost_fun(self, slot):
        return self.costVect[slot.index]

    def update_slot(self):
        """
        replace .slot by .newSlot if the latter exists
        """

        if not self.newSlot is None:
            self.slot = self.newSlot
            self.newSlot = None

    def reset_slot(self):
        self.newSlot = None
