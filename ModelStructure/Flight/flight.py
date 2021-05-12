import numpy as np
from typing import List, Callable

from ModelStructure.Slot.slot import Slot


class Flight:

    def __init__(self, slot: Slot, flight_name: str, airline_name: str,
                 eta: int, delay_cost_vect=np.array, udpp_priority=None, tna=None,
                 slope=None, margin_1=None, jump_1=None, margin_2=None, jump_2=None):

        self.temp = None

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

        self.udppPriority = None

        self.tna = None


        # ISTOP attributes  *************

        self.slope = None

        self.margin1 = None

        self.jump1 = None

        self.margin2 = None

        self.jump2 = None

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

    def set_num(self, i):
        self.num = i

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
        notCompatibleSlots = []
        for slot in slots:
            if slot not in self.compatibleSlots:
                notCompatibleSlots.append(slot)
        self.notCompatibleSlots = notCompatibleSlots

    def set_eta_slot(self, slots):
        i = 0
        while slots[i].time < self.eta:
            i += 1
        self.etaSlot = slots[i]

    def get_attributes(self):
        return self.slot, self.name, self.airlineName, self.eta, self.delayCostVect, \
               self.udppPriority, self.tna, self.slope, self.margin1, self.jump1, self.margin2, self.jump2

    def set_cost_fun(self, delay_cost_fun):
        self.delay_cost_fun = delay_cost_fun

    def costFun(self, slot):
        # delay = slot.time - self.eta
        return self.costVect[slot.index]
