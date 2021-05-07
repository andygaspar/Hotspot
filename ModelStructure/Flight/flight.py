import numpy as np
from typing import List, Callable

from ModelStructure.Slot.slot import Slot


class Flight:

    def __init__(self, flight_type: str, slot: Slot, num, flight_name: str, airline_name: str,
                 eta: int, cost_fun: Callable, udpp_priority: int = None, margins: int = None):

        self.type = flight_type

        self.slot = slot

        self.name = flight_name

        self.airlineName = airline_name

        self.eta = eta

        self.num = num

        self.delay_cost_fun = cost_fun

        self.udppPriority = udpp_priority

        self.flight_id = None

        # attribute  handled by ModelStructure

        self.airline = None

        self.etaSlot = None

        self.costVect = None

        self.delayVect = None

        self.compatibleSlots = None

        self.notCompatibleSlots = None

        self.localNum = None

        self.newSlot = None

        # ISTOP attributes  *************

        self.preference = None

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
        return self.type, self.slot, self.num, self.name, self.airlineName, self.eta, \
               self.delay_cost_fun, self.udppPriority

    def set_cost_fun(self, delay_cost_fun):
        self.delay_cost_fun = delay_cost_fun

    def costFun(self, slot):
        delay = slot.time - self.eta
        return self.delay_cost_fun(delay)

    def set_WM_cost_fun(self, cost_fun_obj):
        self.delay_cost_fun = cost_fun_obj.costFun["realistic"][self.flight_id]
