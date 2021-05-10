import numpy as np
from typing import List, Callable

from Hotspot.ModelStructure.Slot import slot as sl


class Flight:

    def __init__(self, line, airline, slots: List[sl.Slot]):

        self.type = line["type"]

        self.slot = slots[line["slot"]]

        self.num = line["num"]

        self.newSlot = None

        self.name = line["flight"]

        self.airline = airline

        self.eta = line["eta"]

        self.etaSlot = self.get_eta_slot(self.eta, slots) #slots[len([slot for slot in slots if slot.time <= self.eta])]

        self.fpfs = line['fpfs']

        self.cost = line["cost"]

        self.costVect = None

        try:
            self.margin = line["margins"]
        except:
            self.margin = None

        self.costFun = None

        self.compatibleSlots = self.compute_compatible_slots(slots)

        self.notCompatibleSlots = self.compute_not_compatible_slots(slots)

        self.localNum = None

        # ISTOP attributes  *************

        self.priority = line["priority"]

        self.jump = line["jump"]

        self.preference = None

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def set_num(self, i):
        self.num = i

    def set_local_num(self, i):
        self.localNum = i

    def set_cost_fun(self, costFun: Callable):
        self.costFun = costFun

    def delay(self, slot: sl.Slot):
        return slot.time - self.eta

    def compute_compatible_slots(self, slots: List[sl.Slot]):
        try:
            compatible_slots = []
            for slot in slots:
                if slot.time >= self.eta:
                    compatible_slots.append(slot)
            # if compatible_slots[0].index > 0:
            #     compatible_slots.insert(0, slots[compatible_slots[0].index - 1])
            return compatible_slots
        except IndexError:
            raise IndexError("No available slot for flight ", self.name)

    def compute_not_compatible_slots(self, slots):
        notCompatibleSlots = []
        for slot in slots:
            if slot not in self.compatibleSlots:
                notCompatibleSlots.append(slot)
        return notCompatibleSlots

    @staticmethod
    def get_eta_slot(eta, slots):
        i = 0
        while slots[i].time < eta:
            i += 1
        return slots[i]

    def assign(self, solutionSlot: sl.Slot):
        self.newSlot = solutionSlot
        solutionSlot.free = False
