import numpy as np
from typing import List, Callable

#from Hotspot.Istop.Preferences import preference
from Hotspot.ModelStructure.Slot.slot import Slot


class Flight:

    def __init__(self, flight_name: str, airline_name: str,
                 eta: float, slot: Slot=None, delay_cost_vect: np.array=None, cost_vect: np.array = None,
                 **garbage
                 #udpp_priority: str = None, udpp_priority_number: int = None, tna: float = None,
                 # slope: float = None, margin_1: float = None, jump_1: float = None,
                 # margin_2: float = None, jump_2: float = None
                 ):

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

        # self.udppPriority = udpp_priority

        # self.udppPriorityNumber = udpp_priority_number

        # self.tna = tna

        # ISTOP attributes  *************

        # self.slope = slope

        # self.margin1 = margin_1

        # self.jump1 = jump_1

        # self.margin2 = margin_2

        # self.jump2 = jump_2

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
        d = {'slot':getattr(self, 'slot', None),
            'flight_name':getattr(self, 'name', None),
            'airline_name':getattr(self, 'airlineName', None),
            'eta':getattr(self, 'eta', None),
            'delay_cost_vect':getattr(self, 'delayCostVect', None),
            'cost_vect':getattr(self, 'costVect', None),
            'udpp_priority':getattr(self, 'udppPriority', None),
            'udpp_priority_number':getattr(self, 'udppPriorityNumber', None),
            'tna':getattr(self, 'tna', None),
            'slope':getattr(self, 'slope', None),
            'margin_1':getattr(self, 'margin1', None),
            'jump_1':getattr(self, 'jump1', None),
            'margin_2':getattr(self, 'margin2', None),
            'jump_2':getattr(self, 'jump2', None)}

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

    # def compute_delay_cost_vect(self, slots):
    #     """
    #     This is used when costVect is given instead of delayCostVect, but
    #     the latter is still required, for instance for ISTOP
    #     """
    #     #for flight in self.flights:
    #     if self.delayCostVect is None:
    #         self.delayCostVect = []
    #         i = 0
    #         for slot in slots:
    #             if slot.time >= self.eta:
    #                 self.delayCostVect.append(self.costVect[i])
    #             i += 1

    #         self.delayCostVect = np.array(self.delayCostVect)
