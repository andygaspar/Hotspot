import numpy as np
from typing import List, Callable

from ...libs.uow_tool_belt.general_tools import get_first_matching_element

from ..Slot.slot import Slot

def compatible_slots(slots, eta, alternative_rule=False):
    """
    This assumes that slots are ordered by time.
    """
    if not alternative_rule:
        pouic = Slot(index=-1, time=None)
        first_slot_index = next((slot for slot in slots[::-1] if slot.time<eta), pouic).index +1
    else:
        #first_slot_index = max(get_first_matching_element(slots, condition=lambda x:x.time>eta).index-1, 0)    
        pouic = Slot(index=len(slots), time=None)
        first_slot_index = max(get_first_matching_element(slots, condition=lambda x:x.time>eta, default=pouic).index-1, 0)    
    return [slot for i, slot in enumerate(slots) if i>=first_slot_index]

class Flight:

    def __init__(self, flight_name: str, airline_name: str,
                 eta: float, slot: Slot=None, delay_cost_vect: np.array=None,
                 cost_vect: np.array = None, **garbage):

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

        # udpp attributes

        self.udppPriority = None

        self.tna = None

        self.udppPriorityNumber = None

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return str(self.name)

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

    def set_compatible_slots(self, slots: List[Slot], alternative_allocation_rule=False):
        self.compatibleSlots = compatible_slots(slots, self.eta, alternative_rule=alternative_allocation_rule)

    def set_not_compatible_slots(self, slots: List[Slot]):
        not_compatible_slots = [slot for slot in slots if slot not in self.compatibleSlots]
        self.notCompatibleSlots = not_compatible_slots

    def set_eta_slot(self, slots):
        # i = 0
        # #print ('ALLOALLO flight name, eta, delta_t, slots:', self.name, self.eta, delta_t, slots)
        # while slots[i].time < self.eta-delta_t:# and i<len(slots)-1:
        #     #print ('KWABUNGA', slots[i].time)
        #     i += 1
        self.etaSlot = self.compatibleSlots[0]#slots[i]#[i-1]

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
