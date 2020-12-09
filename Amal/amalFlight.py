import numpy as np
from Programma.ModelStructure.Flight import flight as fl


class AmalFlight(fl.Flight):

    def __init__(self, line, airline, model):

        super().__init__(line, airline, model)

        self.priority = line["priority"]

        self.currentDelay = self.fpfs - self.eta

        self.classes = None

        self.slot_range = None

    def set_flight_offer_properties(self, flight_offer_list):
        self.classes = np.unique(np.sort(flight_offer_list))
        self.slot_range = range(self.classes[0], self.classes[-1] + 1)

    def class_range(self, k):
        if k == 0:
            return range(self.etaSlot, self.classes[k] + 1)

        return range(self.classes[k-1]+1, self.classes[k]+1)



