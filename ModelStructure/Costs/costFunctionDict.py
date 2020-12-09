from Programma.ModelStructure.modelStructure import ModelStructure
from Programma.ModelStructure.Flight.flight import Flight
from Programma.ModelStructure.Slot.slot import Slot


class CostFuns:

    def __init__(self):
        self.costFun = {

            "linear": lambda flight, slot: flight.cost * (slot.time - flight.eta),

            "quadratic": lambda flight, slot: (flight.cost * (slot.time - flight.eta) ** 2)/2,

            "step": lambda flight, slot: (slot.time - flight.eta) * flight.cost*10
            if (slot.time - flight.eta) < flight.margin else ((slot.time - flight.eta) * flight.cost*10 + flight.cost*10)

        }
