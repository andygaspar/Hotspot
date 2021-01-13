from ModelStructure.modelStructure import ModelStructure
from ModelStructure.Flight.flight import Flight
from ModelStructure.Slot.slot import Slot


class CostFuns:

    def __init__(self):
        self.costFun = {

            "linear": lambda flight, slot: flight.cost * (slot.time - flight.eta),

            "quadratic": lambda flight, slot: (flight.cost * (slot.time - flight.eta) ** 2)/2,

            "step": lambda flight, slot: 0 if slot.time - flight.eta < 0 else (slot.time - flight.eta) * flight.cost
            if (slot.time - flight.eta) < flight.margin else
            ((slot.time - flight.eta) * flight.cost*10 + flight.cost * 10),

            "jump": lambda flight, slot: 0 if slot.time - flight.eta < 0 else (slot.time - flight.eta) * flight.cost
            if (slot.time - flight.eta) < flight.margin else
            ((slot.time - flight.eta) * flight.cost + flight.cost) + flight.jump

        }
