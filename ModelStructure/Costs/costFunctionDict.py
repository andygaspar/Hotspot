from ModelStructure.modelStructure import ModelStructure
from ModelStructure.Flight.flight import Flight
from ModelStructure.Slot.slot import Slot
import pandas as pd
import os

at_gate = pd.read_csv("ModelStructure/Costs/costs_table_gate.csv", sep=" ")
delay_range = list(at_gate.columns[1:].astype(int))

def get_interval(time):
    for i in range(len(delay_range)-1):
        if delay_range[i] <= time < delay_range[i+1]:
            return i

def compute_gate_costs(flight, slot):
    i = get_interval(slot.time)
    y2 = at_gate[at_gate["flight"] == flight.type][str(delay_range[i+1])].values[0]
    y1 = at_gate[at_gate["flight"] == flight.type][str(delay_range[i])].values[0]
    x2 = delay_range[i+1]
    x1 = delay_range[i]
    return y1 + (slot.time - x1)*(y2 - y1)/(x2 - x1)



class CostFuns:

    def __init__(self):
        self.costFun = {

            "linear": lambda flight, slot: flight.cost * (slot.time - flight.eta),

            "quadratic": lambda flight, slot: (flight.cost * (slot.time - flight.eta) ** 2)/2,

            "step": lambda flight, slot: 0 if slot.time - flight.eta < 0 else (slot.time - flight.eta) * flight.cost
            if (slot.time - flight.eta) < flight.margin else
            ((slot.time - flight.eta) * flight.cost*10 + flight.cost * 30),

            "gate": lambda flight, slot: compute_gate_costs(flight, slot),

            "realistic" : 2

        }


