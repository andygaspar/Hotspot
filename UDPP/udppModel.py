import string
from typing import Union, Callable, List

import pandas as pd

from GlobalFuns.globalFuns import HiddenPrints
from ModelStructure.modelStructure import ModelStructure
from UDPP.LocalOptimised.udppLocalOpt import UDPPlocalOpt
from UDPP.udppMerge import UDPPmerge
from ModelStructure.Solution import solution
from UDPP.AirlineAndFlightAndSlot.udppAirline import UDPPairline
from UDPP.AirlineAndFlightAndSlot.udppFlight import UDPPflight
from UDPP.Local.udppLocal import udpp_local
from ModelStructure.Slot.slot import Slot
import time

import ModelStructure.modelStructure as ms


class UDPPmodel(ModelStructure):

    def __init__(self, df_init: pd.DataFrame, costFun: Union[Callable, List[Callable]]):

        super().__init__(df_init=df_init, costFun=costFun, airline_ctor=UDPPairline)

    def run(self, optimised=True):
        airline: UDPPairline
        start = time.time()
        for airline in self.airlines:
            if optimised:
                with HiddenPrints():
                    UDPPlocalOpt(airline, self.slots)

            else:
                udpp_local(airline, self.slots)

        UDPPmerge(self.flights, self.slots)
        # print(time.time() - start)
        solution.make_solution(self)
        for flight in self.flights:
            if flight.eta > flight.newSlot.time:
                print("************** damage, some negative impact has occured****************",
                      flight, flight.eta, flight.newSlot.time)

    def get_new_df(self):
        self.df: pd.DataFrame
        new_df = self.solution.copy(deep=True)
        new_df.reset_index(drop=True, inplace=True)
        new_df["slot"] = new_df["new slot"]
        new_df["fpfs"] = new_df["new arrival"]
        return new_df

    @staticmethod
    def compute_UDPP_local_cost(flights: List[UDPPflight]):
        return sum([flight.costFun(flight, Slot(None, flight.UDPPlocalSolution)) for flight in flights])

    def change_CCS(self, percentage: int):
        for flight in self.flights:
            flight.slot.time = flight.eta * 3
            self.initialTotalCosts = self.compute_costs(self.flights, "initial")

    def set_priority_value(self, val: string):
        for flight in self.flights:
            flight.priorityValue = val