from typing import Callable, List, Union

from ..ModelStructure import modelStructure as mS

from ..ModelStructure.Flight.flight import Flight
from ..ModelStructure.Solution import solution
from ..ModelStructure.Slot.slot import Slot
from .SolversNNB.gurobi_solver_NNB import NNBoundGurobi
from ..GlobalFuns.globalFuns import HiddenPrints


class NNBoundModel(mS.ModelStructure):
    requirements = ['delayCostVect', 'costVect']

    def __init__(self, slots: List[Slot] = None, flights: List[Flight] = None,
                 xp_problem=None, alternative_allocation_rule=False):

        super().__init__(slots,
                         flights,
                         alternative_allocation_rule=alternative_allocation_rule)

    def run(self, timing=False, verbose=False, time_limit=60):

        m = NNBoundGurobi(self)
        solution_vect = m.run(timing=timing, verbose=verbose, time_limit=time_limit)

        self.assign_flights(solution_vect)
        solution.make_solution(self)

    def assign_flights(self, sol):
        for flight in self.flights:
            for slot in self.slots:
                if sol[flight.index, slot.index] > 0.5:
                    flight.newSlot = slot


