from typing import Callable, List, Union
from .SolversGO.gurobi_solver_go import GOGurobi
from ..ModelStructure import modelStructure as mS
from ..ModelStructure.Flight.flight import Flight
from ..ModelStructure.Solution import solution
from ..ModelStructure.Slot.slot import Slot
from ..libs.other_tools import write_on_file as print_to_void


class GlobalOptimum(mS.ModelStructure):
    requirements = ['delayCostVect', 'costVect']

    def __init__(self, slots: List[Slot] = None, flights: List[Flight] = None, alternative_allocation_rule=0.):
        super().__init__(slots=slots, flights=flights, alternative_allocation_rule=alternative_allocation_rule)

    def run(self, timing=False, update_flights=False, time_limit=60):

        m = GOGurobi(self)
        # print("Using Gurobi")
        solution_vect = m.run(timing=timing, time_limit=time_limit)
        self.assign_flights(solution_vect)
        solution.make_solution(self)

        if update_flights:
            self.update_flights()

    def assign_flights(self, solution_vect):
        with print_to_void():
            for flight in self.flights:
                for slot in self.slots:
                    if solution_vect[flight.index, slot.index] > 0.5:
                        flight.newSlot = slot
