from typing import Union, List, Callable


def set_flights_cost_functions(self, costFun):
    if isinstance(costFun, Callable):
        for flight in self.flights:
            flight.set_cost_fun(costFun)
    else:
        i = 0
        for flight in self.flights:
            flight.set_cost_fun(costFun[i])
            i += 1