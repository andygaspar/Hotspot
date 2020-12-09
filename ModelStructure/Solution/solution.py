import numpy as np
import pandas as pd


def make_performance_df(model):
    from Programma.ModelStructure.modelStructure import ModelStructure
    from Programma.ModelStructure.Airline.airline import Airline
    from Programma.ModelStructure.Flight.flight import Flight
    model: ModelStructure
    airline: Airline
    flight: Flight
    airline_names = ["total"] + [airline.name for airline in model.airlines]
    initial_costs = [model.initialTotalCosts]
    final_costs = [model.compute_costs(model.flights, "final")]
    initial_delay = ["-"]
    final_delay = ["-"]
    for airline in model.airlines:
        initial_costs.append(model.compute_costs(airline.flights, "initial"))
        final_costs.append(model.compute_costs(airline.flights, "final"))
        initial_delay.append(model.compute_delays(airline.flights, "initial"))
        final_delay.append(model.compute_delays(airline.flights, "final"))

    model.report = pd.DataFrame(
        {"airline": airline_names, "initial costs": initial_costs, "final costs": final_costs,
         "initial delay": initial_delay, "final delay": final_delay})


def make_df_solution(model):
    from Programma.ModelStructure.modelStructure import ModelStructure
    model: ModelStructure

    model.solution = model.df.copy(deep=True)

    new_slot = [flight.newSlot.index for flight in model.flights]
    new_arrival = [flight.newSlot.time for flight in model.flights]
    eta_slot = [flight.etaSlot for flight in model.flights]
    model.solution["new slot"] = new_slot
    model.solution["new arrival"] = new_arrival
    model.solution["eta slot"] = eta_slot
    model.solution.sort_values(by="new slot", inplace=True)


def make_solution(model):

    from Programma.ModelStructure.modelStructure import ModelStructure
    from Programma.ModelStructure.Airline.airline import Airline
    from Programma.ModelStructure.Flight.flight import Flight
    model: ModelStructure
    airline: Airline
    flight: Flight
    make_df_solution(model)
    make_performance_df(model)
