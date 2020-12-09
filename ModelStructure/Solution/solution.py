import numpy as np
import pandas as pd


def make_performance_df(model):
    from ModelStructure.modelStructure import ModelStructure
    from ModelStructure.Airline.airline import Airline
    from ModelStructure.Flight.flight import Flight
    model: ModelStructure
    airline: Airline
    flight: Flight
    airline_names = ["total"] + [airline.name for airline in model.airlines]
    num_flights = [model.numFlights]
    initial_costs = [model.initialTotalCosts]
    final_costs = [model.compute_costs(model.flights, "final")]
    reduction = [np.round(
        10000 * (model.initialTotalCosts - model.compute_costs(model.flights, "final")) / model.initialTotalCosts
        ) / 100
    ]
    initial_delay = ["-"]
    final_delay = ["-"]
    for airline in model.airlines:
        num_flights.append(airline.numFlights)
        initial_costs.append(model.compute_costs(airline.flights, "initial"))
        final_costs.append(model.compute_costs(airline.flights, "final"))
        initial_delay.append(model.compute_delays(airline.flights, "initial"))
        final_delay.append(model.compute_delays(airline.flights, "final"))
        reduction.append(np.round(
            10000 * (model.compute_costs(airline.flights, "initial")
                     - model.compute_costs(airline.flights, "final")) /
            model.compute_costs(airline.flights, "initial")
        ) / 100
                         )

    model.report = pd.DataFrame(
        {"airline": airline_names, "num flights": num_flights, "initial costs": initial_costs,
         "final costs": final_costs,
         "reduction %": reduction})  # "initial delay": initial_delay, "final delay": final_delay})


def make_df_solution(model):
    from ModelStructure.modelStructure import ModelStructure
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
    from ModelStructure.modelStructure import ModelStructure
    from ModelStructure.Airline.airline import Airline
    from ModelStructure.Flight.flight import Flight
    model: ModelStructure
    airline: Airline
    flight: Flight
    make_df_solution(model)
    make_performance_df(model)
