from typing import Union, List

from pathlib import Path

import numpy as np
import random
import string
from scipy import stats
import pandas as pd

from ..ModelStructure.Slot.slot import Slot
from ..ModelStructure.Flight.flight import Flight


def avoid_zero(flight_list, num_flights):
    while len(flight_list[flight_list < 1]) > 0:
        for i in range(flight_list.shape[0]):
            if flight_list[i] == 0:
                flight_list[i] += 1
                if sum(flight_list) > num_flights:
                    flight_list[np.argmax(flight_list)] -= 1
    return flight_list


def fill_missing_flights(flight_list, num_flights, num_airlines):
    missing = num_flights - sum(flight_list)
    for i in range(missing):
        custm = stats.rv_discrete(name='custm', values=(np.arange(num_airlines), flight_list / sum(flight_list)))
        flight_list[custm.rvs(size=1)] += 1
    return np.flip(np.sort(flight_list))


def distribution_maker(num_flights, num_airlines, distribution="uniform"):
    dist = []

    if distribution == "uniform":
        h, loc = np.histogram(np.random.uniform(0, 1, 1000), bins=num_airlines)
        dist = ((np.flip(np.sort(h)) / sum(h)) * num_flights).astype(int)

    if distribution == "few_high_few_low":
        f = lambda x: x ** 3 + 1
        base = np.linspace(-1, 1, num_airlines)
        custm = stats.rv_discrete(name='custm', values=(np.arange(num_airlines), f(base) / sum(f(base))))
        h, loc = np.histogram(custm.rvs(size=1000), bins=num_airlines)
        dist = ((np.flip(np.sort(h)) / sum(h)) * num_flights).astype(int)

    if distribution == "few_low":
        f = lambda x: x ** 4 + 1
        base = np.linspace(-1, 1 / 4, num_airlines)
        custm = stats.rv_discrete(name='custm', values=(np.arange(num_airlines), f(base) / sum(f(base))))
        h, loc = np.histogram(custm.rvs(size=1000), bins=num_airlines)
        dist = ((np.flip(np.sort(h)) / sum(h)) * num_flights).astype(int)

    if distribution == "few_high":
        f = lambda x: x ** 2
        base = np.linspace(0, 1, num_airlines)
        val = f(base)
        val[val > 3 / 4] = 3 / 4
        custm = stats.rv_discrete(name='custm', values=(np.arange(num_airlines), val / sum(val)))
        h, l = np.histogram(custm.rvs(size=1000), bins=num_airlines)
        dist = ((np.flip(np.sort(h)) / sum(h)) * num_flights).astype(int)

    if distribution == "increasing":
        f = lambda x: x
        base = np.linspace(0, 1, num_airlines)
        custm = stats.rv_discrete(name='custm', values=(np.arange(num_airlines), f(base) / sum(f(base))))
        h, loc = np.histogram(custm.rvs(size=1000), bins=num_airlines)
        dist = ((np.flip(np.sort(h)) / sum(h)) * num_flights).astype(int)

    if distribution == "hub":
        f = lambda x: x ** 10
        base = np.linspace(0, 1, num_airlines)
        custm = stats.rv_discrete(name='custm', values=(np.arange(num_airlines), f(base) / sum(f(base))))
        h, loc = np.histogram(custm.rvs(size=1000), bins=num_airlines)
        dist = ((np.flip(np.sort(h)) / sum(h)) * num_flights).astype(int)

    dist = avoid_zero(dist, num_flights)
    dist = fill_missing_flights(dist, num_flights, num_airlines)
    return dist


def df_maker(*args, **kwargs):

    slots, flights, eta, fpfs, priority, margins_gap, airline, cost, num,jump, flights_type = schedule_maker(*args, **kwargs)

    return pd.DataFrame(
        {"slot": slots, "flight": flights, "eta": eta, "fpfs": fpfs, "time": fpfs, "priority": priority,
         "margins":margins_gap, "airline": airline, "cost": cost, "num": num, "jump":jump, "type": flights_type})

def slots_flights_maker(*args, **kwargs):
    # TODO modify schedule_maker to inclde margin2/jump2
    slots, flights, etas, fpfss, priorities, margins_gap, airlines, costs, nums, jumps, flights_types = schedule_maker(*args, **kwargs)

    slots_obj, flights_obj = [], []
    for i in range(len(slots)):
        slot_index = slots[i]
        slot_time = fpfss[i]
        slot = Slot(slot_index, slot_time)

        flight_name = flights[i]
        airline_name = airlines[i]
        eta = etas[i]
        delay_cost_vect = None
        udpp_priority = priorities[i]
        tna = None
        slope = costs[i]
        margin_1 = margins_gap[i]
        jump_1 = jumps[i]
        margin_2 = None
        jump_2 = None

        flight = Flight(slot,
                        flight_name,
                        airline_name,
                        eta,
                        delay_cost_vect,
                        udpp_priority=udpp_priority,
                        tna=tna,
                        slope=slope,
                        margin_1=margin_1,
                        jump_1=jump_1,
                        margin_2=margin_2,
                        jump_2=jump_2)

        slots_obj.append(slot)
        flights_obj.append(flight)

    return slots_obj, flights_obj

def schedule_maker(num_flights=20, num_airlines=3, distribution="uniform", capacity=1, new_capacity=2,
    n_flights_first_airlines=None, custom:Union[None, List[int]]= None, min_margin=10,
    max_margin=45, min_jump=10, max_jump=100):

    """
    n_flights_first_airlines can be passed as list of ints and represents number
    of fligihts per airlines. For instance, num_airlines=3 and n_flights_first_airlines=[3]
    can create distribution of flights [3, 4, 3] or [3, 5, 2] etc. All number of flights
    if given by n_flights_first_airlines, they are used as is, for instance 
    num_airlines=3 and n_flights_first_airlines=[1, 5, 4]
    """

    if not n_flights_first_airlines is None:
        if len(n_flights_first_airlines)==num_airlines and custom is None:
            custom = n_flights_first_airlines
        elif not n_flights_first_airlines is None:
            dist_other_flights = distribution_maker(num_flights-sum(n_flights_first_airlines),
                                                    num_airlines-len(n_flights_first_airlines),
                                                    distribution)
            custom = n_flights_first_airlines + list(dist_other_flights)

    if custom is None:
        dist = distribution_maker(num_flights, num_airlines, distribution)
        airline = [[string.ascii_uppercase[j] for i in range(dist[j])] for j in range(num_airlines)]
    else:
        num_airlines = len(custom)
        num_flights = sum(custom)
        airline = [[string.ascii_uppercase[j] for i in range(custom[j])] for j in range(num_airlines)]

    airline = [val for sublist in airline for val in sublist]
    airline = np.random.permutation(airline)
    flights = ["F" + airline[i] + str(i) for i in range(num_flights)]

    slot = np.arange(num_flights)
    eta = slot * capacity
    fpfs = slot * new_capacity
    priority = np.random.uniform(0.5, 2, num_flights)
    priority = []
    for i in range(num_flights):
        m = np.random.choice([0, 1])
        if m == 0:
            priority.append(np.random.normal(0.7, 0.1))
        else:
            priority.append(np.random.normal(1.5, 0.1))

    priority = np.abs(priority)
    cost = priority

    num = range(num_flights)
    margins_gap = np.array([random.choice(range(min_margin, max_margin)) for i in num])
    dir_path = Path(__file__).resolve().parent.parent
    at_gate = pd.read_csv(dir_path / "ModelStructure/Costs/costs_table_gate.csv", sep=" ")
    flights_type = [np.random.choice(at_gate["flight"].to_numpy()) for i in range(num_flights)]
    jump = np.random.randint(min_jump, max_jump, len(num))
    
    return slot, flights, eta, fpfs, priority, margins_gap, airline, cost, num,jump, flights_type

def schedule_types(show=False):
    dfTypeList = ("uniform", "few_low", "few_high", "increasing", "hub")
    if show:
        print(dfTypeList)
    return dfTypeList
