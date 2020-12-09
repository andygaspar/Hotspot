import numpy as np
import pandas as pd
from typing import List, Callable

from ModelStructure.Slot import slot
from ModelStructure.Airline import airline as air


def make_airlines_list(df: pd.DataFrame, slots: List[slot.Slot], airline_ctor: Callable):
    airline_list = []
    airline_index = 0

    for airline in df["airline"].unique():
        if airline != "Empty":
            airline_list.append(
                airline_ctor(df[df["airline"] == airline], airline_index, slots))
            airline_index += 1

    return np.array(airline_list)
