from Programma.ModelStructure.Slot.slot import Slot
import numpy as np
import pandas as pd


def make_slots_list(df: pd.DataFrame):
    slots = []
    slotIndexes = df["slot"].to_numpy()

    slotTimes = df["time"].to_numpy()
    slotTimes.sort()

    for i in range(len(slotIndexes)):
        slots.append(Slot(i, slotTimes[i]))

    return np.array(slots)
