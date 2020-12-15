from ScheduleMaker import scheduleMaker
import random
from UDPP.udppModel import UDPPmodel
from ModelStructure.Costs.costFunctionDict import CostFuns
import numpy as np
import pandas as pd

from UDPP.AIudpp.trainAuxFuns1 import make_batch

final_df: pd.DataFrame

costFun = CostFuns().costFun["step"]

final_df = pd.DataFrame(columns=["instance", "airline", "margins", "priority", "eta", "slot", "new slot"])
print(final_df)

for i in range(100):
    df = scheduleMaker.df_maker(custom=[6, 4, 3, 7, 2, 8])
    df["margins"] = [random.choice(range(10, 50)) for i in range(df.shape[0])]
    udMod = UDPPmodel(df, costFun)
    udMod.run(optimised=True)
    udMod.solution["instance"] = (np.ones(udMod.solution.shape[0]) * i).astype(int)
    for airline in udMod.airlines:
        for flight in airline.flights:
            to_append = [i, flight.airline, flight.margin, flight.priority, flight.eta, flight.slot.time,
                         flight.newSlot.time]
            a_series = pd.Series(to_append, index=final_df.columns)
            final_df = final_df.append(a_series, ignore_index=True)

# standardisation
for col in final_df.columns[2:-1]:
    final_df[col] = (final_df[col] - final_df[col].mean()) / final_df[col].std()

final_df.to_csv("50_5_increase.csv")
