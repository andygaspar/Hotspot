from GlobalFuns.globalFuns import HiddenPrints
from Istop import istop
from ModelStructure.ScheduleMaker import scheduleMaker
from ModelStructure.Costs.costFunctionDict import CostFuns
from ModelStructure.modelStructure import ModelStructure
from ModelStructure import df_to_schedule as converter
from NNBound import nnBound
import random

import numpy as np

random.seed(0)
np.random.seed(0)
scheduleType = scheduleMaker.schedule_types(show=False)

num_flights = 50
num_airlines = 5


distribution = scheduleType[3]
print("schedule type: ", distribution)

df = scheduleMaker.df_maker(num_flights, num_airlines, distribution=distribution)

costFun = CostFuns().costFun["realistic"]

fl_list = converter.make_flight_list(df, costFun)

m = ModelStructure(fl_list)

print(m.df)

print("\nnn bound")
max_model = nnBound.NNBoundModel(fl_list)
max_model.run()
max_model.print_performance()

"""
TO CONSIDER - TO DO

-slot holes due to multiple hotspot
-cost function definition

"""