import sys
sys.path.insert(1, '../..')

import pandas as pd

from Hotspot import *
import numpy as np
import random


# in case remember both
random.seed(0)
np.random.seed(0)


# ************* init

num_flights = 50
num_airlines = 5


df = pd.read_csv("Test/df_test.csv")
df_costs = pd.read_csv("Test/df_costs.csv")

slot_list, fl_list = make_flight_list(df, df_costs)

# fl_list has flight objects: you can now manually set slot, margin1, jump2, margin2, jump2

# **************** models run
global_model = GlobalOptimum(slot_list, fl_list)
global_model.run()


g_test = pd.read_csv("Test/global_solution.csv")
g_rep = pd.read_csv("Test/global_report.csv")
pd.testing.assert_frame_equal(g_rep, global_model.report)
# pd.testing.assert_frame_equal(g_test, global_model.solution)

nnb_model = NNBoundModel(slot_list, fl_list)
nnb_model.run()
nnb_test = pd.read_csv("Test/nnb_solution.csv")
nnb_rep = pd.read_csv("Test/nnb_report.csv")
pd.testing.assert_frame_equal(nnb_rep, nnb_model.report)

udpp_model_xp = UDPPmodel(slot_list, fl_list)
udpp_model_xp.run(optimised=True)
udpp_opt_test = pd.read_csv("Test/udpp_opt_solution.csv")
udpp_opt_rep = pd.read_csv("Test/udpp_opt_report.csv")
pd.testing.assert_frame_equal(udpp_opt_rep, udpp_model_xp.report)


new_fl_list = udpp_model_xp.get_new_flight_list()

xpModel = Istop(slot_list, new_fl_list, triples=False)
xpModel.run(True)

istop_test = pd.read_csv("Test/istop_solution.csv")
istop_rep = pd.read_csv("Test/istop_report.csv")
pd.testing.assert_frame_equal(istop_rep, xpModel.report)


print("Test completed successfully")
