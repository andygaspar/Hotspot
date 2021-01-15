from Istop import istop
from ModelStructure.ScheduleMaker import scheduleMaker

from ModelStructure.Costs.costFunctionDict import CostFuns
from UDPP import udppModel
import pandas as pd

# import matplotlib.pyplot as plt
import numpy as np
# df = pd.read_csv("../data/data_ruiz.csv")
np.random.seed(0)
scheduleType = scheduleMaker.schedule_types(show=True)

num_flights = 50
num_airlines = 4
# df = pd.read_csv("dfcrash")
# df = scheduleMaker.df_maker(50, 4, distribution=scheduleType[3])
df = scheduleMaker.df_maker(num_flights, num_airlines, distribution=scheduleType[3])
df_max = df.copy(deep=True)
df_UDPP = df_max.copy(deep=True)
costFun = CostFuns().costFun["realistic"]
udpp_model_xp = udppModel.UDPPmodel(df_UDPP, costFun)
udpp_model_xp.run(optimised=True)
data = udpp_model_xp.report
data["run"] = [0 for i in range(num_airlines+1)]

for i in range(1, 2):
    df = scheduleMaker.df_maker(num_flights, num_airlines, distribution=scheduleType[3])
    # df.to_csv("df_crah")
    # df = pd.read_csv("df_crah")

    df_max = df.copy(deep=True)
    df_UDPP = df_max.copy(deep=True)

    costFun = CostFuns().costFun["realistic"]



    # print("max from FPFS")
    # max_model = nnBound.NNBoundModel(df_max, costFun)
    # max_model.run()
    # max_model.print_performance()

    """
    print("UDPPnonOpt from FPFS")
    udpp_model_xp = udppModel.UDPPmodel(df_UDPP, costFun)
    udpp_model_xp.run(optimised=False)
    udpp_model_xp.print_performance()
    
    print("max from UDPP")
    maxFromUDPP = nnBound.NNBoundModel(udpp_model_xp.get_new_df(), costFun)
    maxFromUDPP.run()
    maxFromUDPP.print_performance()
    
    
    print("istop from UDPP")
    xpModel = istop.Istop(udpp_model_xp.get_new_df(), costFun)
    xpModel.run(True)
    xpModel.print_performance()
    """


    print("UDPP Opt from FPFS")
    # print(df_UDPP)
    udpp_model_xp = udppModel.UDPPmodel(df_UDPP, costFun)
    # ff = udpp_model_xp.flights[0]
    # plt.plot([slot.index for slot in udpp_model_xp.slots], [ff.costFun(ff, slot) for slot in udpp_model_xp.slots])
    # plt.savefig("mygraph.png")

    udpp_model_xp.run(optimised=True)
    udpp_model_xp.print_performance()
    print(udpp_model_xp.solution)
    udpp_model_xp.report["run"] = [i for j in range(num_airlines+1)]
    # ff = udpp_model_xp.flights[0]
    # plt.plot([slot.index for slot in udpp_model_xp.slots], [ff.costFun(ff, slot) for slot in udpp_model_xp.slots])
    # plt.savefig("mygraph.png")
    #data = data.append(udpp_model_xp.report, ignore_index = True)

    # print("max from UDPP")
    # maxFromUDPP = nnBound.NNBoundModel(udpp_model_xp.get_new_df(), costFun)
    # maxFromUDPP.run()
    # maxFromUDPP.print_performance()

    print("istop from UDPP opt")
    xpModel = istop.Istop(udpp_model_xp.get_new_df(), costFun)
    xpModel.run(True)
    xpModel.print_performance()

#data.to_csv("50flights.csv")
#print(data)