from ScheduleMaker import scheduleMaker
from NNBound import nnBound

from Istop import istop

from ModelStructure.Costs.costFunctionDict import CostFuns
from UDPP import udppModel
import random
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

# df = pd.read_csv("../data/data_ruiz.csv")
scheduleType = scheduleMaker.schedule_types(show=True)
# df = pd.read_csv("dfcrash")
# df = scheduleMaker.df_maker(50, 4, distribution=scheduleType[3])

for i in range(1):
    # df = scheduleMaker.df_maker(custom=[6, 4, 3, 7, 2, 8])
    # df["margins"] = [random.choice(range(10, 50)) for i in range(df.shape[0])]
    # df.to_csv("dfcrash")
    df= pd.read_csv("dfcrash")

    df_max = df.copy(deep=True)
    df_UDPP = df_max.copy(deep=True)

    costFun = CostFuns().costFun["step"]



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
    ff = udpp_model_xp.flights[0]
    # plt.plot([slot.index for slot in udpp_model_xp.slots], [ff.costFun(ff, slot) for slot in udpp_model_xp.slots])
    # plt.savefig("mygraph.png")

    udpp_model_xp.run(optimised=True)
    udpp_model_xp.print_performance()
    print(udpp_model_xp.solution)

    # print("max from UDPP")
    # maxFromUDPP = nnBound.NNBoundModel(udpp_model_xp.get_new_df(), costFun)
    # maxFromUDPP.run()
    # maxFromUDPP.print_performance()

    # print("istop from UDPP opt")
    # xpModel = istop.Istop(udpp_model_xp.get_new_df(), costFun)
    # xpModel.run(True)
    # xpModel.print_performance()


