from ScheduleMaker import scheduleMaker
from NNBound import nnBound

from Istop import istop

from ModelStructure.Costs.costFunctionDict import CostFuns
from UDPP import udppModel
import random
import pandas as pd

# df = pd.read_csv("../data/data_ruiz.csv")
scheduleType = scheduleMaker.schedule_types(show=True)
# df = pd.read_csv("dfcrash")
# df = scheduleMaker.df_maker(50, 4, distribution=scheduleType[3])
for i in range(30):
    df = scheduleMaker.df_maker(custom=[10,10,10,10,10])
    print(df)
    df["margins"] = [random.choice(range(10, 50)) for i in range(df.shape[0])]
    # df.to_csv("dfcrash")
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
    udpp_model_xp = udppModel.UDPPmodel(df_UDPP, costFun)
    udpp_model_xp.run(optimised=True)
    udpp_model_xp.print_performance()

    # print("max from UDPP")
    # maxFromUDPP = nnBound.NNBoundModel(udpp_model_xp.get_new_df(), costFun)
    # maxFromUDPP.run()
    # maxFromUDPP.print_performance()

    print("istop from UDPP opt")
    xpModel = istop.Istop(udpp_model_xp.get_new_df(), costFun)
    xpModel.run(True)
    xpModel.print_performance()


