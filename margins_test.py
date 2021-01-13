from ModelStructure.ScheduleMaker import scheduleMaker

from ModelStructure.Costs.costFunctionDict import CostFuns
from UDPP import udppModel

# import matplotlib.pyplot as plt

# df = pd.read_csv("../data/data_ruiz.csv")
scheduleType = scheduleMaker.schedule_types(show=True)
# df = pd.read_csv("dfcrash")
df = scheduleMaker.df_maker(50, 5, distribution=scheduleType[0])
#df = scheduleMaker.df_maker(custom=[6, 4, 3, 7, 2, 8])
df_UDPP = df.copy(deep=True)
df_UDPP_opt = df_UDPP.copy(deep=True)

costFun = CostFuns().costFun["step"]
udpp = udppModel.UDPPmodel(df_UDPP_opt, costFun)
udpp_opt = udppModel.UDPPmodel(df_UDPP_opt, costFun)

udpp_opt.run(optimised=True)
udpp_opt.print_performance()
print(udpp_opt.solution)


udpp.run(optimised=False)
udpp.print_performance()
dfA = udpp.solution[udpp.solution["airline"] == "A"][["flight", "priority", "eta", "slot", "new slot", "margins"]]
print(dfA)
