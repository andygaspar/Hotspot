
import time
from data import dfMaker
import pandas as pd
from MIp import udppModelMip, istopMip, maxBenefitMip
import numpy as np

# df = pd.read_csv("../data/data_ruiz.csv")



total_initial = []
total_max_ben = []
total_udpp = []
total_model = []

simulations = pd.DataFrame(columns=["run", "airline", " num flights", "initial", "max_reduction", "udpp", "model", "offers"])

num_airlines = 12
num_fligths = 50
np.random.seed(2)
for i in range(1):
    t = time.time()
    print("iterazione ******** ", i)
    df = dfMaker.df_maker(num_fligths, num_airlines, distribution="uniform")
    df_max = df.copy(deep=True)
    df_amal = df.copy(deep=True)
    df_UDPP = df_amal.copy(deep=True)

    print(df)

    max_model = maxBenefitMip.MaxBenefitModel(df_max)
    max_model.run()
    print("max benefit")

    #amal_model = amal.Amal(df_amal, offerMakerFunType="1")
    #amal_model.run()

    udpp_model = udppModelMip.UDPPModel(df_UDPP)

    print("udpp")

    print(udpp_model.get_new_df()[["flight", "new slot", "new arrival", "eta slot", "cost"]])
    model = istopMip.Istop(udpp_model.get_new_df(), 0)
    model.run()
    print(model.offers)

    print(model.df)

    print("model")


    model1 = istopMip.Istop(df, 0)
    model1.run()
    # print("con base")
    # print(model1.offers)

    # total_initial.append(max_model.report["initial costs"][0])
    # total_max_ben.append(max_model.report["final costs"][0])
    # total_udpp.append(udpp_model.report["final costs"][0])
    # total_model.append(model.report["final costs"][0])

    # mmr = model.report
    # udppr = udpp_model.report
    # omr = model.report
    # offer = model.offers
    #
    # simulations = simulations.append(pd.Series([i, "total", num_fligths, mmr["initial costs"][0],
    #                                   mmr["final costs"][0], udppr["final costs"][0],
    #                                  omr["final costs"][0], offer["offers"][0]], index=simulations. columns),
    #                                  ignore_index=True)

    # for airline in mmr["airline"][1:]:
    #     air_num_flights = df[df["airline"] == airline].shape[0]
    #     simulations = simulations.append(pd.Series([i, airline, air_num_flights,
    #                                                 mmr[mmr["airline"] == airline]["initial costs"].values[0],
    #                                                 mmr[mmr["airline"] == airline]["final costs"].values[0],
    #                                                 udppr[udppr["airline"] == airline]["final costs"].values[0],
    #                                                 omr[omr["airline"] == airline]["final costs"].values[0],
    #                                                 offer[offer["airline"] == airline]["offers"].values[0]],
    #                                                index=simulations.columns),
    #                                      ignore_index=True)
    # print(offer[offer["airline"] == "total"])
    print(time.time()-t)
# print(simulations)
#simulations.to_csv("../results/increasing_50_1.csv", index=False)

# import matplotlib.pyplot as plt
#
# plt.plot(total_initial, label="inital")
# plt.plot(total_max_ben, label="max benefit")
# plt.plot(total_udpp, label="udpp")
# plt.plot(total_model, label="model")
# plt.legend()
# plt.show()



# slot = np.array(range(15))
# eta = slot
# airline = ["A","C","B","A","B","A","B","C","B","C","A","A","B","C","A"]
# cost =    [8 , 1 , 6  , 5 ,10 , 7 , 4 , 2 ,10 , 3 ,21 , 9 , 11 , 2 , 15]
# flights = [airline[i]+str(i) for i in range(len(airline))]
# gdp = eta * 2
#
# priority = cost
#
#
# df = pd.DataFrame({"slot": slot, "flight": flights, "eta": eta, "gdp schedule": gdp, "priority": priority, "airline": airline,
#          "cost": cost})