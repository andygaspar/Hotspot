import dill as pickle
import matplotlib
import tk
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fs = pd.read_csv('ModelStructure/ScheduleMaker/flight_schedules.csv')

with open('ModelStructure/Costs/cost_functions_all.pck', 'rb') as dbfile:
    #dbfile = open('./cost_functions_1.pck', 'rb')
    dict_cost_funct = pickle.load(dbfile)
dbfile.close()


flight_id = 33735
delay = 185
print("Cost without Reg261:", dict_cost_funct[flight_id](delay))
print("Cost with Reg261:", dict_cost_funct[flight_id](delay,reg_261=True),"or",dict_cost_funct[flight_id](delay,True))
print("Flight info fs[fs['nid']==flight_id]")
f_info = fs[fs['nid']==flight_id]
f_info


flight_id = 33735
del_min=-10
del_max=450
#
#
#
# fig, ax = plt.subplots(1, 1)
# ax.plot(np.arange(del_min,del_max), [dict_cost_funct[flight_id](x) for x in np.arange(del_min,del_max)])
# plt.show()
#
# fig, ax = plt.subplots(1, 1)
# ax.plot(np.arange(del_min,del_max), [dict_cost_funct[flight_id](x,True) for x in np.arange(del_min,del_max)])
# plt.show()
#
#
# fig, ax = plt.subplots(1, 1)
# ax.plot(np.arange(del_min,del_max), [dict_cost_funct[flight_id](x) for x in np.arange(del_min,del_max)])
# ax.plot(np.arange(del_min,del_max), [dict_cost_funct[flight_id](x,True) for x in np.arange(del_min,del_max)])
# plt.show()

print(dict_cost_funct)
keys = list(dict_cost_funct.keys())
for key in keys[10:15]:
  plt.plot(np.arange(del_min,del_max), [dict_cost_funct[key](x) for x in np.arange(del_min,del_max)])
plt.show()