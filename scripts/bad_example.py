import sys
sys.path.insert(1, '../..')

from numpy import array
import pandas as pd
from collections import OrderedDict

from Hotspot import models, models_correspondence_cost_vect, models_correspondence_approx
from Hotspot import Engine, LocalEngine, HotspotHandler, print_allocation
from Hotspot.libs.other_tools import compute_cost, generate_comparison

class Flight:
	pass

f_airline = [(1245, 677), (1227, 660), (1201, 660), (1251, 660), (1213, 660), (1215, 660), (1177, 720), (1237, 660), (1183, 660), (1253, 660), (1173, 660), (1235, 660), (1163, 660), (1307, 738), (1257, 770), (1273, 660), (1199, 660), (1129, 660), (1207, 660), (1243, 660), (1149, 660), (1247, 660), (1275, 660), (1269, 660), (1121, 660), (1061, 660), (1239, 660), (1185, 660), (1263, 660), (1225, 660), (1077, 721), (1153, 660), (1195, 657), (1291, 736), (1181, 660), (1123, 700), (1155, 660), (1249, 660), (1219, 660), (1293, 677), (1265, 660), (1221, 688), (1303, 660), (1241, 791), (1143, 668), (1203, 660), (1305, 660), (1167, 657), (1197, 669), (1099, 741), (1309, 698), (1131, 665), (1085, 660), (1073, 660), (1317, 674), (1135, 715), (1097, 660), (1079, 701), (1327, 694), (1089, 779), (1295, 656), (1271, 739), (1279, 673), (1287, 733), (1137, 779)]

flight_names = list(zip(*f_airline))[0]

airlines = {}
for flight, airline in f_airline:
	airlines[airline] = airlines.get(airline, []) + [flight]

etas = [1385.22, 1385.5, 1386.149, 1387.672, 1389.229, 1390.0339999999999, 1391.521,
	1393.3229999999999, 1394.276, 1395.064, 1397.5339999999999, 1399.501, 1401.666,
	1401.731, 1401.8899999999999, 1402.389, 1402.57, 1406.0149999999999, 1407.221,
	1409.942, 1411.853, 1412.306, 1412.653, 1412.7630000000001, 1412.903,
	1413.2079999999999, 1415.503, 1415.577, 1416.526, 1416.6709999999998,
	1417.184, 1417.389, 1417.591, 1421.154, 1421.3449999999998, 1423.944,
	1426.164839624805, 1427.2839999999999, 1427.857, 1428.489, 1428.714,
	1428.9582238963333, 1429.8509999999999, 1430.303, 1430.668, 1431.478,
	1432.176, 1434.169, 1434.6896389482479, 1435.502, 1436.069, 1437.1,
	1438.756, 1444.3632344854527, 1447.176, 1452.312, 1454.433481507698,
	1462.0900000000001, 1469.263, 1470.426, 1471.529, 1476.592, 1478.522,
	1483.057, 1483.339]
slot_times = [1385.0, 1386.0, 1387.0, 1388.0, 1389.0, 1390.0, 1391.0, 1393.0, 1394.0,
		1395.0, 1397.0, 1399.0, 1401.0, 1402.0, 1403.0, 1405.0, 1406.0, 1407.0,
		1409.0, 1410.0, 1411.0, 1412.0, 1413.0, 1414.0, 1415.0, 1417.0, 1418.0,
		1419.0, 1421.0, 1422.0, 1423.0, 1424.0, 1425.0, 1426.0, 1427.0, 1428.0,
		1429.0, 1430.0, 1431.0, 1432.0, 1433.0, 1434.0, 1435.0, 1436.0, 1437.0,
		1438.0, 1439.0, 1440.0, 1441.0, 1442.0, 1443.0, 1444.0, 1445.0, 1446.0,
		1447.0, 1452.0, 1454.0, 1462.0, 1469.0, 1470.0, 1471.0, 1476.0, 1478.0,
		1483.0, 1484.0]

df_cost = pd.read_csv('cost_matrix_debug_keep.csv', index_col=0)

costVec_dict = df_cost.T.to_dict(orient='list')

d_f_airline = OrderedDict(f_airline)

print ('Number of flights:', len(flight_names))
print ('Number of slots:', len(slot_times))
print ('Number of airlines:', len(airlines))

print ('First/last ETA:', min(etas), max(etas))
print ('First/last slot:', min(slot_times), max(slot_times))


#print(costVec_dict)

#raise Exception()

 
# costVec_dict = {'A6001': array([  0.  ,   0.  ,  90.83,  90.83,  90.83, 179.12, 179.12, 179.12,
#        179.12, 179.12, 179.12, 179.12, 179.12, 179.12, 179.12, 179.12,
#        179.12, 179.12, 179.12, 179.12]), 'A4003': array([  0.  ,   0.  ,   0.  , 103.42, 103.42, 103.42, 103.42, 103.42,
#        196.41, 196.41, 196.41, 196.41, 196.41, 196.41, 196.41, 196.41,
#        196.41, 196.41, 196.41, 196.41]), 'A4002': array([  0.  ,   0.  ,   0.  , 116.06, 116.06, 116.06, 116.06, 225.62,
#        225.62, 225.62, 225.62, 225.62, 225.62, 225.62, 225.62, 225.62,
#        225.62, 225.62, 225.62, 225.62]), 'A1000': array([  0.  ,   0.  ,   0.  ,   0.  , 111.32, 111.32, 111.32, 216.86,
#        216.86, 216.86, 216.86, 216.86, 216.86, 216.86, 216.86, 216.86,
#        216.86, 216.86, 216.86, 216.86]), 'A6000': array([  0.  ,   0.  ,   0.  ,   0.  , 111.4 , 111.4 , 111.4 , 214.83,
#        214.83, 214.83, 214.83, 214.83, 214.83, 214.83, 214.83, 214.83,
#        214.83, 214.83, 214.83, 214.83]), 'A2000': array([  0.  ,   0.  ,   0.  ,   0.  , 104.71, 104.71, 104.71, 104.71,
#        104.71, 188.47, 188.47, 188.47, 188.47, 188.47, 188.47, 188.47,
#        188.47, 188.47, 188.47, 188.47]), 'A4001': array([  0.  ,   0.  ,   0.  ,  97.23,  97.23,  97.23,  97.23,  97.23,
#        212.37, 212.37, 212.37, 212.37, 212.37, 212.37, 212.37, 212.37,
#        212.37, 212.37, 212.37, 212.37]), 'A5000': array([  0.  ,   0.  ,   0.  ,  82.01,  82.01,  82.01,  82.01,  82.01,
#        187.97, 187.97, 187.97, 187.97, 187.97, 187.97, 187.97, 187.97,
#        187.97, 187.97, 187.97, 187.97]), 'A3000': array([  0.  ,   0.  ,   0.  ,   0.  , 111.44, 111.44, 111.44, 111.44,
#        111.44, 214.11, 214.11, 214.11, 214.11, 214.11, 214.11, 214.11,
#        214.11, 214.11, 214.11, 214.11]), 'A4000': array([  0.  ,   0.  ,   0.  ,   0.  ,  88.14,  88.14,  88.14,  88.14,
#         88.14, 203.41, 203.41, 203.41, 203.41, 203.41, 203.41, 203.41,
#        203.41, 203.41, 203.41, 203.41])}
#airlines = {'A6': ['A6001', 'A6000'], 'A4': ['A4003', 'A4002', 'A4001', 'A4000'], 'A1': ['A1000'], 'A2': ['A2000'], 'A5': ['A5000'], 'A3': ['A3000']}

def test(algo = 'globaloptimum'):
	print ('ALGO TESTED:', algo)
	
	external_flights = [] 
	for i, name in enumerate(flight_names):
		flight = Flight()
		flight.name = name
		flight.eta = etas[i]
		flight.airlineName = d_f_airline[name] #name[:2]
		external_flights.append(flight)

	engine = Engine(algo=algo)

	#cost_func_archetype = 'jump2'
	hotspot_handler = HotspotHandler(engine=engine,
					#cost_func_archetype=cost_func_archetype,
					alternative_allocation_rule=True)

	slots, flights = hotspot_handler.prepare_hotspot_from_flights_ext(flights_ext=external_flights,
																		slot_times=slot_times,
																		attr_map={'name':'flight_name',
																				'airlineName':'airline_name',
																				'eta':'eta'
																				},
																		# set_cost_function_with={'cost_function':'cost_func',
																		# 						'kind':'lambda',
																		# 						'absolute':False,
																		# 						'eta':'eta'}
																		)
	for flight in flights.values():
		flight.costVect = costVec_dict[flight.name]

	# This computes the cost vectors if needed.
	hotspot_handler.prepare_all_flights()

	print ('FPFS allocation:')
	fpfs_allocation = hotspot_handler.get_allocation()
	print_allocation(fpfs_allocation)
	print ()

	# Run model
	allocation = engine.compute_optimal_allocation(hotspot_handler=hotspot_handler,
												kwargs_init={}
												)

	# Allocation is an ordered dict linking flight -> slot
	print ('Final allocation:')
	print_allocation(allocation)
	print ()
	print ('Reduction in cost (last stage of model, approximated costs:')
	engine.print_optimisation_performance()

	print ()
	print ('Reduction in cost (overall, real costs:')
	results = generate_comparison(fpfs_allocation, allocation, airlines, costVec_dict)
	print (results)

	# Put slots in the original flight object as attribute "attr_slot_ext"
	hotspot_handler.assign_slots_to_flights_ext_from_allocation(allocation, attr_slot_ext='slot')
	print ('Slot assigned to first flight:', external_flights[0].slot)

	# One can also use this method to use internal flights object to get the slot, instead of the
	# allocation dict.
	hotspot_handler.update_flight_attributes_int_to_ext({'newSlot':'slot'})
	print ('Slot assigned to last flight:', external_flights[-1].slot)

	# You can also get another shiny list of flights
	new_flights = hotspot_handler.get_new_flight_list()

	# you can see the results of each individual engine by accessing this attribute
	# print ('Merged results:', engine.model.merge_results)

	print ()
	print ()
	print ()

if __name__=='__main__':
	test(algo='nnbound')