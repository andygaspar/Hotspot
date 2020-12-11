from UDPP.AIudpp.trainAuxFuns1 import make_batch

for i in range(100):
    inputs, outputs, airlines, UDPPmodels = make_batch(batchSize)