# Hotspot
Python software to simualate  a single  hotspot at one single airport.
This software allows to generate CCSs defining the capacity reduction, the number of flights involved, the number of arlines involved and the flight's cost 
functions. Given these inputs, initial costs (deriving from FPFS) are computed and stored.
Different models can be then applied: UDPP (with prioritisation as input), UDPP optimised (the local solution is automatically computed by a LP model),
NNBound, Istop. Within the UDPP folder there is also an attempt to generate an AI agent (via neural network) to generate the prioritisation.
