# Hotspot

Hotspot is a library for Air Traffic Management modelling. It allows to solve an Air Traffic Flow Management regulation, i.e. finding an allocation of flights to regulation slots, with several algorithms:
- User-Driven Prioritisation Process, which includes an optimiser for airlines to use it efficiently,
- MINCOST, which finds the minimum total cost
- NNBound, which finds the minimum total cost, without any airline losing from the final allocation,
- ISTOP, a special mechanism that allows global minimisation of surrogate cost functions, called penalty functions.


*usage and more details to follow*

More details about this library can be found here:

*UDPP article to cite*

The following articles used the Hostpot library:

Gérald Gurtner, Tatjana Bolić, 2023. "Impact of cost approximation on the efficiency of collaborative regulation resolution mechanisms", Journal of Air Transport Management, Volume 113, 102471,ISSN 0969-6997, [https://doi.org/10.1016/j.jairtraman.2023.102471](https://doi.org/10.1016/j.jairtraman.2023.102471)


# Authorship

The core Hotspot library has been designed and written by Andrea Gasparin of the Università degli Studi di Trieste, Italy. The utility wrapper and other contributions were made by Gérald Gurtner, University of Westminster, United Kingdom.

Hotspot has been developed partly within the BEACON project, funded by the European Union's Horizon 2020 -- SESAR research and innocation programme under Grant Agreement No. 893100. More information can be found here: [https://www.beacon-sesar.eu](https://www.beacon-sesar.eu)

Hotspot is licensed under the GPL v3 licence. You can find a copy of the licence in LICENCE.TXT.

Copyright 2023 Andrea Gasparin, Gérald Gurtner


