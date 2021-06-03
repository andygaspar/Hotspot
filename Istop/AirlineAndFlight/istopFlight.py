import numpy as np

from Hotspot.ModelStructure.Flight import flight as fl
from Hotspot.Istop.Preferences import preference

def set_automatic_preference_vect(self, max_delay):
    self.slope, self.margin1, self.jump1, self.margin2, self.jump2 = \
        preference.make_preference_fun(max_delay, self.delayCostVect)

def not_paramtrised(self):
    return self.slope == self.margin1 == self.jump2 == self.margin2 == self.jump2 is None

def set_fit_vect(self):
    if self.slope == self.margin1 == self.jump1 == self.margin2 == self.jump2 is None:
        self.fitCostVect = self.costVect

    elif self.slope is not None and self.margin1 == self.jump1 == self.margin2 == self.jump2 is None:
        self.fitCostVect = preference.approx_linear(self.delayVect, self.slope)

    elif self.slope == self.margin1 == self.jump1 is not None and self.margin2 == self.jump2 is None:
        self.fitCostVect = preference.approx_slope_one_margin(self.delayVect, self.slope, self.margin1, self.jump1)

    elif self.slope == self.margin1 == self.jump1 == self.margin2 == self.jump2 is not None:
        self.fitCostVect = preference.approx_slope_two_margins(
            self.delayVect, self.slope, self.margin1, self.jump1, self.margin2, self.jump2)

    elif self.slope == self.margin2 == self.jump2 is None and self.margin1 == self.jump1  is not None:
        self.fitCostVect = preference.approx_one_margins(self.delayVect, self.margin1, self.jump1)

    elif self.slope is None and self.margin1 == self.jump1 == self.margin2 == self.jump2 is not None:
        self.fitCostVect = preference.approx_two_margins(
            self.delayVect, self.margin1, self.jump1, self.margin2, self.jump2)

def wrap_flight_istop(flight):
    flight.priority = None
    flight.fitCostVect = flight.costVect
    flight.flight_id = None
    flight.standardisedVector = None

    flight.set_automatic_preference_vect = set_automatic_preference_vect.__get__(flight)
    flight.not_paramtrised = not_paramtrised.__get__(flight)
    flight.set_fit_vect = set_fit_vect.__get__(flight)


# class IstopFlight(fl.Flight):

#     def __init__(self, flight: fl.Flight):

#         super().__init__(**flight.get_attributes())

#         self.priority = None

#         self.fitCostVect = self.costVect

#         self.flight_id = None

#         self.standardisedVector = None

#     def set_automatic_preference_vect(self, max_delay):
#         self.slope, self.margin1, self.jump2, self.margin2, self.jump2 = \
#             preference.make_preference_fun(max_delay, self.delayCostVect)

#     def not_paramtrised(self):
#         return self.slope == self.margin1 == self.jump2 == self.margin2 == self.jump2 is None

#     def set_fit_vect(self):

#         if self.slope == self.margin1 == self.jump1 == self.margin2 == self.jump2 is None:
#             self.fitCostVect = self.costVect

#         elif self.slope is not None and self.margin1 == self.jump1 == self.margin2 == self.jump2 is None:
#             self.fitCostVect = preference.approx_linear(self.delayVect, self.slope)

#         elif self.slope == self.margin1 == self.jump1 is not None and self.margin2 == self.jump2 is None:
#             self.fitCostVect = preference.approx_slope_one_margin(self.delayVect, self.slope, self.margin1, self.jump1)

#         elif self.slope == self.margin1 == self.jump1 == self.margin2 == self.jump2 is not None:
#             self.fitCostVect = preference.approx_slope_two_margins(
#                 self.delayVect, self.slope, self.margin1, self.jump1, self.margin2, self.jump2)

#         elif self.slope == self.margin2 == self.jump2 is None and self.margin1 == self.jump1  is not None:
#             self.fitCostVect = preference.approx_one_margins(self.delayVect, self.margin1, self.jump1)

#         elif self.slope is None and self.margin1 == self.jump1 == self.margin2 == self.jump2 is not None:
#             self.fitCostVect = preference.approx_two_margins(
#                 self.delayVect, self.margin1, self.jump1, self.margin2, self.jump2)

