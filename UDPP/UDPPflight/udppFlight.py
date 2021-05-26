#from Hotspot.ModelStructure.Flight import flight as fl

def set_prioritisation(self, udpp_priority: str, udpp_priority_number: int = None, tna: float = None):
   self.udppPriority = udpp_priority
   self.udppPriorityNumber = udpp_priority_number
   self.tna = tna
 
# Add attributes and methods to a given flight object.
def wrap_flight_udpp(flight):
   flight.set_prioritisation = set_prioritisation.__get__(flight)
   flight.UDPPLocalSlot = None
   flight.UDPPlocalSolution = None
   flight.test_slots = []

# class UDPPflight(fl.Flight):

#     def __init__(self, flight: fl.Flight):
#         super().__init__(**flight.get_attributes())
        
#         # UDPP attributes ***************

#         self.UDPPLocalSlot = None

#         self.UDPPlocalSolution = None

#         self.test_slots = []

#     def set_prioritisation(self, udpp_priority: str, udpp_priority_number: int = None, tna: float = None):
#         self.udppPriority = udpp_priority
#         self.udppPriorityNumber = udpp_priority_number
#         self.tna = tna




