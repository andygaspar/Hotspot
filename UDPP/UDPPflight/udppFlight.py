from ModelStructure.Flight import flight as fl


class UDPPflight(fl.Flight):

    def __init__(self, flight: fl.Flight):

        super().__init__(*flight.get_attributes())

        # UDPP attributes ***************

        self.UDPPLocalSlot = None

        self.UDPPlocalSolution = None

        self.test_slots = []

    def set_prioritisation(self, udpp_priority: str, udpp_priority_number: int = None, tna: float = None):
        self.udppPriority = udpp_priority
        self.udppPriorityNumber = udpp_priority_number
        self.tna = tna




