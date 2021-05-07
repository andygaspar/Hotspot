from ModelStructure.Flight import flight as fl


class UDPPflight(fl.Flight):

    def __init__(self, flight: fl.Flight):

        super().__init__(*flight.get_attributes())

        # UDPP attributes ***************

        self.UDPPLocalSlot = None

        self.UDPPlocalSolution = None

        self.priorityValue = "M"

        self.test_slots = []

        self.priorityNumber = None

    def set_prioritisation(self, num: float, margin: int):
        pass

    def assign(self, solutionSlot):
        self.newSlot = solutionSlot
        solutionSlot.free = False
        solutionSlot.flight = self



