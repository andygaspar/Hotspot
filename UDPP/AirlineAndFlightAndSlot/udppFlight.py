from ModelStructure.Flight import flight as fl


class UDPPflight(fl.Flight):

    def __init__(self, line, airline, model):

        super().__init__(line, airline, model)

        # UDPP attributes ***************

        self.UDPPLocalSlot = None

        self.UDPPlocalSolution = None

        self.priorityValue = "M"

        self.tna = self.margin

        self.tnb = self.eta

        self.test_slots = []

        self.priorityNumber = None

    def set_prioritisation(self, num: float, margin: int):
        pass

    def assign(self, solutionSlot):
        self.newSlot = solutionSlot
        solutionSlot.free = False
        solutionSlot.flight = self



