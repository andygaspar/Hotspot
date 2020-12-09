class AmalOffer:

    def __init__(self, flight_down, at_most, flight_up, at_least):
        self.flightDown = flight_down
        self.atMost = at_most
        self.flightUp = flight_up
        self.atLeast = at_least

    def __repr__(self):
        return "("+self.flightDown.name + ","+str(self.atMost) + ";" \
               + self.flightUp.name + "," + str(self.atLeast) + ")"


