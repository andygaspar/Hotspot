from Hotspot.ModelStructure.Flight import flight as fl


class IstopFlight(fl.Flight):

    def __init__(self, line, airline, model):

        super().__init__(line, airline, model)

        self.priority = line["priority"]

        self.preference = None

    def set_preference(self, sum_priorities, f):
        self.preference = self.compute_preference(self.airline.numFlights, sum_priorities, f)

    def compute_preference(self, num_flights, sum_priorities, f):
        if sum_priorities < 1:
            sum_priorities = 1
        return f(self.priority, num_flights) / sum_priorities

    def set_priority(self, priority):
        self.priority = priority



