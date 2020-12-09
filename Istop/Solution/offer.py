import Programma.Istop.Solution.solution as sol
import copy


class Offer:

    def __init__(self, flight_offer_list):
        self.airline_a = flight_offer_list[0].airline
        self.airline_b = flight_offer_list[1].airline
        self.flight_offer_list = flight_offer_list

    def __str__(self):
        return self.flight_offer_list

    def __repr__(self):
        return str([str(flight) for flight in self.flight_offer_list])


class OffersList:

    def __init__(self, model):

        self.offers = self.get_offer(model)

    @staticmethod
    def get_offer(model):
        flight_list = model.flights
        selected_flights = []
        offers_list = []
        offer_flight_list = []
        for flight in flight_list:
            if flight.slot != flight.newSlot and flight not in selected_flights:
                offer_flight_list.clear()
                offer_flight_list.append(flight)
                for i in range(3):
                    flight = sol.get_flight(flight.newSlot, model.flights)
                    offer_flight_list.append(flight)
                selected_flights += offer_flight_list
                offers_list.append(Offer(copy.deepcopy(offer_flight_list)))
        return offers_list

    def __str__(self):
        str_to_return = ""
        for offer in self.offers:
            for flight in offer.flight_offer_list:
                str_to_return += (str(flight.slot)) + " " + (str(flight.newSlot)) + " " + (str(flight)) + "  --  "
            str_to_return += "\n"
        return str_to_return
