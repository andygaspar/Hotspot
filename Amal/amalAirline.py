import pandas as pd
from Programma.Amal import amalFlight as modFl
from Programma.ModelStructure.Airline import airline as air
from Programma.Amal import offerMaker as oM


class AmalAirline(air.Airline):

    def __init__(self, df_airline: pd.DataFrame, airline_index, model):

        flight: modFl.AmalFlight

        super().__init__(df_airline, airline_index, model)

        self.offerList = None

    def get_offers_for_flight(self, flight):

        offer_for_flight_list = []

        for offer in self.offerList:
            if offer.flightDown == flight:
                offer_for_flight_list.append(offer.atMost)
            elif offer.flightUp == flight:
                offer_for_flight_list.append(offer.atLeast)

        return offer_for_flight_list

    def set_offers(self, model):

        self.offerList = oM.make_offer_list(model, self)

        for flight in self.flights:
            flight.set_flight_offer_properties(self.get_offers_for_flight(flight))
