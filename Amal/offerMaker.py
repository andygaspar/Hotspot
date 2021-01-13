

def remove_flight(flight_list, flight):
    new_list = []
    for f in flight_list:
        if f != flight:
            new_list.append(f)
    return new_list


def convenient_offer(flight, f_offer, otherFlight, of_offer, model):
    from Programma.ModelStructure.modelStructure import ModelStructure
    model: ModelStructure
    actual_cost = model.cost_function(flight, flight.slot) + model.cost_function(otherFlight, otherFlight.slot)
    offer_cost = model.cost_function(flight, f_offer) + model.cost_function(otherFlight, of_offer)

    # if offer_cost < actual_cost:
    #     print("")
    return offer_cost < actual_cost


def make_offer_list(model, airline):
    from Programma.Amal import amalFlight as modFl
    from Programma.Amal.amalOffer import AmalOffer
    from Programma.Amal import amalAirline

    airline: amalAirline.AmalAirline

    offer_list = []
    flight: modFl.AmalFlight
    otherFlight: modFl.AmalFlight
    for flight in airline.flights:
        default_offer = AmalOffer(flight, flight.slot, flight, flight.slot)
        offer_list.append(default_offer)
        other_flights = remove_flight(airline.flights, flight)

        for otherFlight in other_flights:
            if model.offerMakerFunType == "1":

                flight_offer = flight.etaSlot
                exchange_offer = model.slotIndexes[-1]
                if convenient_offer(flight, flight_offer, otherFlight, exchange_offer, model):
                    offer_list.append(AmalOffer(otherFlight, exchange_offer, flight, flight_offer))


            if model.offerMakerFunType == "2":

                if convenient_offer(flight, flight_offer, otherFlight, exchange_offer, model):
                    offer_list.append(AmalOffer(otherFlight, exchange_offer, flight, flight_offer))
    return offer_list

"""
    def from_csv(self, df_airline):
        amal_offers = []
        df = df_airline[df_airline["amal offer"].notna()]
        if df.shape[0] > 0:
            num_offers = int(max(df["amal offer"]))
            for off_num in range(1, num_offers + 1):
                d = df[df["amal offer"] == off_num]
                flight_down = d[d["amal offer kind"] == "D"]["flight"].values[0]
                at_most = d[d["amal offer kind"] == "D"]["time not after"].values[0]
                flight_up = d[d["amal offer kind"] == "U"]["flight"].values[0]
                at_least = d[d["amal offer kind"] == "D"]["time not after"].values[0]
                amal_offers.append(AmalOffer(self.get_flight_by_name(flight_down), at_most,
                                             self.get_flight_by_name(flight_up), at_least))

        return amal_offers

"""