import os
import sys

from collections import OrderedDict


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def preferences_from_flights(flights, paras={}, if_absent=None):
    d = OrderedDict([(flight.name, {}) for flight in flights])

    for flight in flights:
        d[flight.name] = {}
        for para in paras:
            d[flight.name][para] = getattr(flight, para, if_absent)

    return d

