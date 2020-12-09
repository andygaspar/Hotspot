from __future__ import annotations
from typing import Union


class Slot:

    def __init__(self, index: Union[int, None] = None, time: Union[int, None] = None):
        self.index = index
        self.time = time
        # self.flight = flight

    def __str__(self):
        return str(self.index)

    def __repr__(self):
        return str(self.index)+":"+str(self.time)

    def __eq__(self, other: Slot):
        return self.time == other.time

    def __lt__(self, other: Slot):
        return self.time < other.time

    def __le__(self, other: Slot):
        return self.time <= other.time

    def __gt__(self, other: Slot):
        return self.time > other.time

    def __ge__(self, other: Slot):
        return self.time >= other.time
