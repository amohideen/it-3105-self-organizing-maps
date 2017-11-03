# Project: IT_3105_Module_4
# Created: 03.11.17 15:33
import math


class Decay:
    @staticmethod
    def linear_decay(t: int) -> float:
        return 1 / t

    @staticmethod
    def exp_decay(t: int, time_const: float) -> float:
        return math.exp(-(t / time_const))

    @staticmethod
    def power_series(t: int, epochs: int) -> float:
        return 0.005 ** (t / epochs)
