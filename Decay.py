# Project: IT_3105_Module_4
# Created: 03.11.17 15:33
import math


class Decay:
    @staticmethod
    def linear_decay(t: float) -> float:
        return 1 / (t+1)

    @staticmethod
    def exp_decay(t: float, time_const: float) -> float:
        return math.exp(-(t / time_const))

    @staticmethod
    def power_series(t: float, epochs: int) -> float:
        return 0.005 ** (t / epochs)

    @staticmethod
    def slow_linear_decay(t: float) -> float:
        return 0.999**t
