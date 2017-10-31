# Project: IT_3105_Module_4
# Created: 31.10.17 13:12
import numpy as np
tensor = np.array


class TSMUtils:

    @staticmethod
    def normalize_coordinates(cities: tensor) -> tensor:
        cities = np.copy(cities)
        means = np.mean(cities, axis=0)
        stds = np.std(cities, axis=0)
        for row in range(len(cities)):
            for col in range(1, len(cities[row])):
                cities[row][col] = (cities[row][col] - means[col]) / stds[col]
        return cities
