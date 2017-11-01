# Project: IT_3105_Module_4
# Created: 29.10.17 21:45
import numpy as np
from Utilities import Utils, TSMUtils
from DataReader import DataReader
from Visualization import plot_cities_and_neurons
np.set_printoptions(suppress=True)

tensor = np.array


def example_from_slides():
    cases = [
        tensor([1, 1, 0, 0]),
        tensor([0, 0, 0, 1]),
        tensor([1, 0, 0, 0]),
        tensor([0, 0, 1, 1])
    ]

    n_features = 4
    out_size = 2
    l_rate = 0.6

    weights = np.random.uniform(size=(out_size, n_features))

    for i in range(10):
        for case in cases:
            winner = Utils.get_winning_neuron(case, weights)
            print("\nWinner is neuron %d" % winner)
            Utils.update_weight_matrix(case, l_rate, winner, weights)
            print("New weights are:\n%r" % weights)


def tsm_test():
    cities = DataReader.read_tsm_file(3)
    cities = TSMUtils.normalize_coordinates(cities)

    labels = cities[:, 0:1]
    city_cases = cities[:, 1:]


    n_features = 2
    out_size = len(cities)
    l_rate = 0.6

    weights = np.random.uniform(np.min(city_cases), np.max(city_cases),size=(out_size, n_features))

    line, fig = plot_cities_and_neurons(city_cases, weights)
    for i in range(100):
        for case in city_cases:
            winner = Utils.get_winning_neuron(case, weights)
            Utils.update_weight_matrix(case, l_rate, winner, weights)

        line.set_xdata(weights[:, 0])
        line.set_ydata(weights[:, 1])
        fig.canvas.draw()




tsm_test()
input()