# Project: IT_3105_Module_4
# Created: 29.10.17 21:45
import numpy as np
from typing import Any
import math
from Utilities import Utilities
from Decay import Decay
from DataReader import DataReader
from Visualization import plot_cities_and_neurons
from pprint import pprint
np.set_printoptions(suppress=True)

tensor = np.array


def update_plot(weights, line, fig):
    xs = weights[:, 0]
    ys = weights[:, 1]
    line.set_xdata(np.append(xs, xs[0]))
    line.set_ydata(np.append(ys, ys[0]))
    fig.canvas.draw()


def create_solution(cases: tensor, originals: tensor, neurons: tensor, line: Any, fig: Any):
    solution_map = {}
    for i in range(len(cases)):
        winner = Utilities.get_winning_neuron(cases[i], neurons)
        if winner in solution_map.keys():
            solution_map[winner].append(i)
        else:
            solution_map[winner] = [i]
    solution = []
    for i in range(len(neurons)):
        if i in solution_map.keys():
            solution.extend(solution_map[i])
    norm_solution = tensor(list(map(lambda index: cases[index], solution)))
    xs = norm_solution[:, 0]
    ys = norm_solution[:, 1]
    line.set_xdata(np.append(xs, xs[0]))
    line.set_ydata(np.append(ys, ys[0]))
    fig.canvas.draw()

    # calculate length of the created solution
    solution = tensor(list(map(lambda index: tensor(originals[index]), solution)))
    coordinates = solution[:, 1:]
    total = 0
    for i in range(len(solution) - 1):
        total += Utilities.euclidian_distance(solution[i], solution[i+1])
    total += Utilities.euclidian_distance(solution[-1], solution[0])
    print(total)



def tsm_test():
    epochs = 500
    cities = DataReader.read_tsm_file(9)
    originals = cities
    cities = Utilities.normalize_coordinates(cities)

    labels = cities[:, 0:1]
    city_cases = cities[:, 1:]

    n_features = 2
    out_size = len(cities) * 5
    init_learning_rate = 0.5

    weights = np.random.uniform(np.min(city_cases), np.max(city_cases), size=(out_size, n_features))

    init_radius = out_size / 2

    time_const = epochs / np.log(init_radius)

    line1, line2, fig = plot_cities_and_neurons(city_cases, weights)

    for i in range(epochs):
        for case in city_cases:
            winner = Utilities.get_winning_neuron(case, weights)

            # radius = int(init_radius * exp_decay(i, time_const))
            # radius = int(init_radius * power_series(i, epochs))
            radius = int(init_radius * Decay.linear_decay(i+1))
            # l_rate = init_learning_rate * exp_decay(i, time_const)
            # l_rate = init_learning_rate * power_series(i, epochs)
            l_rate = init_learning_rate * Decay.linear_decay(i+1)

            Utilities.update_weight_matrix(case, l_rate, winner, weights)
            # Update neighbours to the right
            for j in range(winner + 1, winner + int(radius) + 1):
                if radius:
                    influence = math.exp(-(((j - winner)**2) / (2*radius**2)))
                    Utilities.update_weight_matrix(case, influence*l_rate, j % out_size, weights)

            # Update neighbours to the left
            for j in range(winner - 1, (radius - winner) - 1, -1):
                if radius:
                    influence = math.exp(-(((winner - j) ** 2) / (2 * radius ** 2)))
                    Utilities.update_weight_matrix(case, influence * l_rate, j % out_size, weights)

        # update_plot(weights, line1, fig)
        create_solution(city_cases, originals, weights, line2, fig)
        if i % 100 == 0:
            print("Epoch %d/%d" % (i, epochs))
    print("DONE")
    create_solution(city_cases, originals, weights, line2, fig)


tsm_test()
input()
