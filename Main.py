# Project: IT_3105_Module_4
# Created: 29.10.17 21:45
import numpy as np
from typing import Any
import math
from Utilities import Utilities
from Decay import Decay
from DataReader import DataReader
from Visualization import plot_cities_and_neurons
from functools import partial
np.random.seed(123)
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
    return total


def tsm_test(k: int=10):
    tsm_case = 6
    epochs = 400
    node_factor = 6
    radius_divisor = 2
    init_learning_rate = 0.9
    decay = "power"

    cities = DataReader.read_tsm_file(tsm_case)
    originals = cities
    cities = Utilities.normalize_coordinates(cities)

    labels = cities[:, 0:1]
    city_cases = cities[:, 1:]

    n_features = 2
    out_size = len(cities) * node_factor

    weights = np.random.uniform(np.min(city_cases), np.max(city_cases), size=(out_size, n_features))

    init_radius = out_size / radius_divisor

    time_const = epochs / np.log(init_radius)

    line1, line2, fig = plot_cities_and_neurons(city_cases, weights)

    if decay == "linear":
        decay_f = Decay.linear_decay
    elif decay == "exp":
        decay_f = partial(Decay.exp_decay, time_const=time_const)
    elif decay == "power":
        decay_f = partial(Decay.power_series, epochs=epochs)
    else:
        assert False, "Invalid decay function"

    for i in range(epochs):
        for case in city_cases:
            winner = Utilities.get_winning_neuron(case, weights)

            radius = int(init_radius * decay_f(i))
            l_rate = init_learning_rate * decay_f(i)

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
        if i % k == 0:
            update_plot(weights, line1, fig)
            create_solution(city_cases, originals, weights, line2, fig)
            print("Epoch %d/%d" % (i, epochs))
    print("DONE")
    Utilities.store_tsm_result(tsm_case,
                               epochs,
                               node_factor,
                               init_learning_rate,
                               radius_divisor,
                               decay,
                               create_solution(city_cases, originals, weights, line2, fig))


tsm_test()
input()
