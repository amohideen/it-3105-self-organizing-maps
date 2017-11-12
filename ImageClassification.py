# Project: IT_3105_Module_4
# Created: 09.11.17 13:04

import numpy as np
from mnist import mnist_basics
from typing import Tuple, List, Union
from Utilities import Utilities
from Visualization import plot_mnist_color
import math
from functools import partial
from Decay import Decay
import os
np.random.seed(123)
from pprint import pprint

tensor = np.array


def load_mnist(self, train_limit: int = 50000, test_limit: int = 10000) -> Tuple[tensor, tensor, tensor, tensor]:
    cases = mnist_basics.load_all_flat_cases()
    features = tensor(cases[0]) / 255
    labels = tensor(cases[1])
    return features[:train_limit], labels[:train_limit], \
           features[train_limit:train_limit + test_limit], labels[train_limit:train_limit + test_limit]


class SOM:

    def __init__(self,
                 features: tensor,
                 labels: Union[tensor, None],
                 test_features: Union[tensor, None]=None,
                 test_labels: Union[tensor, None]=None,
                 display_interval: int=10):
        self.features = features
        self.labels = labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.display_interval = display_interval


    def generate_neighbour_coordinates(self,
                                       row: int,
                                       col: int,
                                       n_rows: int,
                                       n_cols: int,
                                       radius: float,
                                       wrap: bool = False) -> List[Tuple]:
        def _in_range(r: int, c: int, n_rows: int, n_cols: int) -> bool:
            return 0 <= r < n_rows and 0 <= c < n_cols
        neighbours = []
        for dr in range(1, int(math.ceil(radius)) + 1):
            for dc in range(1, int(math.ceil(radius)) + 1):
                neighbours.append((row + dr, col + dc))
                neighbours.append((row + dr, col - dc))
                neighbours.append((row - dr, col + dc))
                neighbours.append((row - dr, col - dc))
                neighbours.append((row, col - dc))
                neighbours.append((row, col + dc))
                neighbours.append((row - dr, col))
                neighbours.append((row + dr, col))
        if wrap:
            return list(set(map(lambda t: (t[0] % n_rows, t[1] % n_cols), neighbours)))
        else:
            return list(set(filter(lambda t: _in_range(t[0], t[1], n_rows, n_cols), neighbours)))


    def average_memory(self, memory: List):
        for r in range(len(memory)):
            for c in range(len(memory[r])):
                n = len(memory[r][c])
                memory[r][c] = sum(memory[r][c]) / n if n else -1


    def test(self, features: tensor, labels: tensor, memory: List, weights: tensor):
        print("\nStarting Testing\n")
        predictions = []
        for i, case in enumerate(features):
            row, col = Utilities.get_winning_neuron_2d(case, weights)
            prediction = memory[row][col]
            correct = labels[i] == int(prediction)
            predictions.append(correct)
            if not correct:
                print("%d ------> %f" % (labels[i], prediction))
        accuracy = sum(predictions) / len(predictions)
        print("Accuracy: %f%%" % (accuracy * 100))


    def run(self, features: tensor, labels: tensor, test_features: tensor, test_labels: tensor,  k: int=10):
        feature_len = max(map(len, features))       # 784 for mnist
        output_size = 10
        init_radius = 5
        init_l_rate = 0.7
        decay = "power"
        epochs = 20
        time_const = epochs / np.log(init_radius)

        weights = np.random.uniform(size=(output_size, output_size, feature_len))
        memory = None

        if decay == "linear":
            decay_f = Decay.linear_decay
        elif decay == "exp":
            decay_f = partial(Decay.exp_decay, time_const=time_const)
        elif decay == "power":
            decay_f = partial(Decay.power_series, epochs=epochs)
        else:
            assert False, "Invalid decay function"

        for i in range(epochs + 1):

            memory = [[[] for _ in range(output_size)] for _ in range(output_size)]

            radius = int(init_radius * decay_f(i))
            l_rate = init_l_rate * decay_f(i)

            print("Epoch %d,\t radius: %d, l_rate: %f" % (i, radius, l_rate))

            for j, case in enumerate(features):
                row, col = Utilities.get_winning_neuron_2d(case, weights)
                Utilities.update_weight_matrix_2d(case, l_rate, row, col, weights)
                for neighbour in generate_neighbour_coordinates(row, col, output_size, output_size, radius):
                    influence = math.exp(
                        -(Utilities.euclidian_distance(tensor(neighbour), tensor((row, col)) ** 2) / (2 * radius ** 2))
                    )
                    Utilities.update_weight_matrix_2d(case, influence * l_rate, neighbour[0], neighbour[1], weights)

                memory[row][col].append(labels[j])
                print("Epoch %d,\t case %d" % (i, j)) if j % 100 == 0 else None

            average_memory(memory)
            plot_mnist_color(memory, i) if i % k == 0 else None

        test(test_features, test_labels, memory, weights)




# Utilities.delete_previous_output("mnist_images")
#
# run(*load_mnist(train_limit=1000, test_limit=100), k=1)
#
# os.chdir("/home/espen/Documents/AI Prog/IT_3105_Module_4/mnist_images")
# os.system("convert -loop 0 -delay 100 *.png out.gif")
