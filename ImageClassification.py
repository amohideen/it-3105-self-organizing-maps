# Project: IT_3105_Module_4
# Created: 09.11.17 13:04

import numpy as np
from mnist import mnist_basics
from typing import Tuple, List, Union
from Utilities import Utilities
from Visualization import plot_mnist_color, TSMVisualizer
from DataReader import DataReader
import math
from functools import partial
from Decay import Decay
import cProfile
np.random.seed(123)
from pprint import pprint
np.set_printoptions(suppress=True)

tensor = np.array
NoOp = None



def load_mnist(train_limit: int = 50000, test_limit: int = 10000) -> Tuple[tensor, tensor, tensor, tensor]:
    cases = mnist_basics.load_all_flat_cases()
    features = tensor(cases[0]) / 255
    labels = tensor(cases[1])
    return features[:train_limit], labels[:train_limit], \
           features[train_limit:train_limit + test_limit], labels[train_limit:train_limit + test_limit]



class SOM:

    def __init__(self,
                 mnist: bool,
                 features: tensor,
                 n_epochs: int,
                 n_output_rows: int,
                 n_output_cols: int,
                 initial_radius: int,
                 initial_l_rate: float,
                 radius_decay_func: str,
                 l_rate_decay_func: str,
                 labels: Union[tensor, None]=None,
                 test_features: Union[tensor, None]=None,
                 test_labels: Union[tensor, None]=None,
                 display_interval: int=10):
        self.mnist = mnist
        self.features = features
        self.labels = labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.n_epochs = n_epochs
        self.n_output_rows = n_output_rows
        self.n_output_cols = n_output_cols
        self.display_interval = display_interval
        self.initial_radius = initial_radius
        self.initial_l_rate = initial_l_rate
        self.feature_len = max(map(len, self.features))

        time_const = self.n_epochs / np.log(self.initial_radius)
        if radius_decay_func == "linear":
            self.radius_decay_func = Decay.linear_decay
        elif radius_decay_func == "exp":
            self.radius_decay_func = partial(Decay.exp_decay, time_const=time_const)
        elif radius_decay_func == "power":
            self.radius_decay_func = partial(Decay.power_series, epochs=self.n_epochs)
        else:
            assert False, "Invalid radius decay function"

        time_const = n_epochs
        if l_rate_decay_func == "linear":
            self.l_rate_decay_func = Decay.linear_decay
        elif l_rate_decay_func == "exp":
            self.l_rate_decay_func = partial(Decay.exp_decay, time_const=time_const)
        elif l_rate_decay_func == "power":
            self.l_rate_decay_func = partial(Decay.power_series, epochs=self.n_epochs)
        else:
            assert False, "Invalid learning rate decay function"

        self.weights = np.random.uniform(np.min(features),
                                         np.max(features),
                                         size=(self.n_output_rows, self.n_output_cols, self.feature_len))

        if not mnist:
            self.tsm_visualizer = TSMVisualizer(self.features, self.weights)

    def generate_neighbour_coordinates(self,
                                       row: int,
                                       col: int,
                                       radius: float) -> List[Tuple]:
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
        return list(set(filter(lambda t: _in_range(t[0], t[1], self.n_output_rows, self.n_output_cols),
                               neighbours)))

    def generate_tsm_neighbours(self, col_position: int, radius: int):
        result = []
        for i in range(col_position + 1, col_position + int(radius) + 1):
            result.append((0, i % self.n_output_cols))
        for i in range(col_position - 1,  (radius - col_position) - 1, -1):
            result.append((0, i % self.n_output_cols))
        return result

    def average_memory(self, memory: List):
        for r in range(len(memory)):
            for c in range(len(memory[r])):
                n = len(memory[r][c])
                memory[r][c] = sum(memory[r][c]) / n if n else -1

    def test(self, memory: List, weights: tensor):
        print("\n\nStarting Testing\n")
        predictions = []
        for i, case in enumerate(self.test_features):
            row, col = Utilities.get_winning_neuron_2d(case, weights)
            prediction = memory[row][col]
            correct = self.test_labels[i] == int(round(prediction))
            predictions.append(correct)
            if not correct:
                print("%d ------> %f" % (self.test_labels[i], prediction))
        accuracy = sum(predictions) / len(predictions)
        print("Accuracy: %f%%" % (accuracy * 100))

    def run(self):

        n_cases_to_run = self.n_epochs * len(self.features)
        counter = 0

        memory = None

        print("\nStarting Training Session\n")

        for i in range(self.n_epochs):
            memory = [[[] for _ in range(self.n_output_cols)] for _ in range(self.n_output_rows)]

            radius = int(round(self.initial_radius * self.radius_decay_func(i)))
            l_rate = self.initial_l_rate * self.l_rate_decay_func(i)

            for j, case in enumerate(self.features):
                row, col = Utilities.get_winning_neuron_2d(case, self.weights)
                Utilities.update_weight_matrix_2d(case, l_rate, row, col, self.weights)
                # TODO make one fast implementation
                if self.mnist:
                    neighbours = self.generate_neighbour_coordinates(row, col, radius)
                else:
                    neighbours = self.generate_tsm_neighbours(col, radius)
                for neighbour in neighbours:
                    if radius:
                        influence = math.exp(
                            -(Utilities.euclidian_distance(tensor(neighbour), tensor((row, col))) /
                              (2 * radius ** 2))
                        )
                        Utilities.update_weight_matrix_2d(case,
                                                          influence * l_rate,
                                                          neighbour[0],
                                                          neighbour[1],
                                                          self.weights)
                if self.mnist:
                    memory[row][col].append(self.labels[j])

                counter += 1
                Utilities.print_progress(n_cases_to_run, counter, i, radius, l_rate) if j % 10 == 0 else NoOp

            if self.mnist:
                self.average_memory(memory)
                plot_mnist_color(memory, i) if i % self.display_interval == 0 else NoOp
            else:
                self.tsm_visualizer.update_weights(self.weights) if i % self.display_interval == 0 else NoOp
                pass

        if self.mnist:
            self.test(memory, self.weights)


def main(mnist: bool, city_number: int=1):
    if mnist:
        Utilities.delete_previous_output("mnist_images")
        mnist_features, mnist_labels, mnist_test_features, mnist_test_labels = load_mnist(train_limit=4000,
                                                                                          test_limit=100)
        som = SOM(mnist=True,
                  features=mnist_features,
                  labels=mnist_labels,
                  test_features=mnist_test_features,
                  test_labels=mnist_test_labels,
                  n_epochs=15,
                  initial_radius=5,
                  initial_l_rate=0.7,
                  radius_decay_func="power",
                  l_rate_decay_func="power",
                  n_output_cols=15,
                  n_output_rows=15,
                  display_interval=1)
        som.run()

        Utilities.make_gif(mnist=True)

    else:
        cities = DataReader.read_tsm_file(city_number)
        means, stds, norm_cities = Utilities.normalize_coordinates(cities)
        features = norm_cities[:, 1:]

        # TSM Hyper Params
        node_factor = 6
        radius_divisor = 2

        out_size = len(features) * node_factor
        init_rad = int(out_size / radius_divisor)

        som = SOM(mnist=False,
                  features=features,
                  n_epochs=400,
                  n_output_rows=1,
                  n_output_cols=out_size,
                  initial_radius=init_rad,
                  initial_l_rate=0.7,
                  radius_decay_func="exp",
                  l_rate_decay_func="exp",
                  display_interval=10)

        som.run()



if __name__ == "__main__":
    # cProfile.run("main(False, 1)")
    main(False, 1)
    input()

'''
Epoch 0/400 lrate: 0.700 rad: 156
Epoch 10/400 lrate: 0.613 rad: 136
Epoch 20/400 lrate: 0.537 rad: 119
Epoch 30/400 lrate: 0.470 rad: 104
Epoch 40/400 lrate: 0.412 rad: 91
Epoch 50/400 lrate: 0.361 rad: 80
Epoch 60/400 lrate: 0.316 rad: 70
Epoch 70/400 lrate: 0.277 rad: 61
Epoch 80/400 lrate: 0.243 rad: 54
Epoch 90/400 lrate: 0.213 rad: 47
Epoch 100/400 lrate: 0.186 rad: 41
'''



