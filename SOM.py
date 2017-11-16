# Project: IT_3105_Module_4
# Created: 09.11.17 13:04

import numpy as np
from typing import Tuple, List, Union
from Utilities import Utilities
from Visualization import plot_mnist_color, TSMVisualizer
from DataReader import DataReader
from collections import defaultdict
import math
from functools import partial
from Decay import Decay
import cProfile
np.random.seed(123)
from pprint import pprint
np.set_printoptions(suppress=True)

tensor = np.array
Tensor = np.ndarray
NoOp = None


class SOM:

    def __init__(self,
                 mnist: bool,
                 features: Tensor,
                 n_epochs: int,
                 n_output_rows: int,
                 n_output_cols: int,
                 initial_radius: int,
                 initial_l_rate: float,
                 radius_decay_func: str,
                 l_rate_decay_func: str,
                 labels: Union[Tensor, None]=None,
                 test_features: Union[Tensor, None]=None,
                 test_labels: Union[Tensor, None]=None,
                 originals: Union[Tensor, None]=None,
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
        self.originals = originals
        self.feature_len = max(map(len, self.features))

        self.radius_decay_func = Utilities.create_decay_function(radius_decay_func,
                                                                 self.n_epochs,
                                                                 self.n_epochs / np.log(self.initial_radius))

        self.l_rate_decay_func = Utilities.create_decay_function(l_rate_decay_func,
                                                                 self.n_epochs,
                                                                 self.n_epochs)

        self.weights = np.random.uniform(np.min(features),
                                         np.max(features),
                                         size=(self.n_output_rows, self.n_output_cols, self.feature_len))

        if not mnist:
            self.tsm_visualizer = TSMVisualizer(self.features, self.weights)

    def create_tsm_solution(self, epoch: int):
        solution_map = defaultdict(list)
        for i, feature in enumerate(self.features):
            winner = Utilities.get_winning_neuron_2d(feature, self.weights)
            solution_map[winner].append(i)
        solution_indices = []
        for key in sorted(solution_map.keys(), key=lambda tup: tup[1]):
            solution_indices.extend(solution_map[key])
        normalized_solution = list(map(lambda i: self.features[i], solution_indices))
        solution = list(map(lambda i: self.originals[i], solution_indices))
        total = 0
        for i in range(len(solution) - 1):
            total += Utilities.euclidian_distance(solution[i], solution[i + 1])
        total += Utilities.euclidian_distance(solution[-1], solution[0])
        self.tsm_visualizer.update_solution(tensor(normalized_solution), total, epoch)
        return total

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
        for i in range(1, radius + 1):
            result.append((0, (col_position + i) % self.n_output_cols))
            result.append((0, (col_position - i) % self.n_output_cols))

        return result

    def test(self, memory: List, weights: Tensor, train: bool):
        print("\n\nStarting Testing On ", end="")
        print("Training Set") if train else print("Test Set")
        reduced_memory = Utilities.reduce_memory(memory)
        predictions = []
        features = self.features if train else self.test_features
        labels = self.labels if train else self.test_labels
        for i, case in enumerate(features):
            row, col = Utilities.get_winning_neuron_2d(case, weights)
            correct = labels[i] == reduced_memory[row][col]
            predictions.append(correct)
            if not correct and not train:
                print("%d -------> %d" % (labels[i], reduced_memory[row][col]))
        accuracy = sum(predictions) / len(predictions)
        print("Accuracy: %f%%" % (accuracy * 100))

    def run(self):

        n_cases_to_run = self.n_epochs * len(self.features)
        counter = 0

        memory = None

        print("\nStarting Training Session\n")

        for i in range(self.n_epochs):
            if Utilities.time_to_visualize(i, self.display_interval, self.n_epochs) and self.mnist:
                memory = [[[] for _ in range(self.n_output_cols)] for _ in range(self.n_output_rows)]

            radius = int(round(self.initial_radius * self.radius_decay_func(i)))
            l_rate = self.initial_l_rate * self.l_rate_decay_func(i)

            for j, case in enumerate(self.features):
                row, col = Utilities.get_winning_neuron_2d(case, self.weights)
                Utilities.update_weight_matrix_2d(case, l_rate, row, col, self.weights)
                if self.mnist:
                    neighbours = self.generate_neighbour_coordinates(row, col, radius)
                else:
                    neighbours = self.generate_tsm_neighbours(col, radius)
                for neighbour in neighbours:
                    if radius:
                        if self.mnist:
                            dist = Utilities.euclidian_distance(tensor(neighbour), tensor((row, col)))
                        else:
                            dist = Utilities.ring_distance(neighbour, (row, col), self.n_output_cols)
                        influence = math.exp(-(dist / (2 * radius ** 2)))
                        Utilities.update_weight_matrix_2d(case,
                                                          influence * l_rate,
                                                          neighbour[0],
                                                          neighbour[1],
                                                          self.weights)
                if Utilities.time_to_visualize(i, self.display_interval, self.n_epochs) and self.mnist:
                    memory[row][col].append(self.labels[j])

                counter += 1
                Utilities.print_progress(n_cases_to_run, counter, i, radius, l_rate) if j % 10 == 0 else NoOp

            if Utilities.time_to_visualize(i, self.display_interval, self.n_epochs):
                if self.mnist:
                    reduced_memory = Utilities.reduce_memory(memory)
                    plot_mnist_color(reduced_memory, i)
                else:
                    self.tsm_visualizer.update_weights(self.weights)
                    self.create_tsm_solution(i)

        print("\n\nDone Training\n")

        if self.mnist:
            self.test(memory, self.weights, True)
            self.test(memory, self.weights, False)
        else:
            self.tsm_visualizer.update_weights(self.weights)
            return self.create_tsm_solution(self.n_epochs - 1)


def main(mnist: bool, city_number: int=1):
    if mnist:
        Utilities.delete_previous_output("mnist_images")
        mnist_features, mnist_labels, mnist_test_features, mnist_test_labels = DataReader.load_mnist(train_limit=4000,
                                                                                                     test_limit=100)
        som = SOM(mnist=True,
                  features=mnist_features,
                  labels=mnist_labels,
                  test_features=mnist_test_features,
                  test_labels=mnist_test_labels,
                  n_epochs=5,
                  initial_radius=5,
                  initial_l_rate=0.7,
                  radius_decay_func="power",
                  l_rate_decay_func="power",
                  n_output_cols=20,
                  n_output_rows=20,
                  display_interval=1)
        som.run()

        Utilities.make_gif(mnist=True)

    else:
        Utilities.delete_previous_output("tsm_images")
        cities = DataReader.read_tsm_file(city_number)
        means, stds, norm_cities = Utilities.normalize_coordinates(cities)
        features = norm_cities[:, 1:]

        import random
        while True:
            Utilities.delete_previous_output("tsm_images")

            # TSM Hyper Params
            node_factor = random.randint(2, 10)
            radius_divisor = random.randint(1, 4)
            n_epochs = random.randint(100, 1000)
            l_rate = random.random()

            funcs = ["exp","power"]
            r_decay = funcs[random.randint(0,1)]
            l_decay = funcs[random.randint(0,1)]

            out_size = len(features) * node_factor
            init_rad = int(out_size / radius_divisor)

            som = SOM(mnist=False,
                      features=features,
                      n_epochs=n_epochs,
                      n_output_rows=1,
                      n_output_cols=out_size,
                      initial_radius=init_rad,
                      initial_l_rate=l_rate,
                      radius_decay_func=r_decay,
                      l_rate_decay_func=l_decay,
                      originals=cities[:, 1:],
                      display_interval=10)

            result = som.run()
            Utilities.store_tsm_result(case=city_number,
                                       epochs=n_epochs,
                                       nodes=node_factor,
                                       l_rate=l_rate,
                                       radius=radius_divisor,
                                       l_decay=l_decay,
                                       r_decay=r_decay,
                                       result=result)
        #Utilities.make_gif(mnist=False)


if __name__ == "__main__":
    # cProfile.run("main(False, 1)")
    # cProfile.run("main(True)")
    # main(True)
    main(False, 8)


