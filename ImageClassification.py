# Project: IT_3105_Module_4
# Created: 09.11.17 13:04

import numpy as np
from mnist import mnist_basics
from typing import Tuple, List, Union
from Utilities import Utilities
from Visualization import plot_mnist_color
import math
from termcolor import colored
from functools import partial
from Decay import Decay
import os
np.random.seed(123)
from pprint import pprint

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


    def generate_neighbour_coordinates(self,
                                       row: int,
                                       col: int,
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
            return list(set(map(lambda t: (t[0] % self.n_output_rows, t[1] % self.n_output_cols),
                                neighbours)))
        else:
            return list(set(filter(lambda t: _in_range(t[0], t[1], self.n_output_rows, self.n_output_cols),
                                   neighbours)))


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


    def print_progress(self, total: int, i: int, epoch: int, radius: int, l_rate: float):
        percentage = int((i / total) * 100)
        if percentage < 25:
            color = "red"
        elif 25 <= percentage < 75:
            color = "yellow"
        else:
            color = "green"
        progress_bar = "[%s>%s]" % (("=" * percentage), (" " * (100 - percentage)))
        progress_bar = colored(progress_bar, color)
        print("\r%s %3d%% \t Epoch: %d \t L_Rate: %.3f \t Radius: %d" %
              (progress_bar, percentage, epoch, l_rate, radius),
              end="",
              flush=True)

    def run(self):
        decay = "linear"

        n_cases_to_run = self.n_epochs * len(self.features)
        counter = 0

        weights = np.random.uniform(size=(self.n_output_rows, self.n_output_cols, self.feature_len))
        memory = None

        print("\nStarting Training Session\n")

        for i in range(self.n_epochs):

            memory = [[[] for _ in range(self.n_output_cols)] for _ in range(self.n_output_rows)]

            radius = int(self.initial_radius * self.radius_decay_func(i))
            l_rate = self.initial_l_rate * self.l_rate_decay_func(i)

            for j, case in enumerate(self.features):
                row, col = Utilities.get_winning_neuron_2d(case, weights)
                Utilities.update_weight_matrix_2d(case, l_rate, row, col, weights)
                for neighbour in self.generate_neighbour_coordinates(row, col, radius):
                    influence = math.exp(
                        -(Utilities.euclidian_distance(tensor(neighbour), tensor((row, col)) ** 2) / (2 * radius ** 2))
                    )
                    Utilities.update_weight_matrix_2d(case, influence * l_rate, neighbour[0], neighbour[1], weights)

                memory[row][col].append(self.labels[j])

                counter += 1
                self.print_progress(n_cases_to_run, counter, i, radius, l_rate) if j % 100 else NoOp

            self.average_memory(memory)
            plot_mnist_color(memory, i) if i % self.display_interval == 0 else NoOp

        self.test(memory, weights)




Utilities.delete_previous_output("mnist_images")


mnist_features, mnist_labels, mnist_test_features, mnist_test_labels = load_mnist(train_limit=1000, test_limit=100)

som = SOM(features=mnist_features,
          labels=mnist_labels,
          test_features=mnist_test_features,
          test_labels=mnist_test_labels,
          n_epochs=10,
          initial_radius=5,
          initial_l_rate=0.7,
          radius_decay_func="linear",
          l_rate_decay_func="linear",
          n_output_cols=10,
          n_output_rows=10,
          display_interval=10)
som.run()

os.chdir("/home/espen/Documents/AI Prog/IT_3105_Module_4/mnist_images")
os.system("convert -loop 0 -delay 100 *.png out.gif")
