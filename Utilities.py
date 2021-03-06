# Project: IT_3105_Module_4
# Created: 31.10.17 13:12
import numpy as np
from typing import Tuple, List, Callable
import os
from termcolor import colored
import os
from Decay import Decay
from functools import partial
from copy import deepcopy
tensor = np.array


class Utilities:
    @staticmethod
    def normalize_coordinates(cities: tensor) -> tensor:
        cities = np.copy(cities)
        means = np.mean(cities, axis=0)
        stds = np.std(cities, axis=0)
        for row in range(len(cities)):
            for col in range(1, len(cities[row])):
                cities[row][col] = (cities[row][col] - means[col]) / stds[col]
        return cities

    @staticmethod
    def normalize_coordinates_old(cities: tensor) -> tensor:
        cities = np.copy(cities)
        means = np.mean(cities, axis=0)
        stds = np.std(cities, axis=0)
        for row in range(len(cities)):
            for col in range(1, len(cities[row])):
                cities[row][col] = (cities[row][col] - means[col]) / stds[col]
        return cities


    @staticmethod
    def euclidian_distance(v1: tensor, v2: tensor) -> float:
        assert len(v1) == len(v2), "Tensors must be of equal length to compute distance"
        return np.linalg.norm(v1-v2)

    @staticmethod
    def ring_distance(p1: Tuple, p2: Tuple, size: int):
        i = p1[1]
        j = p2[1]
        return min(abs(i - j), size - abs(i - j))


    @staticmethod
    def get_winning_neuron(case: tensor, weight_matrix: tensor) -> int:
        distances = np.apply_along_axis(Utilities.euclidian_distance, 1, weight_matrix, case)
        return int(np.argmin(distances))

    @staticmethod
    def get_winning_neuron_2d(case: tensor, weight_matrix: tensor) -> Tuple:
        n_rows, n_cols, _ = weight_matrix.shape
        distances = np.empty(shape=(n_rows, n_cols))
        for r in range(n_rows):
            for c in range(n_cols):
                distances[r][c] = Utilities.euclidian_distance(case, weight_matrix[r][c])
        return tuple(np.unravel_index(distances.argmin(), distances.shape))

    @staticmethod
    def update_weight_matrix(case: tensor, l_rate: float, win_index: int, weight_matrix: tensor):
        j = win_index
        weight_matrix[j] = weight_matrix[j] + l_rate * (case - weight_matrix[j])

    @staticmethod
    def update_weight_matrix_2d(case: tensor, l_rate: float, r: int, c: int, matrix):
        matrix[r][c] = matrix[r][c] + l_rate * (case - matrix[r][c])

    @staticmethod
    def store_tsm_result(case: int,
                         epochs: int,
                         nodes: int,
                         l_rate: float,
                         radius: int,
                         l_decay: str,
                         r_decay: str,
                         result: float):
        line = "%d\t\t%d\t\t%d\t\t%.2f\t\t%d\t\t%s\t\t%s\t\t%.2f\n" % \
               (case, epochs, nodes, l_rate, radius, l_decay, r_decay, result)
        with open("tsm_results.txt", "a") as f:
            f.write(line)

    @staticmethod
    def create_decay_function(name: str, epochs: int, time_const: float) -> Callable:
        if name == "lin":
            return Decay.linear_decay
        elif name == "exp":
            return partial(Decay.exp_decay, time_const=time_const)
        elif name == "pow":
            return partial(Decay.power_series, epochs=epochs)
        elif name == "cur":
            return Decay.cur
        else:
            assert False, "Invalid radius decay function"

    @staticmethod
    def delete_previous_output(folder: str):
        for img in os.listdir(folder):
            img_path = os.path.join(folder, img)
            try:
                if os.path.isfile(img_path):
                    os.unlink(img_path)
            except Exception as e:
                print(e)

    @staticmethod
    def reduce_memory(memory: List):
        memory = deepcopy(memory)
        for r in range(len(memory)):
            for c in range(len(memory[r])):
                n = len(memory[r][c])
                if n:
                    counts = np.bincount(memory[r][c])
                    memory[r][c] = np.argmax(counts)
                else:
                    memory[r][c] = -1
        return memory

    @staticmethod
    def make_gif(mnist: bool):
        if mnist:
            os.chdir("/home/espen/Documents/AI Prog/IT_3105_Module_4/mnist_images")
            os.system("convert -loop 0 -delay 100 *.png out.gif")
        else:
            os.chdir("/home/espen/Documents/AI Prog/IT_3105_Module_4/tsm_images")
            os.system("convert -loop 0 -delay 100 *.png out.gif")

    @staticmethod
    def print_progress(total: int, i: int, msg: str=""):
        percentage = int((i / total) * 100)
        if percentage < 25:
            color = "on_red"
        elif 25 <= percentage < 75:
            color = "on_yellow"
        else:
            color = "on_green"
        bar = colored(" " * percentage, "grey", color)
        bar = bar + colored(" ", "grey", color, attrs=["blink"])
        spaces = " " * (100 - percentage)
        progress_bar = colored(" ", "grey", "on_white") + bar + spaces + colored(" ", "grey", "on_white")
        print("\r%s %3d%% \t %s" %
              (progress_bar, percentage, msg),
              end="",
              flush=True)

    @staticmethod
    def time_to_visualize(i: int, display_interval: int, n_epochs: int) -> bool:
        return i % display_interval == 0 or i == n_epochs - 1

