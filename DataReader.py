# Project: IT_3105_Module_4
# Created: 29.10.17 20:50

import numpy as np
from typing import List, Tuple
from mnist import mnist_basics
tensor = np.array


class DataReader:

    @staticmethod
    def read_tsm_file(number: int) -> tensor:
        def _process_line(line: str) -> List:
            return list(map(float, line.split(" ")))

        with open("data/%d.txt" % number) as file:
            file.readline()
            file.readline()
            n_cities = int(file.readline().split(": ")[-1])
            lines = file.readlines()[2:2+n_cities]
            lines = list(map(lambda s: s.strip(), lines))
            lines = list(map(_process_line, lines))
            return tensor(lines)

    @staticmethod
    def load_mnist(train_limit: int = 50000, test_limit: int = 10000) -> Tuple[tensor, tensor, tensor, tensor]:
        cases = mnist_basics.load_all_flat_cases()
        features = tensor(cases[0]) / 255
        labels = tensor(cases[1])
        return features[:train_limit], labels[:train_limit], \
               features[train_limit:train_limit + test_limit], labels[train_limit:train_limit + test_limit]




