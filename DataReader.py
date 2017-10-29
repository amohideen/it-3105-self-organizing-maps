# Project: IT_3105_Module_4
# Created: 29.10.17 20:50

import numpy as np
from typing import List
from pprint import pprint
np.set_printoptions(suppress=True)


class DataReader:

    @staticmethod
    def _read_tsm_file(path: str) -> np.array:
        def _process_line(line: str) -> List:
            return list(map(float, line.split(" ")))

        with open(path) as file:
            lines = file.readlines()[1:]
            lines = list(map(_process_line, lines))
            return np.array(lines)

    @staticmethod
    def read_sahara() -> np.array:
        return DataReader._read_tsm_file("data/sahara.txt")

    @staticmethod
    def read_djibouti() -> np.array:
        return DataReader._read_tsm_file("data/djibouti.txt")


