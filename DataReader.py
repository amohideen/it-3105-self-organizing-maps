# Project: IT_3105_Module_4
# Created: 29.10.17 20:50

import numpy as np
from typing import List
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




