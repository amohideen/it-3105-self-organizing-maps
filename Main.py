# Project: IT_3105_Module_4
# Created: 29.10.17 21:45
import numpy as np
tensor = np.array


def create_som_layer(n_rows: int, n_cols: int, n_features: int) -> tensor:
    return np.random.uniform(0, 1, (n_rows, n_cols, n_features))


print(create_som_layer(10, 5, 3))