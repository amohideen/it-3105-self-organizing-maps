# Project: IT_3105_Module_4
# Created: 29.10.17 21:08
import numpy as np
import matplotlib.pyplot as plt
from DataReader import DataReader
from Utilities import TSMUtils
tensor = np.array


def plot_tsm_points(locations: tensor, show_labels: bool=True):
    plt.figure()

    labels = locations[:, 0]
    xs = locations[:, 1]
    ys = locations[:, 2]
    plt.plot(xs, ys, "ro")
    if show_labels:
        for i in range(len(locations)):
            plt.annotate(str(int(labels[i])), xy=(xs[i], ys[i]))

    plt.show()



cities = DataReader.read_tsm_file(10)
normalized = TSMUtils.normalize_coordinates(cities)
plot_tsm_points(normalized)

