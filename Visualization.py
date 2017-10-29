# Project: IT_3105_Module_4
# Created: 29.10.17 21:08
import numpy as np
import matplotlib.pyplot as plt
from DataReader import DataReader


def plot_tsm_points(locations: np.array, show_labels: bool=True):
    fig, ax = plt.subplots(1)

    labels = locations[:, 0]
    xs = locations[:, 1]
    ys = locations[:, 2]
    ax.plot(xs, ys, "ro")
    if show_labels:
        for i in range(len(locations)):
            ax.annotate(str(int(labels[i])), xy=(xs[i], ys[i]))

    ax.set_yticklabels([])
    ax.set_xticklabels([])

    plt.show()

