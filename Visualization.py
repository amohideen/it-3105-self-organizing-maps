# Project: IT_3105_Module_4
# Created: 29.10.17 21:08
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
from typing import List, Tuple


tensor = np.array


class TSMVisualizer:

    def _weights_to_coordinates(self, weights: tensor) -> Tuple[List, List]:
        xs = []
        ys = []
        for i in range(len(weights[0])):
            xs.append(weights[0][i][0])
            ys.append(weights[0][i][1])
        return xs, ys

    def __init__(self, cities: tensor, weights: tensor):
        self.fig = plt.figure()

        plt.ion()

        # Plot cities
        x_cities = cities[:, 0]
        y_cities = cities[:, 1]
        plt.plot(x_cities, y_cities, "ro")

        # Plot weights
        x_weights, y_weights = self._weights_to_coordinates(weights)
        self.weight_line = plt.plot(x_weights, y_weights, ":b")[0]

        # Solution Line
        self.solution_line = plt.plot([], [], "k", linewidth=2)[0]

        plt.show()

    def update_weights(self, weights: tensor):
        xs = weights[:, :, 0][0]
        ys = weights[:, :, 1][0]
        # xs, ys = self._weights_to_coordinates(weights)
        self.weight_line.set_xdata(np.append(xs, xs[0]))
        self.weight_line.set_ydata(np.append(ys, ys[0]))
        self.fig.canvas.draw()

    def update_solution(self, solution: tensor, distance: float, epoch: int):
        xs = solution[:, 0]
        ys = solution[:, 1]
        self.solution_line.set_xdata(np.append(xs, xs[0]))
        self.solution_line.set_ydata(np.append(ys, ys[0]))
        plt.title("Epoch %3d, Distance: %.2f" % (epoch, distance))
        plt.savefig("tsm_images/%03d.png" % epoch, dpi=150)
        self.fig.canvas.draw()



def plot_cities_and_neurons(cities: tensor, neurons: tensor):
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    plt.ion()
    x_cities = cities[:, 0]
    y_cities = cities[:, 1]

    x_neurons = neurons[:, 0]
    y_neurons = neurons[:, 1]

    line1, = plt.plot(x_neurons, y_neurons, ":b")
    plt.plot(x_cities, y_cities, "ro")
    line2, = plt.plot([], [], "k", linewidth=2)

    plt.show()
    return line1, line2, fig




def plot_mnist_color(memory: tensor, epoch: int):
    fig = plt.figure()
    fig.add_subplot()
    fig.canvas.set_window_title("Epoch %d" % epoch)
    plt.title("Epoch %d" % epoch)

    cmap = colormap.get_cmap("jet")
    cmap.set_under("w")

    plt.pcolormesh(memory, cmap=cmap, vmin=0)
    plt.colorbar(cmap=cmap)
    plt.savefig("mnist_images/%02d.png" % epoch, dpi=150)
    plt.close(fig)

