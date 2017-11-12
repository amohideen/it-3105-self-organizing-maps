# Project: IT_3105_Module_4
# Created: 29.10.17 21:08
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as colormap

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
