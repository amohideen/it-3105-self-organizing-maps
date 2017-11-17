# Project: IT_3105_Module_4
# Created: 17.11.17 09:29
from DataReader import DataReader
from Utilities import Utilities
from SOM import SOM
import cProfile
import random
import sys
import numpy as np
import threading

NoOp = None


def run_mnist(n_train: int=4000, n_test: int=100, visualize: bool=True, profile: bool=False):
    Utilities.delete_previous_output("mnist_images")
    mnist_features, mnist_labels, mnist_test_features, mnist_test_labels = DataReader.load_mnist(train_limit=n_train,
                                                                                                 test_limit=n_test)
    som = SOM(mnist=True,
              features=mnist_features,
              labels=mnist_labels,
              test_features=mnist_test_features,
              test_labels=mnist_test_labels,
              n_epochs=5,
              initial_radius=5,
              initial_l_rate=0.7,
              radius_decay_func="pow",
              l_rate_decay_func="pow",
              n_output_cols=20,
              n_output_rows=20,
              display_interval=1 if visualize else -1)

    pr = cProfile.Profile()
    if profile:
        pr.enable()

    som.run()

    if profile:
        pr.disable()
        pr.print_stats(sort='time')

    Utilities.make_gif(mnist=True)


def run_tsm(city: int, visualize: bool=True, profile: bool = False):
    Utilities.delete_previous_output("tsm_images")
    cities = DataReader.read_tsm_file(city)

    norm_cities = cities # Utilities.normalize_coordinates(cities)
    features = norm_cities[:, 1:]

    # TSM Hyper Params
    node_factor = 3
    radius_divisor = 1
    n_epochs = 400
    l_rate = 0.39
    l_decay = "cur"
    r_decay = "pow"

    out_size = len(features) * node_factor
    init_rad = int(out_size / radius_divisor)

    som = SOM(mnist=False,
              features=features,
              n_epochs=n_epochs,
              n_output_rows=1,
              n_output_cols=out_size,
              initial_radius=init_rad,
              initial_l_rate=l_rate,
              radius_decay_func=r_decay,
              l_rate_decay_func=l_decay,
              originals=cities[:, 1:],
              display_interval=10 if visualize else -1)

    pr = cProfile.Profile()
    if profile:
        pr.enable()

    result = som.run()

    if profile:
        pr.disable()
        pr.print_stats(sort='time')

    Utilities.store_tsm_result(case=city,
                               epochs=n_epochs,
                               nodes=node_factor,
                               l_rate=l_rate,
                               radius=radius_divisor,
                               l_decay=l_decay,
                               r_decay=r_decay,
                               result=result)
    Utilities.make_gif(mnist=False) if visualize else NoOp


def random_search_tsm(city: int):
    cities = DataReader.read_tsm_file(city)
    norm_cities = cities #Utilities.normalize_coordinates(cities)
    features = norm_cities[:, 1:]

    while True:
        node_factor = random.randint(1, 10)
        radius_divisor = random.uniform(1, 5)
        n_epochs = 400
        l_rate = random.random()
        funcs = ["exp", "exp", "pow", "lin", "cur"]
        r_decay = random.choice(funcs)
        l_decay = random.choice(funcs)

        out_size = len(features) * node_factor
        init_rad = int(out_size / radius_divisor)

        som = SOM(mnist=False,
                  features=features,
                  n_epochs=n_epochs,
                  n_output_rows=1,
                  n_output_cols=out_size,
                  initial_radius=init_rad,
                  initial_l_rate=l_rate,
                  radius_decay_func=r_decay,
                  l_rate_decay_func=l_decay,
                  originals=cities[:, 1:],
                  display_interval=-1)

        Utilities.store_tsm_result(case=city,
                                   epochs=n_epochs,
                                   nodes=node_factor,
                                   l_rate=l_rate,
                                   radius=radius_divisor,
                                   l_decay=l_decay,
                                   r_decay=r_decay,
                                   result=som.run())


def main():
    # noinspection PyBroadException
    try:
        mnist = "mnist" in sys.argv
        vis = "-v" in sys.argv
        profile = "-p" in sys.argv
        if mnist:
            run_mnist(profile=profile, visualize=vis)
        else:
            run_tsm(int(sys.argv[1]), visualize=vis, profile=profile)
    except Exception:
        print("\nError Launching Application. Usage:")
        print("\t'python3 Main.py mnist' to run mnist")
        print("\t'python3 Main.py 1...8' to run the specified tsp city")
        print("Supported flags:")
        print("\t '-v' to enable visualization")
        print("\t '-p' to enable profiling")


if __name__ == "__main__":
    main()

