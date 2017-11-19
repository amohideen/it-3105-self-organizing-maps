# Project: IT_3105_Module_4
# Created: 18.11.17 13:59
import click
import cProfile
from DataReader import DataReader
from Utilities import Utilities
from SOM import SOM
NoOp = None


def run_mnist(n_train: int=4000,
              n_test: int=100,
              visualize: bool=True,
              n_epochs: int=5,
              l_rate: float=0.7,
              profile: bool=False):
    Utilities.delete_previous_output("mnist_images")
    mnist_features, mnist_labels, mnist_test_features, mnist_test_labels = DataReader.load_mnist(train_limit=n_train,
                                                                                                 test_limit=n_test)
    som = SOM(mnist=True,
              features=mnist_features,
              labels=mnist_labels,
              test_features=mnist_test_features,
              test_labels=mnist_test_labels,
              n_epochs=n_epochs,
              initial_radius=5,
              initial_l_rate=l_rate,
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


@click.command()
@click.option("--visualize", default=True, type=click.BOOL, help="Create visualizations when running")
@click.option("--epochs", default=5, help="Number of epochs to run")
@click.option("--lrate", default=0.7, help="Learning rate")
@click.option("--trainsize", default=4000, help="Number of training images")
@click.option("--testsize", default=100, help="Number of test images")
def mnist(visualize, epochs, lrate, trainsize, testsize):
    click.echo("Running MNIST")
    run_mnist(n_train=trainsize, n_test=testsize, visualize=visualize, n_epochs=epochs, l_rate=lrate)


if __name__ == "__main__":
    mnist()