# Project: IT_3105_Module_4
# Created: 18.11.17 13:59
import click
import cProfile
from DataReader import DataReader
from Utilities import Utilities
from SOM import SOM
NoOp = None


def run_tsm(city: int, visualize: bool=True, profile: bool = False, n_epochs: int=400, l_rate: float=0.39):
    Utilities.delete_previous_output("tsm_images")
    cities = DataReader.read_tsm_file(city)

    norm_cities = cities
    features = norm_cities[:, 1:]

    # TSM Hyper Params
    node_factor = 3
    radius_divisor = 1
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
    return result

def test_many(n_epochs: int, l_rate: float):
    required = [7541, 6110, 629, 22068, 14379, 108159, 59030, 1211]
    for i in range(0, 8):
        res = run_tsm(i+1, False, n_epochs=n_epochs, l_rate=l_rate)
        assert res < 1.1 * required[i], "City %d failed. Result: %f" % (i+1, res)
    print("All cities solved")


@click.command()
@click.option("--visualize", is_flag=True, help="Create visualizations when running")
@click.option("--epochs", default=400, help="Number of epochs to run")
@click.option("--city", default=1, type=click.INT, help="The city to run")
@click.option("--lrate", default=0.39, help="Learning rate")
@click.option("--testmany", is_flag=True, help="Run many tests")
def tsm(visualize, epochs, city, lrate, testmany):
    click.echo("Running TSM")
    if testmany:
        test_many(epochs, lrate)
    else:
        run_tsm(city, visualize, n_epochs=epochs, l_rate=lrate)


if __name__ == "__main__":
    tsm()