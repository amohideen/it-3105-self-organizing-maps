# Project: IT_3105_Module_4
# Created: 29.10.17 21:45
import numpy as np
from Utilities import Utils
np.set_printoptions(suppress=True)

tensor = np.array


cases = [
    tensor([1, 1, 0, 0]),
    tensor([0, 0, 0, 1]),
    tensor([1, 0, 0, 0]),
    tensor([0, 0, 1, 1])
]

n_features = 4
out_size = 2
l_rate = 0.6

weights = np.random.uniform(size=(out_size, n_features))

for i in range(10):
    for case in cases:
        winner = Utils.get_winning_neuron(case, weights)
        print("\nWinner is neuron %d" % winner)
        Utils.update_weight_matrix(case, l_rate, winner, weights)
        print("New weights are:\n%r" % weights)

