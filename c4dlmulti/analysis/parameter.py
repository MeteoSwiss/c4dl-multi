import numpy as np


def occurrence(batch_gen, dataset="train"):
    """The occurrence of a binary target variable.
    """
    n_occ = 0
    n_total = 0
    num_batches = len(batch_gen.time_coords[dataset]) // \
        batch_gen.batch_size
    for i in range(num_batches):
        if i % 10 == 0:
            print("{}/{}".format(i,num_batches))
        (X,Y) = batch_gen.batch(i)
        n_occ += np.count_nonzero(Y[0])
        n_total += np.prod(Y[0].shape)

    return (n_occ/n_total)
