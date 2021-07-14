"""
Useful functions

"""
__date__ = "July 2021"


import numpy as np



def get_fake_labels(n_labels, n_label_types=2):
    """
    A wrapper around `numpy.random.randint`.

    Parameters
    ----------
    n_labels : int
    n_label_types : int, optional

    Returns
    -------
    labels : numpoy.ndarray
        Shape: [n_labels]
    """
    return np.random.randint(0, high=n_label_types, size=n_labels)



if __name__ == '__main__':
    pass



###
