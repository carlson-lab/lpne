"""
Test lpne.models.

"""
__date__ = "July - November 2021"


from itertools import product
import numpy as np
import pytest

from lpne.models import FaSae, CpSae



def test_factor_analysis_sae():
    """Make some fake data, train briefly, and reload the model."""
    # Set up fake data.
    n = 100 # number of datapoints/windows
    n_features = 100 # total number of LFP features
    n_classes = 3 # number of label types
    features = np.random.randn(n, n_features)
    labels = np.random.randint(n_classes, size=n)
    # Run through different combinations of the model.
    nonnegative_vals = [True, False]
    variational_vals = [True, False]
    for nonnegative, variational in product(nonnegative_vals, variational_vals):
        # Make the model.
        model = FaSae(
                nonnegative=nonnegative,
                variational=variational,
                n_iter=1,
        )
        # Fit the model.
        model.fit(features, labels, print_freq=None)
        # Make some predictions.
        predictions = model.predict(features)
        # Calculate a weighted accuracy.
        weighted_acc_orig = model.score(
                features,
                labels,
        )
        # Get state.
        params = model.get_params()
        # Make a new model and load the state.
        new_model = FaSae()
        new_model.set_params(**params)
        # Calculate a weighted accuracy.
        weighted_acc = new_model.score(
                features,
                labels,
        )
        assert weighted_acc_orig == weighted_acc


def test_cp_sae():
    b, f, r, g = 10, 9, 8, 2
    features = np.random.randn(b, f, r, r)
    labels = np.random.randint(0, 3, size=(b,))
    groups = np.random.randint(0, g, size=(b,))
    weights = np.exp(np.random.randn(b))
    model = CpSae(n_iter=1)
    model.fit(features, labels, groups, print_freq=None)
    # Make some predictions.
    predictions = model.predict(features, groups)
    # Calculate a weighted accuracy.
    weighted_acc_orig = model.score(
            features,
            labels,
            groups,
    )
    # Get state.
    params = model.get_params()
    # Make a new model and load the state.
    new_model = CpSae()
    new_model.set_params(**params)
    # Calculate a weighted accuracy.
    weighted_acc = new_model.score(
            features,
            labels,
            groups,
    )
    assert weighted_acc_orig == weighted_acc



if __name__ == '__main__':
    pass



###
