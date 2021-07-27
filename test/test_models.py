"""
Test lpne.models.

"""
__date__ = "July 2021"


from itertools import product
import numpy as np
import pytest

import lpne



def test_factor_analysis_sae():
    """Make some fake data, train briefly, and reload the model."""
    # Set up fake data.
    n = 100 # number of datapoints/windows
    n_features = 100 # total number of LFP features
    n_classes = 3 # number of label types
    features = np.random.randn(n, n_features)
    labels = np.random.randint(n_classes, size=n)
    class_counts = [len(np.argwhere(labels==i)) for i in range(n_classes)]
    class_weights = n / (n_classes * np.array(class_counts))
    # Run through different combinations of the model.
    nonnegative_vals = [True, False]
    variational_vals = [True, False]
    for nonnegative, variational in product(nonnegative_vals, variational_vals):
        # Make the model.
        model = lpne.FaSae(
                n_features,
                n_classes,
                class_weights=class_weights,
                weight_reg=0.0,
                nonnegative=True,
                variational=True,
                kl_factor=0.1,
        )
        # Fit the model.
        model.fit(features, labels, epochs=30, verbose=False)
        # Make some predictions.
        predictions = model.predict(features)
        # Calculate a weighted accuracy.
        weighted_acc_orig = model.score(
                features,
                labels,
                class_weights,
        )
        # Get state.
        params = model.get_params()
        # Make a new model and load the state.
        new_model = lpne.FaSae(n_features, n_classes)
        new_model.set_params(params)
        # Calculate a weighted accuracy.
        weighted_acc = new_model.score(
                features,
                labels,
                class_weights,
        )
        assert weighted_acc_orig == weighted_acc



if __name__ == '__main__':
    pass



###
