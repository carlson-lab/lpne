"""
Test lpne.models.

"""
__date__ = "July - November 2021"


from itertools import product
import numpy as np
import pytest

from lpne.models import FaSae, CpSae
from lpne.models.dCSFA_NMF import dCSFA_NMF



def test_factor_analysis_sae():
    """Make some fake data, train briefly, and reload the model."""
    b, f, r, g = 10, 9, 8, 2
    features = np.random.randn(b, f, r, r)
    labels = np.random.randint(0, 3, size=(b,))
    groups = np.random.randint(0, g, size=(b,))

    # Run through different combinations of the model.
    nmf_vals = [True, False]
    variational_vals = [True, False]
    for nonnegative, variational in product(nmf_vals, variational_vals):
        # Make the model.
        model = FaSae(
                nonnegative=nonnegative,
                variational=variational,
                n_iter=1,
        )
        # Fit the model.
        model.fit(features, labels, groups, print_freq=None)
        # Make some predictions.
        predictions = model.predict(features)
        # Calculate a weighted accuracy.
        weighted_acc_orig = model.score(
                features,
                labels,
                groups,
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
                groups,
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


def test_dCSFA_NMF():

        n = 100
        n_freqs = 56
        n_brain_regions = 8
        n_tasks = 1

        #Generate data in a similar style to the old pipeline
        n_power_features = n_freqs*n_brain_regions
        n_dir_features = n_freqs*(n_brain_regions*(n_brain_regions-1))
        n_features = n_power_features + n_dir_features
        features = np.random.uniform(low=0,high=1,size=(n,n_features))
        labels = np.random.binomial(1,0.5,size=(n,n_tasks))

        model = dCSFA_NMF(n_components=20,
                        dim_in = features.shape[1])

        model.fit(features,
                labels,
                n_epochs=10,
                n_pre_epochs=10,
                nmf_max_iter=1,
                batch_size=128,
                pretrain=True,
                verbose=False)

        #torch.save(model,"demo.pt")

        y_pred,s = model.predict_proba(features,include_scores=True)
        params = model.state_dict()

        new_model = dCSFA_NMF(n_components=20,
                        dim_in = features.shape[1])

        new_model.load_state_dict(params)

        y_pred_new,s_new = model.predict_proba(features,include_scores=True)

        assert np.unique(y_pred_new==y_pred) == True
        assert np.unique(s==s_new) == True


if __name__ == '__main__':
    pass



###
