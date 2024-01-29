"""
Test the spectral Granger and directed spectrum measures.

"""
__date__ = "January 2024"

import numpy as np

import lpne


def test_directed_measures_1():
    # Synthetic data: example from Dhamala et al. (2008)
    w = 500
    t = 5000
    fs = 200
    C = 0.25  # coupling strength
    X = np.random.randn(w, t, 2)
    for i in range(2, t):
        X[:, i] += 0.55 * X[:, i - 1] - 0.8 * X[:, i - 2]
        X[:, i, 0] += C * X[:, i - 1, 1]  # channel 1 influences channel 0
    X = np.transpose(X, (0, 2, 1))  # [n,t,r] -> [n,r,t]

    f, ds, sg = lpne.get_directed_spectral_measures(
        X,
        fs,
        pairwise=True,
        max_iter=1000,
        tol=1e-6,
        csd_params={},
    )  # [n,f,r,r] [n,f,r,r]
    ds = np.transpose(ds, (0, 2, 3, 1)).mean(axis=0)  # [r,r,f]
    sg = np.transpose(sg, (0, 2, 3, 1)).mean(axis=0)  # [r,r,f]

    for arr, name in zip([ds, sg], ["Directed spectrum", "Spectral Granger"]):
        assert np.min(arr) >= 0.0, f"{name} should be nonnegative!"
        assert (
            np.max(np.diagonal(arr, axis1=0, axis2=1)) == 0.0
        ), f"{name} diagonal should be zero!"
        assert np.sum(arr[1, 0]) > np.sum(
            arr[0, 1]
        ), f"{name} causal direction is backwards!"
        f_max = f[np.argmax(arr[1, 0])]
        assert abs(f_max - 40.0) < 5.0, f"{name} peak is too far from 40 Hz!"


if __name__ == "__main__":
    pass


###
