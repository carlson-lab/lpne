"""
Test the bispectral functions

"""
__date__ = "March 2023"

import numpy as np

import lpne


def test_bispectrum_bicoherence():
    """
    f1, f2, and f1+f2 have related phases
    f3 has unrelated phases
    """
    i1, i2, i3 = 30, 7, 15  # i1 >= i2
    n = 10
    t = 100
    fs = 1
    ts = np.arange(t) / fs
    freqs = np.linspace(0, 0.5, 51)
    f1, f2, f3 = 2 * np.pi * freqs[np.array([i1, i2, i3])]

    ds = []
    for _ in range(n):
        phi1, phi2, phi3 = 2 * np.pi * np.random.rand(3)
        d = np.cos(ts * f1 - phi1)
        d += np.cos(ts * f2 - phi2)
        d += np.cos(ts * (f1 + f2) - phi1 - phi2)
        d += np.cos(ts * f3 - phi3)
        ds.append(d)
    ds = np.array(ds)
    ds = ds.reshape(1, n, t)

    # Test the bispectrum.
    bispec, _ = lpne.get_bispectrum(ds, fs=fs, max_freq=1.0)
    bispec = bispec[0]
    a1, a2 = np.argmax(bispec.sum(axis=1)), np.argmax(bispec.sum(axis=0))
    assert (a1, a2) == (i1, i2), f"{(a1,a2)} != {(i1,i2)}"

    # Test the bicoherence.
    bicoh, _ = lpne.get_bicoherence(ds, fs=fs, max_freq=1.0)
    bicoh = bicoh[0]
    a1, a2 = np.argmax(bicoh.sum(axis=1)), np.argmax(bicoh.sum(axis=0))
    assert (a1, a2) == (i1, i2), f"{(a1,a2)} != {(i1,i2)}"


if __name__ == "__main__":
    pass


###
