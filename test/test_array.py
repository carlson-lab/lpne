"""
Test lpne.array_utils functions.

"""
__date__ = "February 2022"

import numpy as np
import pytest

import lpne


def test_bispec_array_utils():
    data = np.random.rand(100, 100)
    for f in range(17, 23):
        arr = lpne.get_bispectrum(data, max_freq_bins=f)
        temp = lpne.squeeze_bispec_array(arr)
        out = lpne.unsqueeze_bispec_array(temp)
        assert arr.shape == out.shape, f"{arr.shape} != {out.shape}"
        max_dev = np.max(np.abs(arr - out))
        assert np.allclose(arr, out), f"{arr.shape} {temp.shape} {max_dev}"


if __name__ == "__main__":
    pass


###
