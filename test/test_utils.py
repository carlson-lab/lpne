"""
Test lpne.utils

"""

import numpy as np

import lpne


def test_unsqueeze_triangular_array():
    # Shape 1
    arr = np.random.randn(3, 10, 8)
    new_arr = lpne.unsqueeze_triangular_array(arr, dim=1)
    assert new_arr.shape == (3, 4, 4, 8)
    residual = new_arr - np.transpose(new_arr, [0, 2, 1, 3])
    assert np.allclose(residual, np.zeros_like(residual))
    # Shape 2
    arr = np.random.randn(10, 8)
    new_arr = lpne.unsqueeze_triangular_array(arr, dim=0)
    assert new_arr.shape == (4, 4, 8)
    residual = new_arr - np.swapaxes(new_arr, 0, 1)
    assert np.allclose(residual, np.zeros_like(residual))
    # Shape 3
    arr = np.random.randn(3, 10)
    new_arr = lpne.unsqueeze_triangular_array(arr, dim=1)
    assert new_arr.shape == (3, 4, 4)
    residual = new_arr - np.swapaxes(new_arr, 1, 2)
    assert np.allclose(residual, np.zeros_like(residual))


def test_squeeze_triangular_array():
    # Shape 1
    arr = np.random.randn(3, 4, 4, 8)
    new_arr = lpne.squeeze_triangular_array(arr, dims=(1, 2))
    assert new_arr.shape == (3, 10, 8)
    # Shape 2
    arr = np.random.randn(4, 4, 8)
    new_arr = lpne.squeeze_triangular_array(arr, dims=(0, 1))
    assert new_arr.shape == (10, 8)
    # Shape 3
    arr = np.random.randn(3, 4, 4)
    new_arr = lpne.squeeze_triangular_array(arr, dims=(1, 2))
    assert new_arr.shape == (3, 10)


def test_squeeze_unsqeeuze_triangular_array():
    arr_1 = np.random.randn(3, 4, 4, 8)
    arr_1 = arr_1 + np.transpose(arr_1, [0, 2, 1, 3])
    arr_2 = lpne.squeeze_triangular_array(arr_1, dims=(1, 2))
    arr_3 = lpne.unsqueeze_triangular_array(arr_2, dim=1)
    arr_4 = lpne.squeeze_triangular_array(arr_3, dims=(1, 2))
    assert np.allclose(arr_1, arr_3)
    assert np.allclose(arr_2, arr_4)


def test_get_weights_1():
    labels = np.array([0, 0, 1, 0, 1], dtype=int)
    groups = np.array([0, 0, 0, 1, 1], dtype=int)
    weights = lpne.get_weights(labels, groups)
    assert np.mean(weights) == 1.0
    assert np.sum(weights[:3]) == np.sum(weights[3:])
    assert np.allclose(weights / weights[0], np.array([1, 1, 2, 2, 2]))


def test_get_weights_2():
    labels = np.array([0, 0, 1, 0, 1, -1, -1], dtype=int)
    groups = np.array([0, 0, 0, 1, 1, 0, 1], dtype=int)
    weights = lpne.get_weights(labels, groups, invalid_label=-1)
    print(weights)
    assert np.mean(weights) == 1.0
    assert np.sum(weights[:3]) == np.sum(weights[3:5])
    assert np.allclose(weights[:5] / weights[0], np.array([1, 1, 2, 2, 2]))
    assert np.allclose(weights[-2:], np.ones(2))


if __name__ == "__main__":
    pass


###
