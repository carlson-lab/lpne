"""
Estimate the bispectrum

"""
__date__ = "December 2022 - February 2023"
__all__ = ["bispectral_power_decomposition", "get_bispectrum"]

import numpy as np
import scipy.fft as sp_fft


def get_bispectrum(x, max_freq_bins=None, complex=False, return_power=False):
    """
    Calculate the bispectrum

    Parameters
    ----------
    x : numpy.ndarray
        Shape: [n,t]
    max_freq_bins : None or int, optional
        Maximum number of frequency bins
    complex : bool, optional
        Whether to return the complex bispectrum or its squared modulus
    return_power : bool, optional
        Whether to return the power spectrum

    Returns
    -------
    bispectrum : numpy.ndarray
        If ``complex``, this is the bispectrum. Otherwise it is the squared modulus of
        the bispectrum. Shape: [f,f']
    power : numpy.ndarray
        Returned if ``return_power``. Shape: [f]
    """
    assert x.ndim == 2, f"len({x.shape}) != 2"
    # Remove the DC offset.
    x -= np.mean(x, axis=1, keepdims=True)

    # Do an FFT.
    freq = sp_fft.rfftfreq(x.shape[1])  # [f]
    fft = sp_fft.rfft(x)  # [n,f]
    power = np.mean(fft * np.conj(fft), axis=0).real  # [f]

    if max_freq_bins is None:
        f = fft.shape[1]
    else:
        f = min(max_freq_bins, fft.shape[1])
        power, freq = power[:f], freq[:f]
    if return_power:
        to_return = (power,)
    else:
        to_return = tuple()

    idx1 = np.arange(len(freq))
    idx2 = idx1[: (len(freq) + 1) // 2]
    idx3a = idx1[:, None]
    idx3b = idx2[None, :]
    idx3 = idx3a + idx3b
    idx3[idx3 >= len(freq)] = 0
    idx3[idx3a < idx3b] = 0
    bispec = fft[:, idx1, None] * fft[:, None, idx2] * np.conj(fft[:, idx3])  # [n,f,f']
    if complex:
        return (bispec,) + to_return
    bispec = np.mean(bispec, axis=0)  # [f,f']
    bispec = bispec.real**2 + bispec.imag**2  # [f,f']
    return (bispec,) + to_return


def bispectral_power_decomposition(x, max_freq_bins=None):
    """
    Decompose the the signal into pure power and bispectral power.

    Parameters
    ----------
    x : numpy.ndarray
        Shape: [n,t]
    max_freq_bins : None or int, optional
        Maximum number of frequency bins

    Returns
    -------
    power_decomp : numpy.ndarray
        The sum along the antidiagonals gives the total power spectrum.
        Shape: [f,f']
    """
    # Get the squared bispectrum and the corresponding power spectrum.
    bispec, power = get_bispectrum(
        x, max_freq_bins=max_freq_bins, return_power=True
    )  # [f,f'], [f]
    f = len(power)

    bc2 = np.zeros_like(bispec)  # squared partial bicoherence [f,f]
    sum_bc2 = np.zeros_like(power)  # [f]
    pure_power = np.zeros_like(power)  # [f]
    pure_power[:2] = power[:2]

    # Zero-pad the bispectrum.
    diff = f - bispec.shape[1]
    bispec = np.pad(bispec, ((0, 0), (0, diff)))

    # Collect the squared partial bicoherence.
    for k in range(f):
        for j in range(k // 2 + 1):
            i = k - j
            if bispec[i, j] > 0 and pure_power[i] * pure_power[j] != 0:
                denom = pure_power[i] * pure_power[j] * power[k]
                bc2[i, j] = bispec[i, j] / (denom + 1e-8)
                sum_bc2[k] += bc2[i, j]
        if sum_bc2[k] >= 1.0:
            for j in range(k // 2 + 1):
                i = k - j
                bc2[i, j] /= sum_bc2[k]
            sum_bc2[k] = 1.0
        if k > 1:
            pure_power[k] = power[k] * (1.0 - sum_bc2[k])

    # Convert the partial bicoherence into the power decomposition.
    power_decomp = np.zeros_like(bc2)
    for i in range(bc2.shape[0]):
        for j in range(bc2.shape[1]):
            k = i + j
            if k < len(power):
                power_decomp[i, j] = bc2[i, j] * power[k]

    # Place the pure power along the first zero-frequency axis and return.
    power_decomp[:, 0] = pure_power[: power_decomp.shape[0]]
    return power_decomp


def compress_bispec(arr):
    """
    Map the sparse bispectrum matrix to a dense matrix.

    ``np.inf`` is inserted for odd f' to be able to recover the original dimensions.

    Parameters
    ----------

    Returns
    -------

    """
    assert arr.ndim == 2
    n1, n2 = arr.shape[-2:]
    top = [2, 1][n1 % 2]
    out = np.zeros((n1 + top, (n2 + 1) // 2), dtype=arr.dtype)
    for i in range(out.shape[1]):
        out[: n1 - 2 * i, i] = arr[i : i + n1 - 2 * i, i]
        j = n2 - 1 - i
        if i == j:
            out[n1 - 2 * i :, i] = np.inf
        else:
            out[n1 - 2 * i :, i] = arr[j : j + n1 - 2 * j, j]
    return out


def expand_bispec(arr):
    """
    Map the dense bispectrum data to a sparse bispectrum matrix.

    Parameters
    ----------


    Returns
    -------

    """
    assert arr.ndim == 2
    a1, a2 = arr.shape[-2:]
    assert a1 % 2 == 0, f"a1 should be even! Got: {a1}"
    flag = np.isinf(arr).sum() > 0
    res = a1 % 4
    if (flag and res == 2) or (not flag and res == 0):
        n1 = a1 - 1
    else:
        n1 = a1 - 2
    n2 = (n1 + 1) // 2
    out = np.zeros((n1, n2), dtype=arr.dtype)
    for i in range(arr.shape[1]):
        out[i : i + n1 - 2 * i, i] = arr[: n1 - 2 * i, i]
        j = n2 - 1 - i
        if i != j:
            out[j : j + n1 - 2 * j, j] = arr[n1 - 2 * i :, i]
    return out


if __name__ == "__main__":
    np.random.seed(42)

    N = 201
    t = np.linspace(0, 100, N)
    xs = []
    for i in range(50):
        s1 = np.cos(2 * np.pi * 5 * t + 0.2)
        s2 = 3 * np.cos(2 * np.pi * 7 * t + 0.5)
        noise = 5 * np.random.normal(0, 1, N)
        signal = s1 + s2 + 0.5 * s1 * s2 + noise
        # xs.append(signal)
        xs.append(noise)
    x = np.array(xs)

    decomp = bispectral_power_decomposition(x, max_freq_bins=23)
    print("decomp", decomp.shape)
    for i in range(decomp.shape[1]):
        decomp[:, i][decomp[:, i] > 0.0] = i

    import matplotlib.pyplot as plt

    plt.subplot(311)
    plt.imshow(decomp)
    # plt.colorbar()

    plt.subplot(312)
    temp = compress_bispec(decomp)
    print("temp", temp.shape)

    plt.imshow(temp)

    temp = expand_bispec(temp)

    print(np.linalg.norm(temp - decomp))
    plt.subplot(313)
    plt.imshow(temp)
    plt.savefig("temp.pdf")


###
