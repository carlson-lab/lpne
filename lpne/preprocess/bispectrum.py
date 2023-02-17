"""
Estimate the bispectrum

"""
__date__ = "December 2022"
__all__ = ["bispectral_power_decomposition"]

import numpy as np
import scipy.fft as sp_fft



def fft_power(x):
    # Remove the DC offset.
    x -= np.mean(x, axis=1, keepdims=True)
    f = sp_fft.rfftfreq(len(x))
    return np.abs(sp_fft.rfft(x))**2, f


def fft_bispec(x, max_f):
    """
    Estimate the bispectrum.
    
    Parameters
    ----------
    x : numpy.ndarray
        Shape: [n,t]
    max_f : int
        HERE

    Returns
    -------
    bispec : numpy.ndarray
        Shape: [f,f]
    """
    # Remove the DC offset.
    x -= np.mean(x, axis=1, keepdims=True)
    f = sp_fft.rfftfreq(x.shape[1])
    fft = sp_fft.rfft(x)

    max_f = min(max_f, len(f))
    f, fft = f[:max_f], fft[:max_f]

    # NOTE: HERE!

    print("f", f.shape, "fft", fft.shape)
    idx = np.arange(len(f)//2 + 1)
    idx2 = idx[:, None] + idx[None, :]
    bispec = fft[:,idx, None] * fft[:,None, idx] * np.conj(fft[:,idx2])
    bispec = np.mean(bispec, axis=0)
    return bispec


def bispectral_power_decomposition(x, f=50):
    """
    Decompose the the signal into pure power and bispectral power.
    
    Parameters
    ----------
    x : numpy.ndarray
        Shape: [n,t]
    f : int, optional
        Maximum number of frequency bins

    Returns
    -------
    power_decomp : numpy.ndarray
        The sum along the antidiagonals gives the total power spectrum.
        Shape: [f,f]
    """
    assert x.ndim == 2, f"len({x.shape}) != 2"
    print("x", x.shape)

    # Remove the DC offset.
    x -= np.mean(x, axis=1, keepdims=True)

    # Do an FFT.
    freq = sp_fft.rfftfreq(x.shape[1]) # [f]
    fft = sp_fft.rfft(x) # [n,f]
    print("fft", fft.shape)
    power = np.mean(fft * np.conj(fft), axis=0).real # [f]
    print("power", power.shape)
    # f = min(f, fft.shape[1])
    f = fft.shape[1]
    power, freq = power[:f], freq[:f]

    print("f", f)

    # Calculate the bispectrum.
    idx = np.arange(len(freq)//2 + 1)
    print("idx", idx.shape)
    idx2 = idx[:, None] + idx[None, :]
    bispec = fft[:,idx, None] * fft[:,None, idx] * np.conj(fft[:,idx2])
    bispec = np.mean(bispec, axis=0) # [f,f]
    bispec = np.abs(bispec)**2 # [f,f]

    print("bispec", bispec.shape)


    bc2 = np.zeros_like(bispec) # squared partial bicoherence [f,f]

    # Zero-pad the bispectrum.
    diff = len(power) - len(bispec)
    bispec = np.pad(bispec, ((0,diff),(0,diff)))
    print("bispec", bispec.shape)


    sum_bc2 = np.zeros_like(power) # [f]
    pure_power = np.zeros_like(power) # [f]
    print("pure_power", pure_power.shape)
    pure_power[:2] = power[:2]

    for k in range(f):
        for j in range(k // 2 + 1):
            i = k - j
            if (bispec[i,j] > 0 and pure_power[i]*pure_power[j] != 0):
                denom = pure_power[i] * pure_power[j] * power[k]
                bc2[i,j] = bispec[i,j] / (denom + 1e-8)
                sum_bc2[k] += bc2[i,j]
        # print("sum", sum_bc2[k])
        
        if sum_bc2[k] >= 1.0:
            for j in range(k // 2 + 1):
                i = k - j
                bc2[i,j] /= sum_bc2[k]
            sum_bc2[k] = 1.0

            print("\tsum: ", sum_bc2[k])

        # if k > 1:
        pure_power[k] = power[k] * (1.0 - sum_bc2[k])
        assert pure_power[k] >= 0.0, f"{pure_power[k]}, {k}"

    # Convert the partial bicoherence into the power decomposition.
    power_decomp = np.zeros_like(bc2)
    for i in range(len(bc2)):
        for j in range(len(bc2)):
            k = i + j
            power_decomp[i,j] = bc2[i,j] * power[k]
    # Place the pure power along the zero-frequency axis.
    power_decomp[:,0] = pure_power[:len(bc2)]

    import matplotlib.pyplot as plt
    plt.imshow(power_decomp, origin='lower')
    plt.colorbar()
    plt.savefig('temp.pdf')

    print("done")
    quit()



if __name__ == "__main__":
    np.random.seed(42)

    N = 200
    t = np.linspace(0, 100, N)
    xs = []
    for i in range(1000):
        s1 = np.cos(2 * np.pi * 5 * t + 0.2)
        s2 = 3 * np.cos(2 * np.pi * 7 * t + 0.5)
        noise = 5 * np.random.normal(0, 1, N)
        signal = s1 + s2 + 0.5 * s1 * s2 + noise
        xs.append(signal)
    x = np.array(xs)

    decomp = bispectral_power_decomposition(x)


###
