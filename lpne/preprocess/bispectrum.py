"""
Estimate the bispectrum

"""
__date__ = "December 2022"
__all__ = ["bispectrum"]

import numpy as np
import scipy.fft as sp_fft
from scipy.signal import spectrogram




def fft_power(x):
    f = sp_fft.rfftfreq(len(x))
    return np.abs(sp_fft.rfft(x))**2, f


def cheap_power(x):
    """
    
    """
    f, _, spec = spectrogram(x)
    return (np.abs(spec)**2).mean(axis=-1), f # [b,f], [f]


def fft_bispec(x):
    f = sp_fft.rfftfreq(x.shape[1])
    fft = sp_fft.rfft(x)
    print("f", f.shape, "fft", fft.shape)
    idx = np.arange(len(f)//2 + 1)
    idx2 = idx[:, None] + idx[None, :]
    bispec = fft[:,idx, None] * fft[:,None, idx] * np.conj(fft[:,idx2])
    bispec = np.mean(bispec, axis=0)
    return bispec


def cheap_bispec(x):
    """
    
    """
    f, _, spec = spectrogram(x)
    # df = f[1] - f[0]
    spec = np.swapaxes(spec, -1, -2) # [b,f,t] -> [b,t,f]
    idx = np.arange((len(f) + 1) // 2) # [f']
    idx2 = idx[..., :, None] + idx[..., None, :] # [f',f']
    temp = spec[..., :, idx, None] * spec[..., :, None, idx] # [b,t,f',f']
    temp *= np.conjugate(spec[..., :, idx2]) # [b,t,f',f']
    bispec = np.mean(temp, axis=-3) # [b,f',f']
    return bispec


def cheap_decomp(x):
    """
    
    
    """
    power, f = cheap_power(x) # [b,f]
    assert power.shape[1] >= 2
    df = f[1] - f[0]
    print("df", df)
    print("power", power.shape)
    bispec = cheap_bispec(x) # [b,f',f']
    bispec = np.abs(bispec)**2
    print("bispec", bispec.shape, bispec.dtype)

    bc2 = np.zeros_like(bispec)
    sum_bc2 = np.zeros_like(power)

    pure_power = np.zeros_like(power)
    for i in [0,1]:
        pure_power[...,i] = power[...,i]

    for k in range(len(f)):
        for j in range(k // 2 + 1):
            i = k - j
            idx1 = np.argwhere(bispec[:,i,j] > 0).flatten()
            idx2 = np.argwhere(pure_power[:,i]!= 0).flatten()
            idx3 = np.argwhere(pure_power[:,j] != 0).flatten()
            idx = np.intersect1d(np.intersect1d(idx1, idx2), idx3)
            denom = pure_power[idx,i] * pure_power[idx,j] * power[idx,k] * df
            bc2[idx,i,j] = bispec[idx,i,j] / denom
            sum_bc2[idx,k] += bc2[idx,i,j]
        print("sum", sum_bc2[...,k])
        print("power", power[...,k])
        idx = np.argwhere(sum_bc2[...,k] > 1.0).flatten()
        if len(idx) > 0:
            for j in range(k // 2 + 1):
                i = k - j
                bc2[idx,i,j] /= sum_bc2[idx,k]
            sum_bc2[idx,k] = 1.0
            print("\tsum: ", sum_bc2[...,k])
        if k > 1:
            pure_power[...,k] = power[...,k] * (1.0 - sum_bc2[...,k])
        assert np.min(pure_power[...,k]) >= 0.0, f"{pure_power[...,k]}, {k}"

    print("sdfasdf")
    quit()


def fft_decomp(x):
    """
    
    
    """
    bispec = fft_bispec(x)
    bispec = np.abs(bispec)**2
    print("bispec", bispec.shape)

    # import matplotlib.pyplot as plt
    # plt.imshow(bispec, origin='lower')
    # plt.colorbar()
    # plt.savefig('temp.pdf')

    fft = sp_fft.rfft(x)
    power = np.mean(fft * np.conj(fft), axis=0).real
    # power = np.abs(fft)**2
    print("power", power.shape)

    # Zero-pad the bispectrum.
    new_bispec = np.zeros((len(power), len(power)))
    new_bispec[:len(bispec),:len(bispec)] = bispec
    bispec = new_bispec

    bc2 = np.zeros_like(bispec)
    sum_bc2 = np.zeros_like(power)

    pure_power = np.zeros_like(power)
    pure_power[:2] = power[:2]

    for k in range(len(power)):
        for j in range(k // 2 + 1):
            i = k - j
            if (bispec[i,j] > 0 and pure_power[i]*pure_power[j] != 0):
                denom = pure_power[i] * pure_power[j] * power[k]
                bc2[i,j] = bispec[i,j] / (denom + 1e-8)
                sum_bc2[k] += bc2[i,j]
        # print("sum", sum_bc2[k])
        # print("power", power[k])
        
        if sum_bc2[k] >= 1.0:
            for j in range(k // 2 + 1):
                i = k - j
                bc2[i,j] /= sum_bc2[k]
            sum_bc2[k] = 1.0

            print("\tsum: ", sum_bc2[k])

        # if k > 1:
        pure_power[k] = power[k] * (1.0 - sum_bc2[k])
        assert pure_power[k] >= 0.0, f"{pure_power[k]}, {k}"

    print(np.sum(bc2, axis=0)[:5])
    print(np.sum(bc2, axis=1)[:5])

    import matplotlib.pyplot as plt
    plt.imshow(bc2, origin='lower')
    plt.colorbar()
    plt.savefig('temp.pdf')

    print("sdfasdf")
    quit()



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.random.seed(42)
    x = np.random.randn(100, 100)

    decomp = fft_decomp(x)
    quit()

    decomp = cheap_decomp(x)
    
    print("decomp", decomp.shape)
    plt.imshow(decomp[0], origin='lower')
    plt.colorbar()
    plt.show()
    plt.close('all')


###
