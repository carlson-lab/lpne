"""
Plot a spectrogram.

"""
__date__ = "May 2022"


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft


EPSILON = 1e-8


DEFAULT_STFT_PARAMS = {
    'detrend': 'constant',
    'window': 'hann',
    'nperseg': 1024,
    'noverlap': 896,
    'nfft': None,
}
"""Default parameters sent to ``scipy.signal.stft``"""



def plot_spec(lfp, fs, max_freq=50.0, stft_params={}, roi=None,
    min_max_quantiles=[0.005,0.995], fn='temp.pdf'):
    """
    Plot a spectrogram of the given LFP.
    
    Parameters
    ----------
    lfp : numpy.ndarray
        The local field potential.
    fs : int
        Samplerate
    max_freq : float, optional
        Plot only the frequency content below this frequency.
    stft_params : dict, optional
        Parameters sent to ``scipy.signal.stft``
    roi : None or str, optional
        Name of the channel
    min_max_quantiles : list of float, optional
        Used for color normalization
    fn : str, optional
        Image filename
    """
    assert lfp.ndim == 1, f"len({lfp.shape}) != 1"
    # Make the spectrogram.
    params = {**DEFAULT_STFT_PARAMS, **stft_params}
    f, t, Zxx = stft(lfp, fs=fs, **params)
    idx = np.searchsorted(f, max_freq)
    f = f[:idx]
    Zxx = Zxx[:idx]
    Zxx = np.log(np.abs(Zxx)+EPSILON)
    vmin, vmax = np.quantile(Zxx, min_max_quantiles)
    # Plot the spectrogram.
    plt.imshow(
            Zxx, 
            extent=[t[0], t[-1], f[0], f[-1]],
            origin='lower',
            aspect='auto',
            vmin=vmin,
            vmax=vmax,
    )
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.colorbar()
    if roi is not None:
        pretty_roi = roi.replace('_', ' ')
        plt.title(pretty_roi)
    plt.savefig(fn)
    plt.close('all')



if __name__ == '__main__':
    pass


###