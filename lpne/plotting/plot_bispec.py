"""
Plot the bispectrum.

"""
__date__ = "February 2023"


import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

import lpne


def plot_bispec(
    bispec, freq=None, dense=False, sigma=0.0, mode="square", fn="temp.pdf"
):
    """
    Plot the bispectrum.

    Parameters
    ----------
    bispec : numpy.ndarray
        Bispectrum
        Shape: [f,f']
    freq : None or numpy.ndarray
        Shape: [f]
    dense : bool, optional
        Whether the bispectrum is in a dense or sparse format
    sigma : float, optional
        Bandwidth of smoothing in units of frequency bins.
    mode : {"square", "triangle"}, optional
        Whether to plot the symmetric square region of the bispectrum ("square") or the
        full symmetric triangular region of the bispecturm ("triangle").
    fn : str, optional
        Image filename
    """
    assert bispec.ndim == 2, f"len({bispec.shape}) != 2"
    assert mode in ["square", "triangle"]
    if dense:
        bispec = lpne.unsqueeze_bispec_array(bispec)

    # Add the redundant cells region.
    n1, n2 = bispec.shape
    new_bispec = np.zeros((n1, n1), dtype=bispec.dtype)
    new_bispec[:, :n2] = bispec
    diag_terms = np.diag(np.diag(bispec))
    new_bispec[:n2, :] += bispec.T
    new_bispec[:n2, :n2] -= diag_terms  # don't double count the diagonal
    bispec = new_bispec

    # Smooth
    if sigma >= 0.0:
        bispec = gaussian_filter(bispec, sigma)

    if freq is None:
        extent = None
        if mode == "triangle":
            ymin, ymax = 0, len(bispec)
        else:
            ymin, ymax = 0, len(bispec) / 2
    else:
        min, max = freq[0], freq[-1]
        extent = [min, max, min, max]
        if mode == "triangle":
            ymin, ymax = freq[0], freq[-1]
        else:
            ymin, ymax = freq[0], freq[len(freq) // 2]
    plt.imshow(bispec.T, origin="lower", extent=extent)
    plt.colorbar()
    plt.ylim(ymin, ymax)
    plt.xlim(ymin, ymax)
    plt.savefig(fn)
    plt.close("all")


if __name__ == "__main__":
    pass


###
