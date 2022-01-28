"""
Code for calculating directed spectrum features.

This code is adapted from: https://github.com/neil-gallagher/directed-spectrum/
Commit: 09e08fe6c9d3b372b17a94c5db372e31663736e8
Paper: https://openreview.net/forum?id=AhlzUugOFIo

Copyright (c) 2021, Neil Gallagher
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
__date__ = "January 2022"

from warnings import warn
import numpy as np
from scipy.signal import csd
from scipy.fft import fft, ifft


TOL = 1e-6
"""Wilson factorization convergence tolerance parameter"""

DEFAULT_CSD_PARAMS = {
    'detrend': 'linear',
    'window': 'hann',
    'nperseg': 512,
    'noverlap': 256,
    'nfft': None,
}
"""Default parameters sent to ``scipy.signal.csd``"""



def get_directed_spectrum(X, fs, max_iter=1000, csd_params={}):
    """
    Calculate the directed spectrum from the signal ``X``.

    Parameters
    ----------
    X : numpy.ndarray
        Timeseries data from multiple channels
        Shape: ``[n_roi, time]`` or ``[n_window, n_roi, time]``
    fs : float
        Sampling frequency
    max_iter : int
        Maximum number of Wilson factorization iterations
    csd_params : dict, optional
        Parameters sent to ``scipy.signal.csd``

    Returns
    -------
    f : numpy.ndarray
        Array of sample frequencies
        Shape: ``[n_freq]``
    ds : numpy.ndarray
        Directed spectrum
        Shape: ``[n_window, n_freq, n_roi, n_roi]``
    """
    if X.ndim == 2:
        X = X.reshape(1, X.shape[0], X.shape[1]) # [r,t] -> [1,r,t]
    assert X.ndim == 3, f"len({X.shape}) != 3"
    # Get a cross power spectral density matrix.
    csd_params = {**DEFAULT_CSD_PARAMS, **csd_params}
    f, cpsd = csd(
            X[:,np.newaxis],
            X[:,:,np.newaxis],
            fs=fs,
            return_onesided=False,
            **csd_params,
    ) # [f], [n,r,r,f]
    cpsd = np.moveaxis(cpsd, 3, 1) # [n,r,r,f] -> [n,f,r,r]
    # Factorize the CPSD into h: [n,f,r,r] and sigma: [n,r,r]
    h, sigma = _wilson_factorize(cpsd, fs, max_iter)
    h = np.power(np.abs(h),2) # convert to squared magnitude
    # Remove the redundant negative frequencies.
    new_f = (len(f) // 2) + 1
    f = f[:new_f]
    n, r = sigma.shape[:2]
    ds = np.zeros((n,new_f,r,r), dtype=np.float64) # [n,f,r,r]
    # Iterate through pairs of ROIs calculate DS from the CPSD factorization.
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            if i == j:
                continue
            temp = (np.abs(sigma[:,i,j]) / np.abs(sigma[:,j,j]))**2
            temp = sigma[:,i,i].real - temp * sigma[:,j,j].real
            temp = h[:,:new_f,j,i] * temp[:,np.newaxis]
            ds[:,:,i,j] = temp
    return f, ds


def _wilson_factorize(cpsd, fs, max_iter, eps_multiplier=100):
    """
    Factorize CPSD into transfer matrix H and covariance Sigma.

    Implements the algorithm outlined in:

    .. G. Tunnicliffe. Wilson, “The Factorization of Matricial Spectral
       Densities,” SIAM J. Appl. Math., vol. 23, no. 4, pp. 420426, Dec. 1972,
       doi:10.1137/0123044.

    This code is based on an original implementation in MATLAB provided by M.
    Dhamala (mdhamala@mail.phy-ast.gsu.edu).

    Parameters
    ----------
    cpsd : numpy.ndarray
        Cross power spectral density matrix
        Shape: ``[n,f,r,r]``
    fs : float
        Sampling rate
    max_iter : int
        Maximum number of Wilson factorization iterations
    eps_multiplier : int
        Constant multiplier used in stabilizing the Cholesky decomposition

    Returns
    -------
    h : numpy.ndarray
        Wilson factorization solutions for transfer matrix
        Shape: ``[n,f,r,r]``
    sigma : numpy.ndarray
        Wilson factorization solutions for innovation covariance matrix
        Shape: ``[n,r,r]``
    """
    psi, a0 = _init_psi(cpsd)
    # Make sure the CPSD is well-conditioned.
    epsilon = np.spacing(np.abs(cpsd)).max() * eps_multiplier
    chol = np.linalg.cholesky(cpsd + epsilon*np.eye(cpsd.shape[-1]))
    h = np.zeros_like(psi)
    sigma = np.zeros_like(a0)
    failed_windows = 0
    for w in range(cpsd.shape[0]):
        flag = False # indicates whether convergence occurs
        for i in range(max_iter):
            # These lines implement: g = psi \ cpsd / psi* + I
            psi_inv_cpsd = np.linalg.solve (psi[w], chol[w])
            g = psi_inv_cpsd @ psi_inv_cpsd.conj().transpose(0, 2, 1)
            g = g + np.identity(cpsd.shape[-1])
            gplus, g0 = _plus_operator(g)
            # S is chosen so that g0 + S is upper triangular; S + S* = 0
            s = -np.tril(g0, -1)
            s = s - s.conj().transpose()
            gplus = gplus + s
            psi_prev = psi[w].copy()
            psi[w] = psi[w] @ gplus
            a0_prev = a0[w].copy()
            a0[w] = a0[w] @ (g0 + s)
            flag = _converged(psi[w], psi_prev) and _converged(a0[w], a0_prev)
            if flag:
                break
        if not flag:
            failed_windows += 1
        # Do a right-side solve.
        sol = np.linalg.solve(a0[w].T, psi[w].transpose(0, 2, 1))
        h[w] = sol.transpose(0, 2, 1)
        sigma[w] = (a0[w] @ a0[w].T)
    if failed_windows > 0:
        warn(f"{failed_windows} of {cpsd.shape[0]} windows failed to converge!")
    return h, sigma


def _init_psi(cpsd):
    """
    Get an initial psi value for wilson factorization.

    Parameters
    ----------
    cpsd : numpy.ndarray
        Cross power spectral density matrix
        Shape: ``[n,f,r,r]``

    Returns
    -------
    psi : numpy.ndarray
        Initial value for psi used in Wilson factorization.
        Shape: ``[n,f,r,r]``
    h : numpy.ndarray
        Initial value for A0 used in Wilson factorization.
        Shape: ``[n,r,r]``
    """
    gamma = fft(cpsd, axis=1)
    gamma0 = gamma[:, 0]
    # Remove the assymetry in gamma0 caused by rounding errors.
    gamma0 = np.real((gamma0 + gamma0.conj().transpose(0, 2, 1)) / 2.0)
    h = np.linalg.cholesky(gamma0).conj().transpose(0, 2, 1)
    psi = np.tile(h[:, np.newaxis], (1, gamma.shape[1], 1, 1)).astype(complex)
    return psi, h


def _plus_operator(g):
    """
    Remove all negative lag components from time-domain representation.

    Parameters
    ----------
    g: numpy.ndarray
        Frequency-domain representation to which transformation will be applied
        Shape: ``[f,r,r]``

    Returns
    -------
    g_pos : numpy.ndarray
        Transformed version of g with negative lag components removed
        Shape: ``[f,r,r]``
    gamma_0 : numpy.ndarray
        Zero-lag component of ``g`` in time-domain
        Shape: ``[r,r]``
    """
    # Remove the imaginary components from ifft due to rounding errors.
    gamma = ifft(g, axis=0).real
    # Take half of zero lag.
    gamma[0] *= 0.5
    # Take half of nyquist component if fft had even # of points.
    f = gamma.shape[0]
    n = f // 2
    if f % 2 == 0:
        gamma[n] *= 0.5
    # Zero out negative frequencies.
    gamma[n+1:] = 0
    gp = fft(gamma, axis=0)
    return gp, gamma[0]


def _converged(x, x0, tol=TOL):
    """
    Determine whether maximum relative change is lower than tolerance.

    Parameters
    ----------
    x : numpy.dnarray
        Current matrix/array
    x0 : numpy.ndarray
        Previous matrix/array
    tol : float, optional
        Tolerance value for convergence check

    Returns
    -------
    converged : bool
        Indicates whether convergence has occured
    """
    x_diff = np.abs(x - x0)
    ab_x = np.abs(x)
    this_eps = np.finfo(ab_x.dtype).eps
    ab_x[ab_x <= 2*this_eps] = 1
    rel_diff = x_diff / ab_x
    converged = rel_diff.max() < tol
    return converged



if __name__ == '__main__':
    pass



###
