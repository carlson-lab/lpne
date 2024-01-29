"""
Code for directed spectral measures: spectral Granger and directed spectrum

Some of this code is adapted from: https://github.com/neil-gallagher/directed-spectrum/
Commit: 1c0d69f
Paper: https://openreview.net/forum?id=AhlzUugOFIo

Copyright (c) 2021, Neil Gallagher
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met: 1. Redistributions of source
code must retain the above copyright notice, this list of conditions and the following
disclaimer. 2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or other
materials provided with the distribution. 3. Neither the name of the copyright holder
nor the names of its contributors may be used to endorse or promote products derived
from this software without specific prior written permission. THIS SOFTWARE IS PROVIDED
BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from itertools import combinations
from warnings import warn
import numpy as np
from scipy.signal import csd
from scipy.fft import fft, ifft
from numpy.linalg import cholesky, solve


DEFAULT_CSD_PARAMS = {
    "detrend": "constant",
    "window": "hann",
    "nperseg": 512,
    "noverlap": 256,
    "nfft": None,
}
"""Default parameters sent to ``scipy.signal.csd``"""


def get_directed_spectral_measures(
    X,
    fs,
    return_spectral_granger=True,
    return_directed_spectrum=True,
    pairwise=True,
    max_iter=1000,
    tol=1e-6,
    csd_params={},
    conditional_covar_epsilon=1e-10,
):
    """
    Calculate spectral Granger and directed spectrum from the signal ``X``.

    Parameters
    ----------
    X : numpy.ndarray
        Timeseries data from multiple channels
        Shape: ``[n_roi, time]`` or ``[n_window, n_roi, time]``
    fs : float
        Sampling frequency
    return_spectral_granger : bool, optional
        Whether to return the spectral Granger measure. Default: ``True``
    return_directed_spectrum : bool, optional
        Whether to return the directed spectrum measure. Default: ``True``
    pairwise : bool, optional
        Whether to calculate the pairwise directed spectrum. Default: ``True``
    max_iter : int, optional
        Maximum number of Wilson factorization iterations. Default: ``1000``
    tol : float, optional
        Tolerance for Wilson factorization. Default: ``1e-6``
    csd_params : dict, optional
        Parameters sent to ``scipy.signal.csd``. Default: ``DEFAULT_CSD_PARAMS``
    conditional_covar_epsilon : float, optional
        Used to prevent division by zero when calculating conditional innovation
        covariances. Default: ``1e-10``

    Returns
    -------
    f : numpy.ndarray
        Array of sample frequencies
        Shape: ``[n_freq]``
    spectral_granger : numpy.ndarray
        Spectral Granger measure. Returned if ``return_spectral_granger``.
        Shape: ``[n_window, n_freq, n_roi, n_roi]``
    directed_spectrum : numpy.ndarray
        Directed spectrum measure. Returned if ``return_directed_spectrum``.
        Shape: ``[n_window, n_freq, n_roi, n_roi]``
    """
    # Check the input parameters.
    assert return_spectral_granger or return_directed_spectrum
    if X.ndim == 2:
        X = X.reshape(1, X.shape[0], X.shape[1])  # [r,t] -> [1,r,t]
    assert X.ndim == 3, f"len({X.shape}) != 3"
    r = X.shape[1]
    csd_params = {**DEFAULT_CSD_PARAMS, **csd_params}
    if "return_onesided" in csd_params:
        assert not csd_params["return_onesided"], "Spectrum must be two-sided!"
    else:
        csd_params["return_onesided"] = False
    if "scaling" in csd_params:
        temp = csd_params["scaling"]
        assert temp == "spectrum", f"Found invalid scaling: {temp}"
    else:
        csd_params["scaling"] = "spectrum"

    # Get a cross power spectral density matrix.
    f, cpsd = csd(
        X[:, np.newaxis],
        X[:, :, np.newaxis],
        fs=fs,
        **csd_params,
    )  # [f], [n,r,r,f]
    cpsd = np.moveaxis(cpsd, 3, 1)  # [n,r,r,f] -> [n,f,r,r]

    # Calculate the directed spectrum measure.
    if pairwise:
        ds = np.zeros((X.shape[0], csd_params["nperseg"], r, r), dtype=np.float64)
        for i1, i2 in combinations(range(r), 2):
            idx = np.array([i1, i2], dtype=int)
            sub_cpsd = cpsd[:, :, idx][:, :, :, idx]
            # Factorize the CPSD.
            H, Sigma = _wilson_factorize(sub_cpsd, max_iter, tol)  # [n,f,2,2], [n,2,2]
            # Get the directed spectrum.
            sub_ds = _H_Sigma_to_ds(H, Sigma, conditional_covar_epsilon)  # [n,f,2,2]
            ds[:, :, i1, i2] = sub_ds[:, :, 0, 1]
            ds[:, :, i2, i1] = sub_ds[:, :, 1, 0]
    else:
        H, Sigma = _wilson_factorize(cpsd, max_iter, tol)  # [n,f,r,r], [n,r,r]
        ds = _H_Sigma_to_ds(H, Sigma, conditional_covar_epsilon)

    # Now calculate spectral Granger.
    if return_spectral_granger:
        S_bb = np.diagonal(np.abs(cpsd), axis1=-1, axis2=-2)[:, :, None]  # [n,f,1,r]
        sg = np.log(S_bb / (S_bb - ds))  # [n,f,r,r]

    # Zero-out the diagonals.
    ds[:, :, np.arange(r), np.arange(r)] = 0.0
    if return_spectral_granger:
        sg[:, :, np.arange(r), np.arange(r)] = 0.0

    # Convert to one-sided spectrum.
    nyquist = int(np.floor(len(f) / 2))
    ds = ds[:, : (nyquist + 1)]
    if return_spectral_granger:
        sg = sg[:, : (nyquist + 1)]
    f = np.abs(f[: (nyquist + 1)])

    # Return.
    to_return = [f]
    if return_spectral_granger:
        to_return.append(sg)
    if return_directed_spectrum:
        to_return.append(ds)
    return tuple(to_return)


def _H_Sigma_to_ds(H, Sigma, eps=1e-10):
    """
    Calculate directed spectrum from the transfer matrix and innovation covariance.

    Parameters
    ----------
    H : numpy.ndarray
        Transfer matrix. Shape: [n,f,r,r]
    Sigma : numpy.ndarray
        Innovation covariance. Shape: [n,r,r]

    Returns
    -------
    ds : numpy.ndarray
        Directed spectrum measure. Shape: [n,f,r,r]
    """
    Sigma_diag = np.diagonal(Sigma, axis1=-1, axis2=-2)
    Sigma_aa = Sigma_diag[:, :, None]  # [n,r,1]
    Sigma_bb = Sigma_diag[:, None]  # [n,1,r]
    Sigma_cond = Sigma_aa - Sigma**2 / (Sigma_bb + eps)  # [n,r,r]
    H2 = H.real**2 + H.imag**2  # [n,f,r,r]
    return Sigma_cond[:, None] * H2.swapaxes(-1, -2)  # [n,f,r,r]


def _wilson_factorize(cpsd, max_iter=1000, tol=1e-6, eps_multiplier=100.0):
    """Factorize CPSD into transfer matrix (H) and covariance (Sigma).

    Implements the algorithm outlined in the following reference:
    G. Tunnicliffe. Wilson, “The Factorization of Matricial Spectral Densities,” SIAM J.
    Appl. Math., vol. 23, no. 4, pp. 420426, Dec. 1972, doi: 10.1137/0123044.

    This code is based on an original implementation in MATLAB provided by M. Dhamala
    (mdhamala@mail.phy-ast.gsu.edu).

    Parameters
    ----------
    cpsd : numpy.ndarray
        Cross power spectral density matrix.
        Shape: (n_windows, n_frequencies, n_signals, n_signals)
    max_iter : int, optional
        Maximum number of iterations. Default: ``1000``
    tol : float, optional
        Convergence tolerance value. Default: ``1e-6``
    eps_multiplier : float, optional
        Constant multiplier used in stabilizing the Cholesky decomposition for positive
        semidefinite CPSD matrices. Default: ``100.0``

    Returns
    -------
    H : numpy.ndarray
        shape (n_windows, n_frequencies, n_signals, n_signals)
        Wilson factorization solutions for transfer matrix.
    Sigma : numpy.ndarray
        shape(n_windows, n_signals, n_signals)
        Wilson factorization solutions for innovation covariance matrix.
    """
    cpsd_cond = np.linalg.cond(cpsd)
    if np.any(cpsd_cond > (1 / np.finfo(cpsd.dtype).eps)):
        warn("CPSD matrix is singular!")
        # Regularize the CPSD matrix.
        this_eps = np.spacing(np.abs(cpsd)).max()
        cpsd = cpsd + np.eye(cpsd.shape[-1]) * this_eps * eps_multiplier

    psi, A0 = _initialize_psi(cpsd)
    L = cholesky(cpsd)
    H = np.zeros_like(psi)
    Sigma = np.zeros_like(A0)

    for w in range(cpsd.shape[0]):
        for _ in range(max_iter):
            # These lines implement: g = psi \ cpsd / psi* + I
            psi_inv_cpsd = solve(psi[w], L[w])
            g = psi_inv_cpsd @ psi_inv_cpsd.conj().transpose(0, 2, 1)
            g = g + np.identity(cpsd.shape[-1])
            gplus, g0 = _plus_operator(g)

            # S is chosen so that g0 + S is upper triangular; S + S* = 0
            S = -np.tril(g0, -1)
            S = S - S.conj().transpose()
            gplus = gplus + S
            psi_prev = psi[w].copy()
            psi[w] = psi[w] @ gplus

            A0_prev = A0[w].copy()
            A0[w] = A0[w] @ (g0 + S)

            if _check_convergence(psi[w], psi_prev, tol):
                if _check_convergence(A0[w], A0_prev, tol):
                    break
        else:
            warn("Wilson factorization failed to converge.", stacklevel=2)

        # Collect the optimization results.
        H[w] = (solve(A0[w].T, psi[w].transpose(0, 2, 1))).transpose(0, 2, 1)
        Sigma[w] = A0[w] @ A0[w].T
    return H, Sigma


def _initialize_psi(cpsd):
    """Return an initial psi value for Wilson factorization.

    Parameters
    ----------
    cpsd : numpy.ndarray
        Cross power spectral density matrix. Shape: [n,f,r,r]

    Returns
    -------
    psi : numpy.ndarray
        Initial value for psi used in Wilson factorization. Shape: [n,f,r,r]
    h : numpy.ndarray
        Initial value for A0 used in Wilson factorization. Shape: [n,r,r]
    """
    gamma = ifft(cpsd, axis=1)
    gamma0 = gamma[:, 0]
    gamma0 = np.real((gamma0 + gamma0.conj().transpose(0, 2, 1)) / 2.0)
    h = cholesky(gamma0).conj().transpose(0, 2, 1)
    psi = np.tile(h[:, np.newaxis], (1, cpsd.shape[1], 1, 1)).astype(complex)
    return psi, h


def _plus_operator(g):
    """Remove all negative lag components from time-domain representation.

    Parameters
    ----------
    g: numpy.ndarray
        shape (n_frequencies, n_groups, n_groups)
        Frequency-domain representation to which transformation will be applied.

    Returns
    -------
    g_pos : numpy.ndarray
        shape (n_frequencies, n_groups, n_groups)
        Transformed version of g with negative lag components removed.
    gamma[0] : numpy.ndarray
        shape (n_groups, n_groups)
        Zero-lag component of g in time-domain.
    """
    # Remove imaginary components from ifft due to rounding error.
    gamma = ifft(g, axis=0).real
    # Take half of zero lag.
    gamma[0] *= 0.5
    # Take half of nyquist component if the FFT had even number of points.
    F = gamma.shape[0]
    N = np.floor(F / 2).astype(int)
    if F % 2 == 0:
        gamma[N] *= 0.5
    # Zero out the negative frequencies to make things causal.
    gamma[N + 1 :] = 0
    # Reconstitute things in the original domain.
    gp = fft(gamma, axis=0)
    return gp, gamma[0]


def _check_convergence(x, x_prev, tol):
    """Determine whether maximum relative change is lower than tolerance."""
    x_diff = np.abs(x - x_prev)
    ab_x = np.abs(x)
    this_eps = np.finfo(ab_x.dtype).eps
    ab_x[ab_x <= 2 * this_eps] = 1
    rel_diff = x_diff / ab_x
    return rel_diff.max() < tol


if __name__ == "__main__":
    pass


###
