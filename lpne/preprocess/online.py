"""
Online spectral decompositions

Adapted from: https://github.com/HazyResearch/state-spaces

TODO:
* Hann windows
* Bispectrum
* Movies
"""
__date__ = "March 2023"

import numpy as np
import scipy.fft as sp_fft
from scipy.signal import cont2discrete


class OnlineDecomp:
    def __init__(
        self,
        fs,
        spec_timescale=1.0,
        bicoh_timescale=1.0,
        window="hann",
        N=64,
        max_freq=55.0,
    ):
        """

        Parameters
        ----------

        """
        assert N % 2 == 0, f"{N} % 2 != 0"
        self.fs = fs
        self.dt = 1.0 / fs
        self.spec_time_factor = np.exp(-self.dt / spec_timescale)
        self.bicoh_time_factor = np.exp(-self.dt / bicoh_timescale)
        self.window = window
        self.N = N

        freq = sp_fft.rfftfreq(self.N, 1 / self.dt)
        fi = np.searchsorted(freq, max_freq)
        self.freq = freq[:fi]

        self.spec = None
        self.bispec = None

        A, B = get_fourier_ssm(self.N)
        system = (A, B, np.ones((1, N)), np.zeros((1,)))
        self.dA, self.dB, _, _, _ = cont2discrete(system, dt=self.dt, method="zoh")
        self.dB = self.dB.flatten()

        self.state = np.zeros(self.N)
        self.iter = 0

    def update(self, sample):
        """

        Parameters
        ----------
        sample : float
        """
        self.state = self.dA @ self.state + self.dB * sample
        self.iter += 1
        if self.iter >= self.N:
            temp = self.state.reshape(-1, 2)
            temp = np.sum(temp**2, axis=1)
            if self.iter == self.N:
                self.spec = temp
            elif self.iter > self.N:
                p = self.spec_time_factor
                self.spec = p * self.spec + (1.0 - p) * temp

    def predict(self, ts):
        mat = get_eval_matrix(self.N, ts + self.dt)
        return (mat @ self.state)[::-1]


def get_fourier_ssm(N):
    freqs = np.arange(N // 2)
    d = np.stack([np.zeros(N // 2), freqs], axis=-1).reshape(-1)[1:]
    A = 2 * np.pi * (-np.diag(d, 1) + np.diag(d, -1))
    B = np.zeros(N)
    B[0::2] = 2
    B[0] = 2**0.5
    A = A - B[:, None] * B[None, :]
    B *= 2**0.5
    B = B[:, None]
    return A, B


def get_eval_matrix(N, vals):
    cos = 2**0.5 * np.cos(
        2 * np.pi * np.arange(N // 2)[:, None] * (vals)
    )  # (N/2, T/dt)
    sin = 2**0.5 * np.sin(
        2 * np.pi * np.arange(N // 2)[:, None] * (vals)
    )  # (N/2, T/dt)
    cos[0] /= 2**0.5
    eval_matrix = np.stack([cos.T, sin.T], axis=-1).reshape(-1, N)  # (T/dt, N)
    return eval_matrix


if __name__ == "__main__":
    fs = 100
    ts = 1 / fs * np.arange(200)
    signal = np.sin(5 * ts) + np.cos(20.3 * ts)

    decomp = OnlineDecomp(fs, N=50, spec_timescale=0.3)

    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    from moviepy.video.io.bindings import mplfig_to_npimage
    import matplotlib.pyplot as plt

    fig, axarr = plt.subplots(ncols=2)
    axarr[0].plot(ts, signal)

    axarr[1].set_ylim(0, 0.5)

    frames = []
    for i, x in enumerate(signal):
        decomp.update(x)
        if i % 4 == 0:
            handles = []
            pred = decomp.predict(ts[:i])
            temp_h = axarr[0].plot(ts[:i], pred, c="tab:orange")
            handles += temp_h

            if decomp.spec is not None:
                temp_h = axarr[1].plot(decomp.spec, c="tab:blue")
                handles += temp_h
            frames.append(mplfig_to_npimage(fig))
            for handle in handles:
                handle.remove()

    animation = ImageSequenceClip(frames, fps=10)
    animation.write_gif("temp.gif", fps=10)


###
