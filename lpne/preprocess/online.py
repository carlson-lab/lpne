"""
Online spectral decompositions

Adapted from: https://github.com/HazyResearch/state-spaces

TODO:
* Hann windows
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
        eps=1e-8,
    ):
        """

        Parameters
        ----------
        fs : int
        spec_timescale : float, optional
        bicoh_timescale : float, optional
        window : str, optional
        N : int, optional
        max_freq : float, optional
            Maximum frequency, in Hz.
        """
        assert N % 2 == 0, f"{N} % 2 != 0"
        self.fs = fs
        self.dt = 1.0 / fs
        self.spec_time_factor = np.exp(-self.dt / spec_timescale)
        self.bicoh_time_factor = np.exp(-self.dt / bicoh_timescale)
        self.window = window
        self.N = N
        self.max_freq = max_freq
        self.eps = eps

        # Truncate the frequencies.
        freq = sp_fft.rfftfreq(self.N, self.dt)
        fi = np.searchsorted(freq, self.max_freq)
        assert fi > 1, f"max_freq is too low! {max_freq}, {fi}, {np.max(freq)}"
        assert fi <= N // 2, f"max_freq is too high! {max_freq}, {fi}, {N}"
        self.freq = freq[:fi]

        # Make the bicoherence indices.
        self.idx1 = np.arange(fi)
        self.idx2 = self.idx1[: (fi + 1) // 2]
        idx3a = self.idx1[:, None]
        idx3b = self.idx2[None, :]
        self.idx3 = idx3a + idx3b
        self.idx3[self.idx3 >= fi] = 0
        self.idx3[idx3a < idx3b] = 0

        self.spec = 0.0
        self.bicoh_num = 0.0
        self.bicoh_denom1 = 0.0
        self.bicoh_denom2 = 0.0

        A, B = get_fourier_ssm(self.N)
        system = (A, B, np.ones((1, N)), np.zeros((1,)))
        self.dA, self.dB, _, _, _ = cont2discrete(system, dt=self.dt, method="zoh")
        self.dB = self.dB.flatten()

        self.state = np.zeros(self.N)
        self.iter = 0

    def observe(self, sample):
        """
        Update the Fourier decomposition with a new sample.

        Parameters
        ----------
        sample : float
        """
        # Update the state.
        self.state = self.dA @ self.state + self.dB * sample
        self.iter += 1

        # Update the running spectrum and bicoherence.
        # if self.iter >= self.N:
        fi = len(self.freq)
        fft = self.state.reshape(-1, 2)[:fi]

        t_spec = np.sum(fft**2, axis=1)
        fft2 = fft[:]
        fft2[0, :] = 0.0  # remove the DC offset
        fft2c = fft2[..., 0] + 1j * fft2[..., 1]
        i1, i2, i3 = self.idx1, self.idx2, self.idx3
        t_bn = fft2c[i1, None] * fft2c[None, i2] * np.conj(fft2c[i3])
        t_bd1 = np.abs(fft2c[i1, None] * fft2c[None, i2]) ** 2
        t_bd2 = fft2c.real**2 + fft2c.imag**2

        p, pm = self.spec_time_factor, 1.0 - self.spec_time_factor
        self.spec = p * self.spec + pm * t_spec
        p, pm = self.bicoh_time_factor, 1.0 - self.bicoh_time_factor
        self.bicoh_num = p * self.bicoh_num * pm * t_bn
        self.bicoh_denom1 = p * self.bicoh_denom1 * pm * t_bd1
        self.bicoh_denom2 = p * self.bicoh_denom2 * pm * t_bd2

    def predict(self, ts):
        mat = get_eval_matrix(self.N, ts + self.dt)
        return (mat @ self.state)[::-1]

    def get_bicoherence(self):
        num = self.bicoh_num.real**2 + self.bicoh_num.imag**2
        denom = self.bicoh_denom1 * self.bicoh_denom2[self.idx3] + self.eps
        return num / denom


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

    decomp = OnlineDecomp(
        fs, N=64, spec_timescale=0.1, bicoh_timescale=0.1, max_freq=50.0
    )

    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    from moviepy.video.io.bindings import mplfig_to_npimage
    import matplotlib.pyplot as plt

    fig, axarr = plt.subplots(ncols=3)
    axarr[0].plot(ts, signal)

    axarr[1].set_ylim(0, 0.5)

    frames = []
    for i, x in enumerate(signal):
        decomp.observe(x)
        if i % 4 == 0:
            handles = []
            pred = decomp.predict(ts[:i])
            temp_h = axarr[0].plot(ts[:i], pred, c="tab:orange")
            handles += temp_h

            if not isinstance(decomp.spec, float):
                temp_h = axarr[1].plot(decomp.spec, c="tab:blue")
                handles += temp_h

                bicoh = decomp.get_bicoherence()
                temp_h = axarr[2].imshow(bicoh, vmin=0.0, vmax=0.01)
                handles.append(temp_h)

            frames.append(mplfig_to_npimage(fig))
            for handle in handles:
                handle.remove()

    animation = ImageSequenceClip(frames, fps=10)
    animation.write_gif("temp.gif", fps=10)


###
