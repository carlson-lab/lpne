"""
Make some example bispectrum plots.

"""
__date__ = "March 2023"

import matplotlib.pyplot as plt
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.video.io.bindings import mplfig_to_npimage
import numpy as np
from tqdm import tqdm

import lpne


if __name__ == "__main__":
    fps = 8
    fs = 1000
    jump_samples = int(fs / fps)
    channel = ["Amy_CeA_L_01", "NAc_Core_L_02", "Cx_PrL_R_01"][2]
    lfps = lpne.load_lfps("test_data/data/example_LFP.mat")

    signal = lfps[channel][:10000]
    signal /= np.median(np.abs(signal))
    ts = 1 / fs * np.arange(len(signal))

    decomp = lpne.OnlineDecomp(
        fs, N=128, spec_timescale=0.5, bicoh_timescale=2.0, max_freq=100.0
    )

    fig, axarr = plt.subplots(ncols=3)
    axarr[0].plot(ts, signal)

    frames = []
    for i, x in tqdm(enumerate(signal), "Online decomposition", len(signal)):
        decomp.observe(x)
        if i % jump_samples == 0:
            handles = []
            pred = decomp.predict(ts[:i])
            temp_h = axarr[0].plot(ts[:i], pred, c="tab:orange")
            handles += temp_h

            if not isinstance(decomp.spec, float):
                temp_h = axarr[1].plot(decomp.freq, decomp.spec, c="tab:blue")
                handles += temp_h

                bicoh = decomp.get_bicoherence()
                temp_h = axarr[2].imshow(bicoh.T, vmin=0.0, vmax=1.0, origin="lower")
                handles.append(temp_h)

            frames.append(mplfig_to_npimage(fig))
            for handle in handles:
                handle.remove()

    animation = ImageSequenceClip(frames, fps=fps)
    animation.write_gif("temp.gif", fps=fps)


###
