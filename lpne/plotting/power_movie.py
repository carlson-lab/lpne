"""
Plot power in a movie.

"""
__date__ = "August 2021"


import numpy as np
import matplotlib.pyplot as plt
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.video.io.bindings import mplfig_to_npimage

import lpne

COLOR = 'tab:blue'
FONTSIZE = 12



def make_power_movie(lfps, duration, window_duration, fs=1000, speed_factor=5,
    fps=10, fn='out.mp4'):
    """
    Make a movie of the power decomposition.

    Parameters
    ----------
    lfps : dict
        Maps channel names to LFP waveforms (Numpy arrays)
    duration : float
        Duration of LFPs.
    window_duration : float
        Duration used to calculate power features.
    fs : int, optional
        LFP samplerate, in Hz.
    speed_factor : float, optional
    fps : int, optional
        Frames per second
    fn : str, optional
        Movie filename.
    """
    # Get the features.
    movie_duration = duration / speed_factor
    window_step = speed_factor / fps
    res = lpne.make_features(
            lfps,
            fs=fs,
            window_duration=window_duration,
            window_step=window_step,
    )
    # Set up the plot.
    power = res['power']
    n_freqs = power.shape[2]
    rois = list(lfps.keys())
    fig, axarr = _set_up_plot(power[0], rois)
    # Iterate over frames.
    frames = []
    title = fig.suptitle("", fontsize=FONTSIZE, y=0.93)
    for k in range(power.shape[0]):
        to_remove = []
        t = k*window_step
        time_str = str(int(np.floor(t / 3600))).zfill(2)
        time_str += ":" + str(int(np.floor(t / 60))).zfill(2)
        time_str += ":" + str(int(np.floor(t % 60))).zfill(2)
        title.set_text(time_str)
        for i in range(len(rois)):
            for j in range(i+1):
                idx = (i * (i+1)) // 2 + j
                to_remove_part = axarr[i,j].fill_between(
                        np.arange(n_freqs),
                        power[k,idx],
                        fc=COLOR,
                )
                to_remove.append(to_remove_part)
                if j < i:
                    to_remove_part = axarr[j,i].fill_between(
                            np.arange(n_freqs),
                            power[k,idx],
                            fc=COLOR,
                    )
                    to_remove.append(to_remove_part)
        frames.append(mplfig_to_npimage(fig))
        for obj in to_remove:
            obj.remove()
    animation = ImageSequenceClip(frames, fps=fps)
    if fn.endswith('.gif'):
        animation.write_gif(fn, fps=fps)
    else:
        animation.write_videofile(fn, fps=fps)
    plt.close('all')


def _set_up_plot(power, rois):
    """Make the base plot."""
    pretty_rois = [roi.replace('_', ' ') for roi in rois]
    ylim = (-0.05*np.max(power), 1.05*np.max(power))
    n = int(round((-1 + np.sqrt(1+8*power.shape[0]))/2))
    assert n == len(rois)
    fig, axarr = plt.subplots(n,n,figsize=(8,8))
    for i in range(n):
        for j in range(i+1):
            for dir in axarr[i,j].spines:
                axarr[i,j].spines[dir].set_visible(False)
            plt.sca(axarr[i,j])
            plt.xticks([])
            plt.yticks([])
            plt.ylim(ylim)
            if j < i:
                for dir in axarr[i,j].spines:
                    axarr[j,i].spines[dir].set_visible(False)
                plt.sca(axarr[j,i])
                plt.xticks([])
                plt.yticks([])
                plt.ylim(ylim)
            if j == 0:
                plt.sca(axarr[i,j])
                plt.ylabel(pretty_rois[i], size='xx-small', rotation=30)
            if i == n-1:
                plt.sca(axarr[i,j])
                plt.xlabel(pretty_rois[j], size='xx-small', rotation=30)
    return fig, axarr



if __name__ == '__main__':
    pass


###
