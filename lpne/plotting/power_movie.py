"""
Plot power in a movie.

"""
__date__ = "August 2021 - July 2022"


import numpy as np
import matplotlib.pyplot as plt
try:
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    from moviepy.video.io.bindings import mplfig_to_npimage
    MOVIEPY_INSTALLED = True
except ModuleNotFoundError:
    MOVIEPY_INSTALLED = False
from tqdm import tqdm

import lpne

COLOR_1 = 'tab:blue'
COLOR_2 = 'tab:orange'
FONTSIZE = 12



def make_power_movie(lfps, duration, window_duration, fs=1000, feature='power',
    speed_factor=5, fps=10, model=None, fn='out.mp4'):
    """
    Make a movie of LFP power features.

    Parameters
    ----------
    lfps : dict
        Maps channel names (``str``) to LFP waveforms (``np.ndarray``)
    duration : float
        Duration of LFPs
    window_duration : float
        Duration used to calculate power features
    fs : int, optional
        LFP samplerate, in Hz
    feature : {``'power'``, ``'dir_spec'``}, optional
        What features to plot
    speed_factor : float, optional
    fps : int, optional
        Frames per second
    model : None or lpne.BaseModel
        If a model is passed, reconstructed powers will also be plotted.
    fn : str, optional
        Movie filename
    """
    assert MOVIEPY_INSTALLED, "moviepy needs to be installed!"
    assert feature in ['power', 'dir_spec'], f"{feature} not valid!"
    # Get the features.
    window_step = speed_factor / fps
    max_n_windows = int((duration - window_duration) / window_step)
    res = lpne.make_features(
            lfps,
            fs=fs,
            window_duration=window_duration,
            window_step=window_step,
            max_n_windows=max_n_windows,
            directed_spectrum=(feature == 'dir_spec'),
    )
    # Set up the plot.
    if feature == 'power':
        power = lpne.unsqueeze_triangular_array(res['power'], dim=1) #[w,r,r,f]
    else:
        power = res['dir_spec'] # [w,r,r,f]
    # Get reconstructed volumes.
    flag = (model is not None)
    if flag:
        rec_power = model.reconstruct(np.transpose(power, [0,3,1,2]))
        rec_power = np.transpose(rec_power, [0,2,3,1]) # [w,r,r,f]
        alpha = 0.6
    else:
        alpha = 1.0
    freqs = np.arange(power.shape[3])
    rois = list(lfps.keys())
    fig, axarr = _set_up_plot(power, rois)
    # Iterate over frames.
    frames = []
    title = fig.suptitle("", fontsize=FONTSIZE, y=0.93)
    for k in tqdm(range(power.shape[0])):
        t = k * window_step
        time_str = str(int(np.floor(t / 3600))).zfill(2)
        time_str += ":" + str(int(np.floor(t / 60))).zfill(2)
        time_str += ":" + str(int(np.floor(t % 60))).zfill(2)
        title.set_text(time_str)
        to_remove = []
        for i in range(len(rois)):
            for j in range(len(rois)):
                handle = axarr[i,j].fill_between(
                        freqs,
                        power[k,i,j],
                        fc=COLOR_1,
                        alpha=alpha,
                )
                to_remove.append(handle)
                if flag:
                    handle = axarr[i,j].fill_between(
                            freqs,
                            rec_power[k,i,j],
                            fc=COLOR_2,
                            alpha=alpha,
                    )
                    to_remove.append(handle)
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
    """
    Make the base plot.

    Parameters
    ----------
    power : numpy.ndarray
        Shape: ``[w,r,r,f]``
    rois : list of str
        Shape: ``[f]``
    """
    pretty_rois = [roi.replace('_', ' ') for roi in rois]
    ylim = (-0.05*np.max(power), 1.05*np.max(power))
    n = power.shape[1]
    assert n == len(rois), f"{n} != len({rois})"
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
