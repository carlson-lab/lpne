"""
Make a movie of power features.

"""
__date__ = "August 2021 - November 2022"
__all__ = ["make_power_movie"]


import numpy as np
import matplotlib.pyplot as plt

MOVIEPY_INSTALLED = True
try:
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    from moviepy.video.io.bindings import mplfig_to_npimage
except ModuleNotFoundError:
    MOVIEPY_INSTALLED = False
from tqdm import tqdm

import lpne

from .circle_plot import _set_up_circle_plot, _update_circle_plot, R1

COLOR_1 = "tab:blue"
COLOR_2 = "tab:orange"
GRID_FONTSIZE = 12
CIRCLE_FONTSIZE = 18
DEFAULT_CIRCLE_PARAMS = dict(
    freq_ticks=None,
    max_alpha=0.7,
    buffer_percent=1.0,
    outer_radius=1.2,
    min_max_quantiles=(0.5, 0.9),
    min_max_vals=None,
    color=COLOR_1,
)


def make_power_movie(
    lfps,
    duration,
    window_duration,
    fs=1000,
    feature="power",
    speed_factor=5,
    fps=10,
    alpha=0.8,
    model=None,
    mode="grid",
    circle_params={},
    fn="out.mp4",
):
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
    alpha : float, optional
        Feature transparency
    model : None or lpne.BaseModel
        If a model is passed, reconstructed powers will also be plotted.
    mode : {``'grid'``, ``'circle'``}, optional
    circle_params : dict, optional
    fn : str, optional
        Movie filename
    """
    assert MOVIEPY_INSTALLED, "moviepy needs to be installed!"
    assert feature in ["power", "dir_spec"], f"Feature {feature} not valid!"
    assert mode in ["grid", "circle"], f"Mode {mode} not valid!"
    assert (
        feature == "power" or mode == "grid"
    ), "Circle plot is only implemented for power features!"
    # Get the features.
    window_step = speed_factor / fps
    max_n_windows = int((duration - window_duration) / window_step)
    res = lpne.make_features(
        lfps,
        fs=fs,
        window_duration=window_duration,
        window_step=window_step,
        max_n_windows=max_n_windows,
        directed_spectrum=(feature == "dir_spec"),
    )
    freq, rois = res["freq"], res["rois"]
    pretty_rois = [roi.replace("_", " ") for roi in rois]
    if feature == "power":
        power = lpne.unsqueeze_triangular_array(res["power"], dim=1)  # [w,r,r,f]
    elif feature == "dir_spec":
        power = res["dir_spec"]  # [w,r,r,f]
    else:
        raise NotImplementedError(feature)

    # Get reconstructed features.
    flag = model is not None
    if flag:
        rec_power = model.reconstruct(np.transpose(power, [0, 3, 1, 2]))
        rec_power = np.transpose(rec_power, [0, 2, 3, 1])  # [w,r,r,f]
    else:
        rec_power = None

    def get_time_str(k):
        t = k * window_step
        time_str = str(int(np.floor(t / 3600))).zfill(2)
        time_str += ":" + str(int(np.floor(t / 60))).zfill(2)
        time_str += ":" + str(int(np.floor(t % 60))).zfill(2)
        return time_str

    # Set up the plot and write all the frames.
    if mode == "grid":
        fig, axarr = _set_up_grid_plot(power, pretty_rois)
        title = fig.suptitle("", fontsize=GRID_FONTSIZE, y=0.93)
        frames = []
        for k in tqdm(range(power.shape[0])):
            title.set_text(get_time_str(k))
            handles = _update_grid_plot(
                k, rois, freq, power, alpha, flag, axarr, rec_power
            )
            frames.append(mplfig_to_npimage(fig))
            for handle in handles:
                handle.remove()
    elif mode == "circle":
        circle_params = {**DEFAULT_CIRCLE_PARAMS, **circle_params}
        # The circle plot can only handle one factor at a time.
        power = rec_power if flag else power

        # Make some variables.
        r2 = circle_params["outer_radius"]
        center_angles = np.linspace(0, 2 * np.pi, len(rois) + 1)
        buffer = circle_params["buffer_percent"] / 100.0 * 2 * np.pi
        start_angles = center_angles[:-1] + buffer
        stop_angles = center_angles[1:] - buffer
        freq_diff = (stop_angles[0] - start_angles[0]) / (len(freq) + 1)
        min_val, max_val = np.quantile(power, circle_params["min_max_quantiles"])
        power = np.clip((power - min_val) / (max_val - min_val), 0, 1)
        power *= circle_params["max_alpha"]

        fig, ax = _set_up_circle_plot(
            start_angles,
            stop_angles,
            R1,
            r2,
            pretty_rois,
            freq,
            circle_params["freq_ticks"],
        )

        title = fig.suptitle("", fontsize=CIRCLE_FONTSIZE, y=0.93)
        frames = []
        for k in tqdm(range(power.shape[0])):
            title.set_text(get_time_str(k))
            handles = _update_circle_plot(
                power[k],
                ax,
                start_angles,
                stop_angles,
                freq_diff,
                circle_params["outer_radius"],
                circle_params["color"],
            )
            frames.append(mplfig_to_npimage(fig))
            for handle in handles:
                handle.remove()

    else:
        raise NotImplementedError(mode)

    # Save the movie.
    animation = ImageSequenceClip(frames, fps=fps)
    if fn.endswith(".gif"):
        animation.write_gif(fn, fps=fps)
    else:
        animation.write_videofile(fn, fps=fps, bitrate="1000k")
    plt.close("all")


def _update_grid_plot(k, rois, freq, power, alpha, flag, axarr, rec_power):
    """ """
    handles = []
    for i in range(len(rois)):
        for j in range(len(rois)):
            handle = axarr[i, j].fill_between(
                freq,
                power[k, i, j],
                fc=COLOR_1,
                alpha=alpha,
            )
            handles.append(handle)
            if flag:
                handle = axarr[i, j].fill_between(
                    freq,
                    rec_power[k, i, j],
                    fc=COLOR_2,
                    alpha=alpha,
                )
                handles.append(handle)
    return handles


def _set_up_grid_plot(power, pretty_rois):
    """
    Make the base plot.

    Parameters
    ----------
    power : numpy.ndarray
        Shape: ``[w,r,r,f]``
    pretty_rois : list of str
        Shape: ``[f]``
    """
    ylim = (-0.05 * np.max(power), 1.05 * np.max(power))
    n = power.shape[1]
    assert n == len(pretty_rois), f"{n} != len({pretty_rois})"
    fig, axarr = plt.subplots(n, n, figsize=(8, 8))
    for i in range(n):
        for j in range(i + 1):
            for dir in axarr[i, j].spines:
                axarr[i, j].spines[dir].set_visible(False)
            plt.sca(axarr[i, j])
            plt.xticks([])
            plt.yticks([])
            plt.ylim(ylim)
            if j < i:
                for dir in axarr[i, j].spines:
                    axarr[j, i].spines[dir].set_visible(False)
                plt.sca(axarr[j, i])
                plt.xticks([])
                plt.yticks([])
                plt.ylim(ylim)
            if j == 0:
                plt.sca(axarr[i, j])
                plt.ylabel(pretty_rois[i], size="xx-small", rotation=30)
            if i == n - 1:
                plt.sca(axarr[i, j])
                plt.xlabel(pretty_rois[j], size="xx-small", rotation=30)
    return fig, axarr


if __name__ == "__main__":
    pass


###
