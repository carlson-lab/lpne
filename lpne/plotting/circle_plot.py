"""
Circle plots for representing power and cross-power features or factors

"""
__date__ = "November 2022 - April 2023"
__all__ = ["circle_plot"]

import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt


R1 = 1.0  # inner radius of power plots

import lpne


def circle_plot(
    factor,
    rois=None,
    freqs=None,
    freq_ticks=None,
    max_alpha=0.7,
    buffer_percent=1.0,
    outer_radius=1.2,
    min_max_quantiles=(0.5, 0.9),
    color="tab:blue",
    negative_color="tab:red",
    fn="temp.pdf",
):
    """
    Make a circle plot representing power and cross-power features or factors.

    Parameters
    ----------
    factor : numpy.ndarray
        Shape: ``[n_roi,n_roi,n_freq]``
    rois : list of str, optional
        ROI names
    freqs : None or numpy.ndarray
    freq_ticks : None or list
    max_alpha : float, optional
        Maximum transparency
    buffer_percent : float, optional
    outer_radius : float, optional
    min_max_quantiles : None or tuple
    color : str, optional
        The color used to plot the positive values of ``factor``.
    negative_color : str, optional
        The color used to plot the negative values of ``factor``.
    fn : str, optional
        Image filename
    """
    # Check the arguments.
    assert factor.ndim == 3
    assert factor.shape[0] == factor.shape[1]
    assert max_alpha >= 0.0 and max_alpha <= 1.0, f"{max_alpha}"
    n_roi, n_freq = factor.shape[1:]
    assert freqs is None or len(freqs) == n_freq, f"{len(freqs)} != {n_freq}"
    if rois is not None:
        assert len(rois) == n_roi, f"{len(rois)} != {n_roi}"
        pretty_rois = [roi.replace("_", " ") for roi in rois]

    # Make some variables.
    r2 = outer_radius
    center_angles = np.linspace(0, 2 * np.pi, n_roi + 1)
    buffer = buffer_percent / 100.0 * 2 * np.pi
    start_angles = center_angles[:-1] + buffer
    stop_angles = center_angles[1:] - buffer
    freq_diff = (stop_angles[0] - start_angles[0]) / (n_freq + 1)
    min_val, max_val = np.quantile(np.abs(factor), min_max_quantiles)
    factor1 = max_alpha * np.clip((factor - min_val) / (max_val - min_val), 0, 1)
    factor2 = max_alpha * np.clip((-factor - min_val) / (max_val - min_val), 0, 1)

    # Set up axes and labels and ticks.
    _, ax = _set_up_circle_plot(
        start_angles, stop_angles, R1, r2, pretty_rois, freqs, freq_ticks
    )

    # Add the power and chord plots.
    _update_circle_plot(
        factor1,
        ax,
        start_angles,
        stop_angles,
        freq_diff,
        outer_radius,
        color,
    )

    # Add the negative color.
    if np.min(factor) < 0.0:
        _update_circle_plot(
            factor2,
            ax,
            start_angles,
            stop_angles,
            freq_diff,
            outer_radius,
            negative_color,
        )

    # Save and close.
    plt.savefig(fn)
    plt.close("all")


def _update_circle_plot(
    factor, ax, start_angles, stop_angles, freq_diff, outer_radius, color
):
    """ """
    r2 = outer_radius
    handles = []
    n_roi, n_freq = factor.shape[1:]
    # Draw the power plots.
    for i, (c1, c2) in enumerate(zip(start_angles, stop_angles)):
        for j in range(n_freq):
            if factor[i, i, j] > 0:
                diff_1 = j * (c2 - c1) / n_freq
                diff_2 = (j + 1) * (c2 - c1) / n_freq
                alpha = factor[i, i, j]
                h = _arc_patch(
                    R1, r2, c1 + diff_1, c1 + diff_2, color, ax, n=5, alpha=alpha
                )
                handles.append(h)
    # Draw the chords to represent cross-power.
    for i in range(n_roi - 1):
        for j in range(i + 1, n_roi):
            for k in range(n_freq):
                if factor[i, j, k] > 0.0:
                    theta_1 = start_angles[i] + freq_diff * k
                    theta_2 = start_angles[j] + freq_diff * k
                    alpha = factor[i, j, k]
                    h = _plot_poly_chord(
                        theta_1,
                        theta_2,
                        freq_diff,
                        color,
                        ax,
                        alpha=alpha,
                    )
                    handles.append(h)
    return handles


def _set_up_circle_plot(
    start_angles, stop_angles, r1, r2, pretty_rois, freqs, freq_ticks
):
    """ """
    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    # Set up axes and draw power plots.
    for i, (c1, c2) in enumerate(zip(start_angles, stop_angles)):
        _draw_power_axis(r1, r2, c1, c2, ax)
        if freqs is not None and freq_ticks is not None:
            _plot_ticks(r2, c1, c2, ax, freqs, freq_ticks)
        if pretty_rois is not None:
            _plot_roi_name(r2, 0.5 * (c1 + c2), ax, pretty_rois[i])

    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(-1.5, 1.5)
    plt.axis("off")
    return fig, ax


def _plot_poly_chord(theta_1, theta_2, diff, color, ax, n=50, alpha=0.5):
    """ """
    points_1 = _chord_helper(theta_1, theta_2, n=n)
    rot_mat = np.array([[np.cos(diff), -np.sin(diff)], [np.sin(diff), np.cos(diff)]])
    points_2 = rot_mat @ points_1
    points = np.concatenate([points_1, points_2[:, ::-1]], axis=1).T
    poly = Polygon(points, closed=True, fc=to_rgba(color, alpha=alpha))
    ax.add_patch(poly)
    return poly


def _chord_helper(theta_1, theta_2, n=50):
    """ """
    a1, a2 = np.cos(theta_1), np.sin(theta_1)
    b1, b2 = np.cos(theta_2), np.sin(theta_2)
    denom = a1 * b2 - a2 * b1
    if np.abs(denom) < 1e-5:
        xs = np.linspace(a1, b1, n)
        ys = np.linspace(a2, b2, n)
        return np.vstack([xs, ys])
    v, w = 2 * (a2 - b2) / denom, 2 * (b1 - a1) / denom
    center = (-v / 2, -w / 2)
    radius = np.sqrt(-1 + (v**2 + w**2) / 4)
    angle_1 = np.arctan2(a2 - center[1], a1 - center[0])
    angle_2 = np.arctan2(b2 - center[1], b1 - center[0])
    angle_1, angle_2 = min(angle_1, angle_2), max(angle_1, angle_2)
    if angle_2 - angle_1 > np.pi:
        angle_1, angle_2 = angle_2, angle_1 + 2 * np.pi
    theta = np.linspace(angle_1, angle_2, n)
    xs = radius * np.cos(theta) + center[0]
    ys = radius * np.sin(theta) + center[1]
    return np.vstack([xs, ys])


def _arc_patch(r1, r2, theta1, theta2, color, ax, n=50, alpha=1.0, **kwargs):
    """ """
    thetas = np.linspace(theta1, theta2, n)
    sin_thetas, cos_thetas = np.sin(thetas), np.cos(thetas)
    points = np.vstack([cos_thetas, sin_thetas]).T
    points = np.concatenate([r1 * points, r2 * points[::-1]], axis=0)
    poly = Polygon(points, closed=True, fc=to_rgba(color, alpha=alpha), **kwargs)
    ax.add_patch(poly)
    return poly


def _draw_power_axis(r1, r2, theta1, theta2, ax, n=50, **kwargs):
    """ """
    thetas = np.linspace(theta1, theta2, n)
    sin_thetas, cos_thetas = np.sin(thetas), np.cos(thetas)
    points = np.vstack([cos_thetas, sin_thetas]).T
    points = np.concatenate([r1 * points, r2 * points[::-1]], axis=0)
    points = np.concatenate([points, points[:1]], axis=0)
    handle = ax.plot(points[:, 0], points[:, 1], c="k", **kwargs)
    return handle


def _plot_ticks(
    r,
    theta_1,
    theta_2,
    ax,
    freqs,
    freq_ticks,
    tick_extent=0.03,
    tick_label_extent=0.08,
    n=5,
    **kwargs,
):
    offset = 0 if np.cos((theta_1 + theta_2) / 2) > 0 else 180
    for freq in freq_ticks:
        theta = theta_1 + (theta_2 - theta_1) * (freq - freqs[0]) / (
            freqs[-1] - freqs[0]
        )
        x = [r * np.cos(theta), (r + tick_extent) * np.cos(theta)]
        y = [r * np.sin(theta), (r + tick_extent) * np.sin(theta)]
        ax.plot(x, y, c="k", **kwargs)
        x = (r + tick_label_extent) * np.cos(theta)
        y = (r + tick_label_extent) * np.sin(theta)
        rotation = theta * 180 / np.pi + offset
        ax.text(x, y, str(freq), rotation=rotation, ha="center", va="center")


def _plot_roi_name(r, theta, ax, roi, extent=0.25, fontsize=16):
    x, y = (r + extent) * np.cos(theta), (r + extent) * np.sin(theta)
    rotation = -90 + theta * 180 / np.pi
    if np.sin(theta) < 0:
        rotation += 180.0
    ax.text(x, y, roi, rotation=rotation, ha="center", va="center", fontsize=fontsize)


if __name__ == "__main__":
    factor = np.random.randn(8, 8, 29)
    freqs = np.arange(factor.shape[2])
    freq_ticks = [0, 5, 8, 15]
    rois = ["A_sdfk", "B_asds", "C_asds", "D_qwe", "E_sdf", "F_sadf", "G", "H"]

    circle_plot(
        factor,
        freqs=freqs,
        freq_ticks=freq_ticks,
        rois=rois,
        min_max_quantiles=[0.5, 1.0],
    )


###
