"""
Make a scatterplot on the 2-simplex.

"""
__date__ = "December 2022"
__all__ = ["simplex_plot"]


import matplotlib.pyplot as plt
import numpy as np

GAP = 0.05


def simplex_plot(
    probs, color="k", alpha=0.7, scatter_size=1.5, class_names=None, fn="temp.pdf"
):
    """
    Make a scatterplot on the 2-simplex.

    Parameters
    ----------
    probs : numpy.ndarray
        Class probabilities.
        Shape: ``[n_points, 3]``
    color : str, optional
        Scatter color
    alpha : float, optional
        Scatter transparency
    scatter_size : float, optional
        Scatter size
    class_names : None or list of str, optional
        Class names to put at the corners of the simplex
    fn : str, optional
        Image filename
    """
    assert probs.ndim == 2, f"len({probs.shape}) != 2"
    assert probs.shape[1] == 3, f"{probs.shape}[1] != 3, probs must have three classes!"
    assert np.allclose(
        np.sum(probs, axis=1), np.ones(len(probs))
    ), f"probs must be normalized!"
    x, y = (probs[:, 0] + 2 * probs[:, 2]) / np.sqrt(3), probs[:, 0]
    plt.plot([0, 1 / np.sqrt(3), 2 / np.sqrt(3), 0], [0, 1, 0, 0], c="k", lw=1.5)
    plt.scatter(x, y, alpha=alpha, s=scatter_size, c=color)
    plt.gca().set_aspect("equal")
    plt.axis("off")
    if class_names is not None:
        assert len(class_names) == 3, "Three class names must be provided!"
        plt.text(1 / np.sqrt(3), 1 + GAP, class_names[0], ha="center", va="center")
        plt.text(-GAP, -GAP, class_names[1], ha="center", va="center")
        plt.text(2 / np.sqrt(3) + GAP, -GAP, class_names[2], ha="center", va="center")
    plt.savefig(fn)
    plt.close("all")


if __name__ == "__main__":
    pass


###
