"""
Useful functions for SAE models

"""
__date__ = "October 2022"

from sklearn.decomposition import PCA
import numpy as np


def get_reconstruction_stats(model, features):
    """
    Get statistics related to the quality of reconstructions under the model.

    NOTE: A zero-mean PCA would give a tighter bound for our case.

    Parameters
    ----------
    model : BaseModel
    features : numpy.ndarray
        Shape: ``[b,x]`` or ``[b,f,r,r]``

    Returns
    -------
    res : dict
        Maps ``'r2'`` to the proportion of variance explained by the model (both the
        particular encoder and decoder). Maps ``'r2_ub'`` to a PCA-derived upper bound
        on the R^2 value given the number of latent dimensions. Maps ``'r2_ubs'`` to
        PCA-derived upper bounds for succesively increasing number of latent dimensions.
    """
    assert features.ndim in [2, 4]
    if features.ndim == 2:
        idx = np.argwhere(np.isnan(features).sum(axis=1) == 0).flatten()
    else:
        idx = np.argwhere(np.isnan(features).sum(axis=(1, 2, 3)) == 0).flatten()
    features = features[idx]
    n = len(features)
    # Make sure we have enough windows to do PCA.
    assert (
        n >= model.z_dim
    ), f"We need at least {model.z_dim} windows to do PCA, found {n}"
    # Calculate the R^2.
    rec_features = model.reconstruct(features)
    mean_features = np.mean(features, axis=0, keepdims=True)
    orig_variance = np.power(features - mean_features, 2).sum() / n
    residual_variance = np.power(features - rec_features, 2).sum() / n
    r2 = 1.0 - residual_variance / orig_variance
    # Calculate upper bounds of the R^2 using PCA.
    flat_features = features.reshape(n, -1)
    pca = PCA(n_components=model.z_dim).fit(flat_features)
    r2_ubs = np.cumsum(pca.explained_variance_ratio_)
    r2_ub = r2_ubs[-1]
    return dict(r2=r2, r2_ub=r2_ub, r2_ubs=r2_ubs)


def get_reconstruction_summary(model, features):
    """
    Return a message summarizing the statistics related to reconstructions.

    Parameters
    ----------
    model : BaseModel
    features : numpy.ndarray
        Shape: ``[b,x]`` or ``[b,f,r,r]``

    Returns
    -------
    msg : str
        Message
    """
    res = get_reconstruction_stats(model, features)
    msg = (
        f"Reconstruction stats:\n\tProportion of explained variance: {res['r2']:.3f}"
        f"\n\tLinear upper bound: {res['r2_ub']:.3f}\n\tSuccessive UBs for increasing"
        f" latent dims: "
    )
    if model.z_dim <= 5:
        msg += ", ".join([f"{i:.3f}" for i in res["r2_ubs"]])
    else:
        q = res["r2_ubs"]
        msg += f"{q[0]:.3f}, {q[1]:.3f}, ..., {q[-2]:.3f}, {q[-1]:.3f}"
    return msg


if __name__ == "__main__":
    pass


###
