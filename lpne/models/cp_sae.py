"""
CANDECOMP/PARAFAC supervised autoencoder with deterministic factors

"""
__date__ = "November 2021 - July 2022"


import numpy as np
from sklearn.utils.validation import check_is_fitted
import torch
from torch.distributions import Categorical, MultivariateNormal
import torch.nn.functional as F
import warnings

from .base_model import BaseModel
from .. import INVALID_LABEL, INVALID_GROUP
from ..utils.utils import get_weights, squeeze_triangular_array


FLOAT = torch.float32
INT = torch.int64
DEFAULT_GP_PARAMS = {
    "mean": 0.0,
    "ls": 0.3,
    "obs_noise_var": 1e-3,
    "reg": 0.1,
    "kernel": "se",
}
"""Default frequency factor GP parameters"""


class CpSae(BaseModel):
    """
    A supervised autoencoder with a CP-style generative model

    Parameters
    ----------
    reg_strength : float, optional
        This controls how much we weight the reconstruction loss. This
        should be positive, and larger values indicate more regularization.
    z_dim : int, optional
        Latent dimension/number of networks.
    gp_params : dict, optional
        Maps the frequency component GP prior parameter names to values.
        ``'mean'`` : float, optional
            Mean value
        ``'ls'`` : float, optional
            Lengthscale, in units of frequency bins
        ``'obs_noise_var'`` : float, optional
            Observation noise variances
        ``'reg'`` : float, optional
            Regularization strength
        ``'kernel'`` : {``'ou'``, ``'se'``}, optional
            Denotes Ornstein-Uhlenbeck or squared exponential kernels
    encoder_type : str, optional
        One of ``'linear'``, ``'pinv'``, or ``'irls'``. If
        ``rec_loss_type`` is ``'lad'``, the encoder should be ``'linear'``
        or ``'pinv'``. If ``rec_loss_type`` is ``'ls'``, the encoder should
        be ``'linear'`` or ``'irls'``. Defaults to ``'linear'``.
    rec_loss_type : str, optional
        One of ``'lad'`` for least absolute deviations or ``'ls'`` for
        least squares. Defaults to ``'lad'``.
    irls_iter : int, opional
        Number of iterations to run iteratively reweighted least squares.
        Defaults to ``1``.
    """

    MODEL_NAME = "CpSae"
    FIT_ATTRIBUTES = ["classes_", "groups_", "iter_"]

    def __init__(
        self,
        reg_strength=1.0,
        z_dim=32,
        gp_params=DEFAULT_GP_PARAMS,
        encoder_type="linear",
        rec_loss_type="lad",
        irls_iter=1,
        **kwargs,
    ):
        super(CpSae, self).__init__(**kwargs)
        assert isinstance(reg_strength, (int, float)), f"found {type(reg_strength)}"
        self.reg_strength = float(reg_strength)
        assert isinstance(z_dim, int), f"found {type(z_dim)}"
        self.z_dim = z_dim
        self.gp_params = {**DEFAULT_GP_PARAMS, **gp_params}
        assert encoder_type in ["linear", "pinv", "irls"]
        self.encoder_type = encoder_type
        assert rec_loss_type in ["lad", "ls"]
        self.rec_loss_type = rec_loss_type
        assert isinstance(irls_iter, int), f"found {type(irls_iter)}"
        self.irls_iter = irls_iter

    @torch.no_grad()
    def _initialize(self):
        """
        Initialize the network parameters.

        """
        _, n_freqs, n_rois, _ = self.features_shape_
        n_classes = len(self.classes_)
        # Make the recognition model.
        self.rec_model = torch.nn.Linear(n_freqs * n_rois**2, self.z_dim)
        # Set up the frequency factor GP.
        kernel = torch.arange(n_freqs).unsqueeze(0)
        kernel = torch.abs(kernel - torch.arange(n_freqs).unsqueeze(1))
        if self.gp_params["kernel"] == "se":
            kernel = 2 ** (-1 / 2) * torch.pow(kernel / self.gp_params["ls"], 2)
        elif self.gp_params["kernel"] == "ou":
            kernel = torch.abs(kernel / self.gp_params["ls"])
        else:
            raise NotImplementedError(self.gp_params["kernel"])
        kernel = torch.exp(-kernel)
        kernel = kernel + self.gp_params["obs_noise_var"] * torch.eye(n_freqs)
        self.gp_dist = MultivariateNormal(
            self.gp_params["mean"] * torch.ones(n_freqs).to(self.device),
            covariance_matrix=kernel.to(self.device),
        )
        # Make the frequency factors.
        self.freq_factors = torch.nn.Parameter(
            self.gp_dist.sample(sample_shape=(self.z_dim,)),
        )  # [z,f]
        # Make the ROI factors.
        self.roi_1_factors = torch.nn.Parameter(
            -5 + torch.randn(self.z_dim, n_rois),
        )  # [z,r]
        self.roi_2_factors = torch.nn.Parameter(
            -5 + torch.randn(self.z_dim, n_rois),
        )  # [z,r]
        self.logit_weights = torch.nn.Parameter(
            -5 * torch.ones(1, n_classes),
        )  # [1,c]
        self.logit_biases = torch.nn.Parameter(torch.zeros(1, n_classes))
        super(CpSae, self)._initialize()

    def forward(self, features, labels, groups, weights, return_logits=False):
        """
        Calculate a loss.

        Parameters
        ----------
        features : torch.Tensor
            Shape: ``[b,f,r,r]``
        labels : None or torch.Tensor
            Shape: ``[b]``
        groups : None or torch.Tensor
            Shape: ``[b]``
        weights : torch.Tensor
            Shape: ``[b]``

        Returns
        -------
        loss : torch.Tensor
            Shape: ``[]``
        """
        if labels is not None:
            unlabeled_mask = torch.isinf(1 / (labels - INVALID_LABEL))
            labels[unlabeled_mask] = 0

        # Get latents.
        zs = self.get_latents(features)  # [b,z]

        if groups is None:
            assert return_logits
            logits = zs[:, : self.n_classes] * F.softplus(self.logit_weights)
            logits = logits + self.logit_biases  # [b,c]
            return logits

        # Get the reconstruction loss.
        rec_features = self.project_latents(zs)  # [b,fr^2]
        flat_features = features.view(features.shape[0], -1)  # [b,fr^2]
        diff = flat_features - rec_features  # [b,fr^2]
        if self.rec_loss_type == "lad":
            rec_loss = torch.abs(diff).mean(dim=1)  # [b]
        elif self.rec_loss_type == "ls":
            rec_loss = 0.5 * torch.pow(diff, 2).mean(dim=1)  # [b]
        rec_loss = self.reg_strength * rec_loss  # [b]

        # Predict the labels and get weighted label log probabilities.
        logits = zs[:, : self.n_classes] * F.softplus(self.logit_weights)
        logits = logits + self.logit_biases  # [b,c]
        if return_logits:
            return logits
        log_probs = Categorical(logits=logits).log_prob(labels)  # [b]
        log_probs = weights * log_probs  # [b]
        log_probs[unlabeled_mask] = 0.0  # disregard the unlabeled data

        # Calculate the GP loss.
        freq_f = F.softplus(self.freq_factors)  # [g,z,f]
        freq_norm = torch.sqrt(torch.pow(freq_f, 2).sum(dim=-1, keepdim=True))
        freq_f = freq_f / freq_norm
        gp_loss = -self.gp_dist.log_prob(freq_f).sum()
        gp_loss = self.gp_params["reg"] * gp_loss

        # Combine all the terms into a composite loss.
        label_loss = -torch.sum(log_probs)  # []
        rec_loss = torch.sum(rec_loss)  # []
        loss = label_loss + rec_loss + gp_loss
        return loss

    def get_latents(self, features, reg=1e-3):
        """
        Get the latents corresponding to the given features.

        Parameters
        ----------
        features : torch.Tensor
            Shape: ``[b,f,r,r]``
        reg : float, optional
            Regularization for IRLS

        Returns
        -------
        latents : torch.Tensor
            Shape: ``[b,z]``
        """
        if self.encoder_type == "linear":
            flat_features = features.view(features.shape[0], -1)  # [b,fr^2]
            latents = F.softplus(self.rec_model(flat_features))  # [b,z]
        elif self.encoder_type in ["pinv", "irls"]:
            # [z,f,r,r], [z,f], [z,r], [z,r]
            H, f1, f2, f3 = self._get_H(flatten=False, return_factors=True)
            H = H.unsqueeze(0)  # [1,z,f,r,r]
            prod = (features.unsqueeze(1) * H).sum(dim=(2, 3, 4))  # [b,z]
            inner = (f1 @ f1.t()) * (f2 @ f2.t()) * (f3 @ f3.t())  # [z,z]
            latents = torch.linalg.solve(
                inner.unsqueeze(0),
                prod.unsqueeze(-1),
            )  # [b,z,1]
            if self.encoder_type == "pinv":
                latents = torch.clamp(latents.squeeze(-1), min=0.0)
            else:
                # Do iteratively re-weighted least squares.
                flat_features = features.view(features.shape[0], -1)  # [b,fr^2]
                flat_H = H.view(self.z_dim, -1)  # [z,fr^2]
                for _ in range(self.irls_iter):
                    # diffs: [b,x], weights: [b,fr^2]
                    diffs = flat_features - latents.squeeze(-1) @ flat_H
                    weights = 1.0 / torch.clamp(torch.abs(diffs), reg, None)
                    inner = torch.einsum(
                        "zx,bx,xw->bzw",
                        flat_H,
                        weights,
                        flat_H.t(),
                    )  # [b,z,z]
                    prod = torch.einsum(
                        "zx,bx->bz",
                        flat_H,
                        weights * flat_features,
                    )  # [b,z]
                    latents = torch.linalg.solve(
                        inner,
                        prod.unsqueeze(-1),
                    )  # [b,z,1]
                latents = torch.clamp(latents.squeeze(-1), min=0.0)
        else:
            raise NotImplementedError(self.encoder_type)
        return latents

    def project_latents(self, latents):
        """
        Feed latents through the model to get observations.

        Parameters
        ----------
        latents : torch.Tensor
            Shape: [b,z]

        Returns
        -------
        x_pred : torch.Tensor
            Shape: [b,x]
        """
        H = self._get_H(flatten=True)  # [z,x]
        rec_features = latents.unsqueeze(1) @ H.unsqueeze(0)  # [b,1,x]
        rec_features = rec_features.squeeze(1)  # [b,x]
        return rec_features

    def _get_H(self, flatten=True, return_factors=False):
        """
        Get the factors.

        Parameters
        ----------
        flatten : bool, optional
        return_factors : bool, optional

        Returns
        -------
        H: torch.Tensor
            Shape: ``[z,frr]`` if ``flatten``, ``[z,f,r,r]`` otherwise
        freq_factor : torch.Tensor
            Returned if ``return_factors``
            Shape : [z,f]
        roi_1_factor : torch.Tensor
            Returned if ``return_factors``
            Shape : [z,r]
        roi_2_factor : torch.Tensor
            Returned if ``return_factors``
            Shape : [z,r]
        """
        freq_f = F.softplus(self.freq_factors)  # [z,f]
        roi_1_f = F.softplus(self.roi_1_factors)  # [z,r]
        roi_2_f = F.softplus(self.roi_2_factors)  # [z,r]
        volume = torch.einsum(
            "zf,zr,zs->zfrs",
            freq_f,
            roi_1_f,
            roi_2_f,
        )  # [z,f,r,r]
        if flatten:
            volume = volume.view(volume.shape[0], -1)
        if return_factors:
            return volume, freq_f, roi_1_f, roi_2_f
        return volume

    @torch.no_grad()
    def _get_mean_projection(self):
        return self._get_H(flatten=False)  # [z,f,r,r]

    @torch.no_grad()
    def predict_proba(
        self, features, groups=None, to_numpy=True, return_logits=False, warn=True
    ):
        """
        Get prediction probabilities for the given features.

        Note
        ----
        * This should be consistent with `self.forward`.

        Parameters
        ----------
        features : numpy.ndarray or torch.Tensor
            Shape: ``[b,f,r,r]``
        groups : ``None`` or numpy.ndarray
            Shape: ``[b]``
        to_numpy : bool, optional
            Whether to return a NumPy array or a Pytorch tensor
        return_logits : bool, optional
            Return unnormalized logits instead of probabilities.
        warn : bool, optional
            Whether to warn the user when there are unrecognized groups

        Returns
        -------
        probs : numpy.ndarray
            Shape: ``[batch, n_classes]``
        """
        check_is_fitted(self, attributes=self.FIT_ATTRIBUTES)
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=FLOAT)
        # Figure out group mapping.
        if groups is not None:
            if isinstance(groups, torch.Tensor):
                groups = groups.detach().cpu().numpy()
            # Figure out the group mapping.
            temp_groups = np.unique(groups)
            setdiff = np.setdiff1d(
                temp_groups,
                self.groups_,
                assume_unique=True,
            )
            # Warn the user if there are groups we didn't see in training.
            if len(setdiff) != 0 and warn:
                warnings.warn(
                    f"Found unexpected groups: {setdiff}\n"
                    f"Passed to predict: {temp_groups}\n"
                    f"Passed to fit: {self.groups_}",
                )
            new_groups = np.zeros_like(groups)
            group_list = self.groups_.tolist()
            for temp_group in temp_groups:
                idx = groups == temp_group
                if temp_group in group_list:
                    new_groups[idx] = group_list.index(temp_group)
                else:
                    new_groups[idx] = INVALID_GROUP
            groups = new_groups
            # To PyTorch Tensors.
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=FLOAT)
            groups = torch.tensor(groups, dtype=INT).to(self.device)
        logits = []
        i = 0
        while i <= len(features):
            batch_f = features[i : i + self.batch_size].to(self.device)
            batch_g = None if groups is None else groups[i : i + self.batch_size]
            batch_logit = self(batch_f, None, batch_g, None, return_logits=True)
            logits.append(batch_logit)
            i += self.batch_size
        logits = torch.cat(logits, dim=0)
        if return_logits:
            to_return = logits
        else:
            to_return = F.softmax(logits, dim=1)  # [b, n_classes]
        if to_numpy:
            return to_return.detach().cpu().numpy()
        return to_return

    @torch.no_grad()
    def predict(self, features, groups=None, warn=True):
        """
        Predict class labels for the features.

        Parameters
        ----------
        features : numpy.ndarray
            Shape: ``[b,f,r,r]``
        groups : None or numpy.ndarray, optional
            Shape: ``[b]``
        warn : bool, optional
            Whether to warn the user when there are unrecognized groups.

        Returns
        -------
        predictions : numpy.ndarray
            Shape: ``[b]``
        """
        # Checks
        assert features.ndim == 4
        assert features.shape[2] == features.shape[3]
        check_is_fitted(self, attributes=self.FIT_ATTRIBUTES)
        # Feed through model.
        probs = self.predict_proba(features, groups, to_numpy=True, warn=warn)
        predictions = np.argmax(probs, axis=1)
        return self.classes_[predictions]

    @torch.no_grad()
    def score(self, features, labels, groups=None, warn=True):
        """
        Get a class weighted accuracy.

        This is the objective we really care about, which doesn't contain the
        regularization in the `forward` method.

        Parameters
        ----------
        features : numpy.ndarray
            Shape: ``[b,f,r,r]``
        labels : numpy.ndarray
            Shape: ``[b]``
        groups : ``None`` or numpy.ndarray, optional
            Shape: ``[b]``
        warn : bool, optional
            Whether to warn the user when there are unrecognized groups.

        Return
        ------
        weighted_accuracy : float
        """
        check_is_fitted(self, attributes=self.FIT_ATTRIBUTES)
        weights = get_weights(labels, groups, invalid_label=INVALID_LABEL)
        predictions = self.predict(features, groups, warn=warn)
        scores = np.zeros(len(features))
        scores[predictions == labels] = 1.0
        scores = scores * weights
        weighted_accuracy = np.mean(scores[labels != INVALID_LABEL])
        return weighted_accuracy

    @torch.no_grad()
    def get_factor(self, factor_num=0):
        """
        Get a linear factor.

        Parameters
        ----------
        feature_num : int
            Which factor to return. ``0 <= factor_num < self.z_dim``

        Returns
        -------
        factor : numpy.ndarray
            Shape: ``[r(r+1)/2,f]``
        """
        check_is_fitted(self, attributes=self.FIT_ATTRIBUTES)
        assert isinstance(factor_num, int), f"found {type(factor_num)}"
        assert factor_num >= 0 and factor_num < self.z_dim
        volume = self._get_mean_projection()[factor_num]  # [f,r,r]
        volume = volume.detach().cpu().numpy()  # [f,r,r]
        volume = squeeze_triangular_array(volume, dims=(1, 2))  # [f,r(r+1)/2]
        return volume.T  # [r(r+1)/2,f]

    @torch.no_grad()
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        super_params = super(CpSae, self).get_params(deep=deep)
        params = {
            "reg_strength": self.reg_strength,
            "z_dim": self.z_dim,
            "gp_params": self.gp_params,
            "encoder_type": self.encoder_type,
            "rec_loss_type": self.rec_loss_type,
            "irls_iter": self.irls_iter,
        }
        params = {**super_params, **params}
        return params

    @torch.no_grad()
    def set_params(
        self,
        reg_strength=None,
        z_dim=None,
        gp_params=None,
        encoder_type=None,
        rec_loss_type=None,
        irls_iter=None,
        **kwargs,
    ):
        """Set the parameters of this estimator."""
        if reg_strength is not None:
            self.reg_strength = reg_strength
        if z_dim is not None:
            self.z_dim = z_dim
        if gp_params is not None:
            self.gp_params = {**DEFAULT_GP_PARAMS, **gp_params}
        if encoder_type is not None:
            self.encoder_type = encoder_type
        if rec_loss_type is not None:
            self.rec_loss_type = rec_loss_type
        if irls_iter is not None:
            self.irls_iter = irls_iter
        super(CpSae, self).set_params(**kwargs)
        return self


if __name__ == "__main__":
    pass


###
