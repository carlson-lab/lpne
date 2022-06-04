"""
Factor Analysis-regularized logistic regression.

"""
__date__ = "June 2021 - June 2022"


import numpy as np
import torch
from torch.distributions import Categorical, Normal, kl_divergence
import torch.nn.functional as F
import warnings

from .base_model import BaseModel
from .. import INVALID_LABEL
from ..utils.utils import get_weights


FLOAT = torch.float32
INT = torch.int64
MAX_LABEL = 1000
EPSILON = 1e-6
FIT_ATTRIBUTES = ['classes_']



class FaSae(BaseModel):

    MODEL_NAME = 'FA SAE'


    def __init__(self, reg_strength=1.0, z_dim=32, weight_reg=0.0,
        nonnegative=True, variational=False, kl_factor=1.0, **kwargs):
        """
        A supervised autoencoder with nonnegative and variational options.

        Notes
        -----
        * The `labels` argument to `fit` and `score` is a bit hacky so that the
          model can work nicely with the sklearn model selection tools. The
          labels should be an array of integers with `label // 1000` encoding
          the individual and `label % 1000` encoding the behavioral label.

        Parameters
        ----------
        reg_strength : float, optional
            This controls how much the classifier is regularized. This should
            be positive, and larger values indicate more regularization.
        z_dim : int, optional
            Latent dimension/number of networks.
        weight_reg : float, optional
            Model L2 weight regularization.
        nonnegative : bool, optional
            Use nonnegative factorization.
        variational : bool, optional
            Whether a variational autoencoder is used.
        kl_factor : float, optional
            How much to weight the KL divergence term in the variational
            autoencoder (VAE). The standard setting is `1.0`. This is a distinct
            regularization parameter from `reg_strength` that can be
            independently set. This parameter is only used if `variational` is
            `True`.
        n_iter : int, optional
            Number of gradient steps during training.
        lr : float, optional
            Learning rate.
        batch_size : int, optional
            Minibatch size
        """
        super(FaSae, self).__init__(**kwargs)
        assert kl_factor >= 0.0, f"{kl_factor} < 0"
        # Set parameters.
        assert isinstance(reg_strength, (int, float))
        assert reg_strength >= 0.0
        self.reg_strength = float(reg_strength)
        assert isinstance(z_dim, int)
        assert z_dim >= 1
        self.z_dim = z_dim
        assert isinstance(weight_reg, (int, float))
        assert weight_reg >= 0.0
        self.weight_reg = float(weight_reg)
        assert isinstance(nonnegative, bool)
        self.nonnegative = nonnegative
        assert isinstance(variational, bool)
        self.variational = variational
        assert isinstance(kl_factor, (int, float))
        assert kl_factor >= 0.0
        self.kl_factor = float(kl_factor)        
        self.classes_ = None


    def _initialize(self, feature_shape):
        """Initialize parameters of the networks before training."""
        _, n_features = feature_shape
        # Check arguments.
        n_classes = len(self.classes_)
        assert n_classes <= self.z_dim, f"{n_classes} > {self.z_dim}"
        if self.nonnegative and self.weight_reg > 0.0:
            self.weight_reg = 0.0
            warnings.warn(
                    f"Weight regularization should be 0.0 "
                    f"for nonnegative factorization"
            )
        # Make the networks.
        self.recognition_model = torch.nn.Linear(n_features, self.z_dim)
        self.rec_model_1 = torch.nn.Linear(n_features, self.z_dim)
        self.rec_model_2 = torch.nn.Linear(n_features, self.z_dim)
        self.linear_layer = torch.nn.Linear(self.z_dim, self.z_dim)
        prior_mean = torch.zeros(self.z_dim).to(self.device)
        prior_std = torch.ones(self.z_dim).to(self.device)
        self.prior = Normal(prior_mean, prior_std)
        self.model = torch.nn.Linear(self.z_dim, n_features)
        self.logit_bias = torch.nn.Parameter(torch.zeros(1,n_classes))
        super(FaSae, self)._initialize()


    def forward(self, features, labels, groups, weights):
        """
        Calculate a loss for the features and labels.

        Parameters
        ----------
        features : torch.Tensor
            Shape: [batch,n_features]
        labels : torch.Tensor
            Shape: [batch]
        groups : None or torch.Tensor
            Ignored
        weights : None or torch.Tensor
            Shape: [batch]

        Returns
        -------
        loss : torch.Tensor
            Shape: []
        """
        if self.variational:
            # Feed through the recognition network to get latents.
            z_mus = self.rec_model_1(features)
            z_log_stds = self.rec_model_2(features)
            # Make the variational posterior and get a KL from the prior.
            dist = Normal(z_mus, EPSILON + z_log_stds.exp())
            kld = kl_divergence(dist, self.prior).sum(dim=1) # [b]
            # Sample.
            zs = dist.rsample() # [b,z]
            # Project.
            zs = self.linear_layer(zs)
        else: # deterministic autoencoder
            # Feed through the recognition network to get latents.
            zs = self.recognition_model(features)
        # Reconstruct the features.
        if self.nonnegative:
            A = F.softplus(self.model.weight)
            features_rec = A.unsqueeze(0) @ F.softplus(zs).unsqueeze(-1)
            features_rec = features_rec.squeeze(-1)
        else:
            A = self.model.weight
            features_rec = self.model(zs)
        # Calculate a reconstruction loss.
        rec_loss = torch.mean((features - features_rec).pow(2), dim=1) # [b]
        rec_loss = self.reg_strength * rec_loss
        # Predict the labels.
        logits = zs[:,:len(self.classes_)-1]
        ones = torch.ones(
                logits.shape[0],
                1,
                dtype=logits.dtype,
                device=logits.device,
        )
        logits = torch.cat([logits, ones], dim=1) + self.logit_bias
        log_probs = Categorical(logits=logits).log_prob(labels) # [b]
        # Weight label log likes by class weights.
        if weights is not None:
            assert weights.shape == labels.shape
            log_probs = weights * log_probs
        # Regularize the model weights.
        l2_loss = self.weight_reg * torch.norm(A)
        # Combine all the terms into a composite loss.
        loss = rec_loss - log_probs
        if self.variational:
            loss = loss + self.kl_factor * kld
        loss = torch.mean(loss) + l2_loss
        return loss


    @torch.no_grad()
    def predict_proba(self, features, to_numpy=True, stochastic=False):
        """
        Probability estimates.

        Note
        ----
        * This should be consistent with `self.forward`.

        Parameters
        ----------
        features : numpy.ndarray
            Shape: [batch, n_features]
        to_numpy : bool, optional
        stochastic : bool, optional

        Returns
        -------
        probs : numpy.ndarray
            Shape: [batch, n_classes]
        """
        if self.variational: # variational autoencoder
            # Feed through the recognition network to get latents.
            z_mus = self.rec_model_1(features)
            z_log_stds = self.rec_model_2(features)
            if stochastic:
                # Make the variational posterior and sample.
                dist = Normal(z_mus, EPSILON + z_log_stds.exp())
                zs = dist.rsample() # [b,z]
            else:
                # Just take the mean.
                zs = z_mus
            # Project.
            zs = self.linear_layer(zs)
        else: # deterministic autoencoder
            # Feed through the recognition network to get latents.
            zs = self.recognition_model(features)
        # Get class predictions.
        logits = zs[:,:len(self.classes_)-1]
        ones = torch.ones(
                logits.shape[0],
                1,
                dtype=logits.dtype,
                device=logits.device,
        )
        logits = torch.cat([logits, ones], dim=1) + self.logit_bias
        probs = F.softmax(logits, dim=1) # [b, n_classes]
        if to_numpy:
            return probs.cpu().numpy()
        return probs


    @torch.no_grad()
    def predict(self, X):
        """
        Predict class labels for the features.

        Parameters
        ----------
        X : numpy.ndarray
            Features
            Shape: ``[batch, n_features]``

        Returns
        -------
        predictions : numpy.ndarray
            Shape: ``[batch]``
        """
        # Feed through model.
        X = torch.tensor(X, dtype=FLOAT).to(self.device)
        probs = self.predict_proba(X, to_numpy=False)
        predictions = torch.argmax(probs, dim=1)
        return self.classes_[predictions.cpu().numpy()]


    @torch.no_grad()
    def score(self, features, labels, groups):
        """
        Get a class-weighted accuracy.

        This is the objective we really care about, which doesn't contain the
        regularization in ``forward``.

        Parameters
        ----------
        features : numpy.ndarray
            Shape: [n_datapoints, n_features]
        labels : numpy.ndarray
            Shape: [n_datapoints]
        groups : None or numpy.ndarray
            Ignored

        Return
        ------
        weighted_acc : float
            Weighted accuracy
        """
        weights = get_weights(labels, groups, invalid_label=INVALID_LABEL)
        predictions = self.predict(features)
        scores = np.zeros(len(features))
        scores[predictions == labels] = 1.0
        scores = scores * weights
        weighted_acc = np.mean(scores)
        return weighted_acc


    @torch.no_grad()
    def get_factor(self, factor_num=0):
        """
        Get a linear factor.

        Parameters
        ----------
        feature_num : int
            Which factor to return. 0 <= `factor_num` < self.z_dim
        """
        assert isinstance(factor_num, int)
        assert factor_num >= 0 and factor_num < self.z_dim
        A = self.model.weight[:,factor_num]
        if self.nonnegative:
            A = F.softplus(A)
        return A.detach().cpu().numpy()


    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        super_params = super(FaSae, self).get_params(deep=deep)
        params = {
            'reg_strength': self.reg_strength,
            'z_dim': self.z_dim,
            'weight_reg': self.weight_reg,
            'nonnegative': self.nonnegative,
            'variational': self.variational,
            'kl_factor': self.kl_factor,
        }
        params = {**super_params, **params}
        return params


    def set_params(self, reg_strength=None, z_dim=None, weight_reg=None,
        nonnegative=None, variational=None, kl_factor=None, **kwargs):
        """Set the parameters of this estimator."""
        if reg_strength is not None:
            self.reg_strength = reg_strength
        if z_dim is not None:
            self.z_dim = z_dim
        if weight_reg is not None:
            self.weight_reg = weight_reg
        if nonnegative is not None:
            self.nonnegative = nonnegative
        if variational is not None:
            self.variational = variational
        if kl_factor is not None:
            self.kl_factor = kl_factor
        super(FaSae, self).set_params(**kwargs)
        return self



if __name__ == '__main__':
    pass


###
