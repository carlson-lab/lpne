"""
Factor Analysis-regularized logistic regression.

"""
__date__ = "June 2021 - July 2022"


import numpy as np
from sklearn.utils.validation import check_is_fitted
import torch
from torch.distributions import Categorical, Normal, MultivariateNormal, \
    kl_divergence
import torch.nn.functional as F


from .base_model import BaseModel
from .. import INVALID_LABEL
from ..utils.utils import get_weights, squeeze_triangular_array


FLOAT = torch.float32
EPSILON = 1e-6
DEFAULT_GP_PARAMS = {
    'mean': 0.0,
    'ls': 0.3,
    'obs_noise_var': 1e-3,
    'reg': 0.1,
    'kernel': 'se',
}
"""Default frequency factor GP parameters"""



class FaSae(BaseModel):
    """
    A supervised autoencoder with nonnegative and variational options.

    Parameters
    ----------
    reg_strength : float, optional
        This controls how much the classifier is regularized. This should
        be positive, and larger values indicate more regularization.
    z_dim : int, optional
        Latent dimension/number of networks.
    nonnegative : bool, optional
        Use nonnegative factorization.
    variational : bool, optional
        Whether a variational autoencoder is used.
    kl_factor : float, optional
        How much to weight the KL divergence term in the variational
        autoencoder (VAE). The standard setting is `1.0`. This is a
        distinct regularization parameter from `reg_strength` that can be
        independently set. This parameter is only used if `variational` is
        `True`.
    encoder_type : str, optional
        One of ``'linear'``, ``'lstsq'``, ``'pinv'``, or ``'nnlstsq'``. The
        least squares or pseudoinverse encoders are "encoderless" in the
        sense that they only rely on the decoder parameters and Gaussian
        priors on the latents to map data to latents. These options are
        only supported when ``variational`` is ``False`` and
        ``nonnegative`` is ``True``. Depending on which device you are
        using and the model dimensions, one of ``'lstsq'`` and ``'pinv'``
        may be substantially faster than the other. However, ``'lstsq'``
        will be more numerically stable. Defaults to ``'linear'``.
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
    rec_loss_type : str, optional
        One of ``'lad'`` for least absolute deviations or ``'ls'`` for
        least squares. Defaults to ``'lad'``.
    irls_iter : int, opional
        Number of iterations to run iteratively reweighted least squares.
        Defaults to ``1``.
    """

    MODEL_NAME = 'FA SAE'
    FIT_ATTRIBUTES = ['classes_', 'iter_', 'n_freqs_', 'n_rois_']


    def __init__(self, reg_strength=1.0, z_dim=32, nonnegative=True,
        variational=False, kl_factor=1.0, encoder_type='linear',
        gp_params=DEFAULT_GP_PARAMS, rec_loss_type='lad', irls_iter=1,
        **kwargs):
        super(FaSae, self).__init__(**kwargs)
        assert kl_factor >= 0.0, f"{kl_factor} < 0"
        # Set parameters.
        assert isinstance(reg_strength, (int, float))
        assert reg_strength >= 0.0
        self.reg_strength = float(reg_strength)
        assert isinstance(z_dim, int)
        assert z_dim >= 1
        self.z_dim = z_dim
        assert isinstance(nonnegative, bool)
        self.nonnegative = nonnegative
        assert isinstance(variational, bool)
        self.variational = variational
        assert isinstance(kl_factor, (int, float))
        assert kl_factor >= 0.0
        self.kl_factor = float(kl_factor)  
        assert encoder_type in ['linear', 'lstsq', 'pinv', 'irls']
        self.encoder_type = encoder_type
        self.gp_params = {**DEFAULT_GP_PARAMS, **gp_params} 
        assert rec_loss_type in ['lad', 'ls']
        self.rec_loss_type = rec_loss_type
        assert isinstance(irls_iter, int)
        self.irls_iter = irls_iter
        self.classes_ = None


    def _initialize(self):
        """Initialize parameters of the networks before training."""
        _, self.n_freqs_, self.n_rois_, _ = self.features_shape_
        n_freqs = self.n_freqs_
        n_features = self.n_freqs_ * self.n_rois_**2
        # Check arguments.
        n_classes = len(self.classes_)
        assert n_classes <= self.z_dim, f"{n_classes} > {self.z_dim}"
        assert not (self.encoder_type == 'solve' and self.variational)
        assert not (self.encoder_type == 'solve' and not self.nonnegative)
        # Set up the frequency factor GP.
        kernel = torch.arange(n_freqs).unsqueeze(0)
        kernel = torch.abs(kernel - torch.arange(n_freqs).unsqueeze(1))
        if self.gp_params['kernel'] == 'se':
            kernel = 2**(-1/2) * torch.pow(kernel / self.gp_params['ls'], 2)
        elif self.gp_params['kernel'] == 'ou':
            kernel = torch.abs(kernel / self.gp_params['ls'])
        else:
            raise NotImplementedError(self.gp_params['kernel'])
        kernel = torch.exp(-kernel)
        kernel = kernel + self.gp_params['obs_noise_var'] * torch.eye(n_freqs)
        self.gp_dist = MultivariateNormal(
                self.gp_params['mean'] * torch.ones(n_freqs).to(self.device),
                covariance_matrix=kernel.to(self.device),
        )
        # Make the networks.
        self.recognition_model = torch.nn.Linear(n_features, self.z_dim)
        self.rec_model_1 = torch.nn.Linear(n_features, self.z_dim)
        self.rec_model_2 = torch.nn.Linear(n_features, self.z_dim)
        self.linear_layer = torch.nn.Linear(self.z_dim, self.z_dim)
        prior_mean = torch.zeros(self.z_dim).to(self.device)
        prior_std = torch.ones(self.z_dim).to(self.device)
        self.prior = Normal(prior_mean, prior_std)
        self.model = torch.nn.Parameter(
                torch.nn.Linear(self.z_dim, n_features).weight.clone(),
        ) # [x,z]
        self.factor_reg = torch.nn.Parameter(
            torch.zeros(self.z_dim),
        ) # [z]
        self.factor_reg_target = torch.nn.Parameter(
            torch.ones(self.z_dim),
        ) # [z]
        self.logit_weights = torch.nn.Parameter(
                -5 * torch.ones(1,n_classes),
        )
        self.logit_biases = torch.nn.Parameter(torch.zeros(1,n_classes))
        super(FaSae, self)._initialize()


    def forward(self, features, labels, groups, weights, return_logits=False,
        stochastic=True):
        """
        Calculate a loss for the features and labels.

        Parameters
        ----------
        features : torch.Tensor
            Shape: [batch,f,r,r]
        labels : torch.Tensor
            Shape: [batch]
        groups : None or torch.Tensor
            Ignored
        weights : torch.Tensor
            Shape: [batch]
        return_logits : bool, optional
            Return only the logits.
        stochastic : bool, optional
            Whether to sample from the approximate posterior

        Returns
        -------
        loss : torch.Tensor
            Returned if ``return_logits`` is ``False``.
            Shape: []
        logits : torch.Tensor
            Returned only if ``return_logits``.
            Shape: [b,c]
        """
        if labels is not None:
                unlabeled_mask = torch.isinf(1/(labels - INVALID_LABEL))
                labels[unlabeled_mask] = 0
        
        # Get latents.
        features = features.view(len(features),-1) # [b,f,r,r] -> [b,x]
        zs, kld = self.get_latents(
                features,
                stochastic=stochastic,
                return_kld=True,
        ) # [b,z],[b]

        # Predict the labels.
        logits = zs[:,:self.n_classes] * F.softplus(self.logit_weights)
        logits = logits + self.logit_biases # [b,c]
        if return_logits:
            return logits
        log_probs = Categorical(logits=logits).log_prob(labels) # [b]
        log_probs = weights * log_probs # [b]
        log_probs[unlabeled_mask] = 0.0 # disregard the unlabeled data
        
        # Reconstruct the features.
        rec_features, A = self.project_latents(zs, return_factor=True) # [b,x]
        
        # Calculate a reconstruction loss.
        diff = features - rec_features # [b,x]
        if self.rec_loss_type == 'lad':
            rec_loss = torch.abs(diff).mean(dim=1) # [b]
        elif self.rec_loss_type == 'ls':
            rec_loss = 0.5 * torch.pow(diff,2).mean(dim=1) # [b]
        rec_loss = self.reg_strength * rec_loss # [b]

        # Calculate the GP loss.
        freq_f = A.view(self.n_freqs_, self.n_rois_, self.n_rois_, self.z_dim)
        freq_norm = torch.sqrt(torch.pow(freq_f,2).sum(dim=0, keepdim=True))
        freq_f = freq_f / freq_norm
        gp_loss = -self.gp_dist.log_prob(torch.swapaxes(freq_f,0,-1)).sum()
        gp_loss = self.gp_params['reg'] * gp_loss
        
        # Combine all the terms into a composite loss.
        loss = rec_loss - log_probs
        if self.variational:
            loss = loss + self.kl_factor * kld
        loss = loss.sum() + gp_loss
        return loss


    def get_latents(self, features, stochastic=True, return_kld=False,
        reg=1e-3):
        """
        Get the latents corresponding to the given features.

        Parameters
        ----------
        features : torch.Tensor
            Shape: [b,x]
        stochastic : bool, optional
        return_kld : bool, optional
        reg : float, optional
            Regularization for IRLS

        Returns
        -------
        latents : torch.Tensor
            Shape: [b,z]
        kld : torch.Tensor
            KL divergence. Returned if ``return_kld``.
            Shape: [b]
        """
        assert features.ndim == 2, f"len({features.shape}) != 2"
        if self.variational:
            # Variational autoencoder
            # Feed through the recognition network to get latents.
            z_mus = self.rec_model_1(features)
            z_log_stds = self.rec_model_2(features)
            # Make the variational posterior and get a KL from the prior.
            dist = Normal(z_mus, EPSILON + z_log_stds.exp())
            kld = kl_divergence(dist, self.prior).sum(dim=1) # [b]
            # Sample.
            if stochastic:
                latents = dist.rsample() # [b,z]
            else:
                latents = z_mus # [b,z]
            # Project.
            latents = self.linear_layer(latents)
        else:
            # Deterministic autoencoder
            kld = None
            if self.encoder_type == 'linear':
                # Feed through the recognition network to get latents.
                latents = self.recognition_model(features) # [b,z]
                latents = F.softplus(latents)
            elif self.encoder_type in ['lstsq', 'pinv', 'irls']:
                # Solve the least squares problem and rectify to get latents.
                A = F.softplus(self.model) # [x,z]
                A_norm = torch.sqrt(torch.pow(A,2).sum(dim=0, keepdim=True))
                A = A / A_norm
                factor_reg = torch.diag_embed(F.softplus(self.factor_reg))
                A = torch.cat([A, factor_reg], dim=0) # [x+z,z]
                pad = self.factor_reg_target.unsqueeze(0) # [1,z]
                pad = pad.expand(len(features),-1) # [b,z]
                target = torch.cat([features, pad], dim=1) # [b,x+z]
                if self.encoder_type == 'lstsq':
                    # https://github.com/pytorch/pytorch/issues/27036
                    latents = torch.linalg.lstsq(
                            A.unsqueeze(0),
                            target.unsqueeze(-1),
                    ).solution.squeeze(-1) # [b,z]
                    latents = torch.clamp(latents, min=0.0)
                elif self.encoder_type == 'pinv':
                    # https://github.com/pytorch/pytorch/issues/41306
                    # This may be much faster on GPU than linalg.lstsq.
                    # Hopefully this will change in future releases.
                    latents = torch.linalg.pinv(A.unsqueeze(0)) \
                               @ target.unsqueeze(-1)
                    latents = torch.clamp(latents.squeeze(-1), min=0.0) # [b,z]
                elif self.encoder_type == 'irls':
                    # features: [b,x]
                    # A : [x,z]
                    latents = torch.linalg.solve(
                        A.t() @ A,
                        A.t() @ target.unsqueeze(-1),
                    ) # [b,z,1]
                    for _ in range(self.irls_iter):
                        diffs = target - latents.squeeze(-1) @ A.t()
                        weights = 1.0/torch.clamp(torch.abs(diffs), reg, None)
                        inner = torch.einsum(
                            'zx,bx,xw->bzw',
                            A.t(),
                            weights,
                            A,
                        ) # [b,z,z]
                        prod = torch.einsum(
                            'zx,bx->bz',
                            A.t(),
                            weights * target,
                        ) # [b,z]
                        latents = torch.linalg.solve(
                            inner,
                            prod.unsqueeze(-1),
                        ) # [b,z,1]
                    latents = torch.clamp(latents.squeeze(-1), min=0.0) # [b,z]
            else:
                raise NotImplementedError(self.encoder_type)
        if return_kld:
            return latents, kld
        return latents


    def project_latents(self, latents, return_factor=False):
        """
        Feed latents through the model to get observations.
        
        Parameters
        ----------
        latents : torch.Tensor
            Shape: [b,z]
        return_factor : bool, optional
            Whether to return the linear model factor

        Returns
        -------
        x_pred : torch.Tensor
            Shape: [b,x]
        A : torch.Tensor
            Linear factor. Returned if ``return_factor``.
            Shape: [x,z]
        """
        if self.nonnegative:
            A = F.softplus(self.model) # [x,z]
        else:
            A = self.model # [x,z]
        A_norm = torch.sqrt(torch.pow(A,2).sum(dim=0, keepdim=True))
        A = A / A_norm
        x_pred = A.unsqueeze(0) @ F.softplus(latents).unsqueeze(-1)
        x_pred = x_pred.squeeze(-1) # [b,x]
        if return_factor:
            return x_pred, A
        return x_pred


    @torch.no_grad()
    def predict_proba(self, features, to_numpy=True, stochastic=False,
        **kwargs):
        """
        Probability estimates.

        Note
        ----
        * This should be consistent with ``self.forward``.

        Parameters
        ----------
        features : torch.Tensor
            The features to make predictions for
            Shape: ``[n,f,r,r]``
        to_numpy : bool, optional
            Whether to return a NumPy array instead of a Pytorch tensor.
            Defaults to ``True``.
        stochastic : bool, optional
            Use stochastic encoding if the model is a VAE. Defaults to
            ``False``.

        Returns
        -------
        probs : numpy.ndarray or torch.Tensor
            The predicted class probabilities
            Shape: ``[n,c]``
        """
        check_is_fitted(self, attributes=self.FIT_ATTRIBUTES)
        logits = []
        i = 0
        while i <= len(features):
            batch_f = features[i:i+self.batch_size].to(self.device)
            batch_logit = self(
                batch_f,
                None,
                None,
                None,
                return_logits=True,
                stochastic=stochastic,
            )
            logits.append(batch_logit)
            i += self.batch_size
        logits = torch.cat(logits, dim=0)
        probs = F.softmax(logits, dim=1) # [b,c]
        if to_numpy:
            return probs.cpu().numpy()
        return probs


    @torch.no_grad()
    def predict(self, X, *args, **kwargs):
        """
        Predict class labels for the features.

        Parameters
        ----------
        X : numpy.ndarray
            The features to make predictions for
            Shape: ``[b,x]``

        Returns
        -------
        predictions : numpy.ndarray
            The predicted classes
            Shape: ``[b]``
        """
        check_is_fitted(self, attributes=self.FIT_ATTRIBUTES)
        # Feed through model.
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=FLOAT).to(self.device)
        probs = self.predict_proba(X, to_numpy=False)
        predictions = torch.argmax(probs, dim=1)
        return self.classes_[predictions.cpu().numpy()]


    @torch.no_grad()
    def score(self, features, labels, groups, *args, **kwargs):
        """
        Get a class-weighted accuracy.

        This is the objective we really care about, which doesn't contain the
        regularization in ``forward``.

        Parameters
        ----------
        features : numpy.ndarray
            Shape: ``[b,x]``
        labels : numpy.ndarray
            Shape: ``[b]``
        groups : None or numpy.ndarray
            Ignored

        Return
        ------
        weighted_acc : float
            Class-weighted accuracy
        """
        check_is_fitted(self, attributes=self.FIT_ATTRIBUTES)
        weights = get_weights(labels, groups, invalid_label=INVALID_LABEL)
        predictions = self.predict(features)
        scores = np.zeros(len(features))
        scores[predictions == labels] = 1.0
        scores = scores * weights
        weighted_acc = np.mean(scores[labels != INVALID_LABEL])
        return weighted_acc


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
        assert isinstance(factor_num, int)
        assert factor_num >= 0 and factor_num < self.z_dim
        A = self.model[:,factor_num]
        if self.nonnegative:
            A = F.softplus(A)
        A_norm = torch.sqrt(torch.pow(A,2).sum(dim=0, keepdim=True))
        A = A / A_norm
        A = A.detach().cpu().numpy()
        A = A.reshape(self.n_freqs_, self.n_rois_, self.n_rois_) # [f,r,r]
        A = squeeze_triangular_array(A, dims=(1,2)) # [f,r(r+1)/2]
        return A.T # [r(r+1)/2,f]


    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        super_params = super(FaSae, self).get_params(deep=deep)
        params = {
            'reg_strength': self.reg_strength,
            'z_dim': self.z_dim,
            'nonnegative': self.nonnegative,
            'variational': self.variational,
            'kl_factor': self.kl_factor,
            'encoder_type': self.encoder_type,
            'gp_params': self.gp_params,
            'rec_loss_type': self.rec_loss_type,
            'irls_iter': self.irls_iter,
        }
        params = {**super_params, **params}
        return params


    def set_params(self, reg_strength=None, z_dim=None, nonnegative=None,
        variational=None, kl_factor=None, encoder_type=None, gp_params=None,
        rec_loss_type=None, irls_iter=None, **kwargs):
        """Set the parameters of this estimator."""
        if reg_strength is not None:
            self.reg_strength = reg_strength
        if z_dim is not None:
            self.z_dim = z_dim
        if nonnegative is not None:
            self.nonnegative = nonnegative
        if variational is not None:
            self.variational = variational
        if kl_factor is not None:
            self.kl_factor = kl_factor
        if encoder_type is not None:
            self.encoder_type = encoder_type
        if gp_params is not None:
            self.gp_params = {**DEFAULT_GP_PARAMS, **gp_params}
        if rec_loss_type is not None:
            self.rec_loss_type = rec_loss_type
        if self.irls_iter is not None:
            self.irls_iter = irls_iter
        super(FaSae, self).set_params(**kwargs)
        return self



if __name__ == '__main__':
    pass


###