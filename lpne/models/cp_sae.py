"""
CANDECOMP/PARAFAC supervised autoencoder with deterministic factors

"""
__date__ = "November 2021 - June 2022"


import numpy as np
from sklearn.utils.validation import check_is_fitted
import torch
from torch.distributions import Categorical, MultivariateNormal
import torch.nn.functional as F

from .base_model import BaseModel
from .. import INVALID_LABEL
from ..utils.utils import get_weights, squeeze_triangular_array


FLOAT = torch.float32
INT = torch.int64
FIT_ATTRIBUTES = ['classes_', 'groups_', 'iter_']
DEFAULT_GP_PARAMS = {
    'mean': 0.0,
    'ls': 0.2,
    'obs_noise_var': 1e-3,
    'reg': 0.1,
    'mode': 'ou',
}
"""Default frequency factor GP parameters"""



class CpSae(BaseModel):

    MODEL_NAME = 'CP SAE'


    def __init__(self, reg_strength=1.0, z_dim=32, gp_params=DEFAULT_GP_PARAMS,
        factor_reg=1e-1, **kwargs):
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
            Maps 'mean', 'ls', 'obs_noise_var', and 'reg' to values.
        factor_reg : float, optional
        """
        super(CpSae, self).__init__(**kwargs)
        self.reg_strength = float(reg_strength)
        self.z_dim = z_dim
        self.gp_params = {**DEFAULT_GP_PARAMS, **gp_params}
        self.factor_reg = float(factor_reg)

    
    @torch.no_grad()
    def _initialize(self, feature_shape):
        """
        Initialize the network parameters.

        Parameters
        ----------
        feature_shape : tuple
        """
        _, n_freqs, n_rois, _ = feature_shape
        n_groups = len(self.groups_)
        n_classes = len(self.classes_)
        # Make the recognition model.
        self.rec_model = torch.nn.Linear(n_freqs * n_rois**2, self.z_dim)
        # Set up the frequency factor GP.
        kernel = torch.arange(n_freqs).unsqueeze(0)
        kernel = torch.abs(kernel - torch.arange(n_freqs).unsqueeze(1))
        if self.gp_params['mode'] == 'se':
            kernel = 2**(-1/2) * torch.pow(kernel / self.gp_params['ls'], 2)
        elif self.gp_params['mode'] == 'ou':
            kernel = torch.abs(kernel / self.gp_params['ls'])
        else:
            raise NotImplementedError(self.gp_params['mode'])
        kernel = torch.exp(-kernel)
        kernel = kernel + self.gp_params['obs_noise_var'] * torch.eye(n_freqs)
        self.gp_dist = MultivariateNormal(
                self.gp_params['mean'] * torch.ones(n_freqs).to(self.device),
                covariance_matrix=kernel.to(self.device),
        )
        # Make the frequency factors.
        freq_factors = self.gp_dist.sample(sample_shape=(self.z_dim,)) # [z,f]
        self.freq_factors = torch.nn.Parameter(
            freq_factors.unsqueeze(0).expand(n_groups,-1,-1).clone(),
        ) # [1,z,f]
        # Make the ROI factors.
        roi_1_factors = -5 + torch.randn(1,self.z_dim,n_rois)
        self.roi_1_factors = torch.nn.Parameter(
           roi_1_factors.expand(n_groups,-1,-1).clone(),
        )
        roi_2_factors = -5 + torch.randn(1,self.z_dim,n_rois)
        self.roi_2_factors = torch.nn.Parameter(
            roi_2_factors.expand(n_groups,-1,-1).clone(),
        )
        self.logit_weights = torch.nn.Parameter(
                -5 * torch.ones(1,n_classes),
        )
        self.logit_biases = torch.nn.Parameter(torch.zeros(1,n_classes))
        super(CpSae, self)._initialize()


    def _get_H(self, flatten=True):
        """
        Get the factors.

        Parameters
        ----------
        flatten : bool, optional

        Returns
        -------
        H: torch.Tensor
            Shape: ``[g,b,frr]`` if ``flatten``, ``[g,b,f,r,r]`` otherwise
        """
        freq_f = F.softplus(self.freq_factors) # [g,z,f]
        freq_norm = torch.sqrt(torch.pow(freq_f,2).sum(dim=-1, keepdim=True))
        freq_f = freq_f / freq_norm
        roi_1_f = F.softplus(self.roi_1_factors) # [g,z,r]
        roi_1_norm = torch.sqrt(torch.pow(roi_1_f,2).sum(dim=-1, keepdim=True))
        roi_1_f = roi_1_f / roi_1_norm
        roi_2_f = F.softplus(self.roi_2_factors) # [g,z,r]
        roi_2_norm = torch.sqrt(torch.pow(roi_2_f,2).sum(dim=-1, keepdim=True))
        roi_2_f = roi_2_f / roi_2_norm
        volume = torch.einsum(
                'gzf,gzr,gzs->gzfrs',
                freq_f,
                roi_1_f,
                roi_2_f,
        ) # [g,z,f,r,r]
        if flatten:
            volume = volume.view(volume.shape[0], volume.shape[1], -1)
        return volume


    @torch.no_grad()
    def _get_mean_projection(self):
        volume = self._get_H(flatten=False) # [g,z,f,r,r]
        return torch.mean(volume, dim=0) # [z,f,r,r]


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
            unlabeled_mask = torch.isinf(1/(labels - INVALID_LABEL))
            labels[unlabeled_mask] = 0

        # Get latents.
        flat_features = features.view(features.shape[0], -1) # [b,fr^2]
        rec_features = torch.zeros_like(flat_features)
        zs = F.softplus(self.rec_model(flat_features)) # [b,z]

        if groups is None:
            assert return_logits
            logits = zs[:,:self.n_classes] * F.softplus(self.logit_weights)
            logits = logits + self.logit_biases # [b,c]
            return logits

        # Get the reconstruction loss.
        zs = zs.unsqueeze(1) # [b,1,z]
        flat_features = flat_features.unsqueeze(1) # [b,1,fr^2]
        H = self._get_H(flatten=True) # [g,z,fr^2]
        group_oh = F.one_hot(groups, len(self.groups_)) # [b,g]
        group_oh = group_oh.unsqueeze(-1).unsqueeze(-1) # [b,g,1,1]
        H = (group_oh * H.unsqueeze(0)).sum(dim=1) # [b,g,z,fr^2] -> [b,z,fr^2]
        rec_features = (zs @ H).squeeze(1) # [b,1,z][b,z,fr^2] -> [b,fr^2]
        flat_features = flat_features.squeeze(1) # [b,fr^2]
        rec_loss = (flat_features-rec_features).abs().mean(dim=1) #[b]

        # Predict the labels and get weighted label log probabilities.
        zs = zs.squeeze(1) # [b,1,z] -> [b,z]
        logits = zs[:,:self.n_classes] * F.softplus(self.logit_weights)
        logits = logits + self.logit_biases # [b,c]
        if return_logits:
            return logits
        log_probs = Categorical(logits=logits).log_prob(labels) # [b]
        log_probs = weights * log_probs # [b]
        log_probs[unlabeled_mask] = 0.0 # disregard the unlabeled data

        # Calculate the GP loss.
        freq_f = F.softplus(self.freq_factors) # [g,z,f]
        freq_norm = torch.sqrt(torch.pow(freq_f,2).sum(dim=-1, keepdim=True))
        freq_f = freq_f / freq_norm
        gp_loss = -self.gp_dist.log_prob(freq_f).sum()
        gp_loss = self.gp_params['reg'] * gp_loss

        # Combine all the terms into a composite loss.
        label_loss = -torch.mean(log_probs) # []
        rec_loss = self.reg_strength * torch.mean(rec_loss) # []
        factor_loss = self.factor_reg * self._get_factor_loss() # []
        loss = label_loss + rec_loss + factor_loss + gp_loss
        return loss


    def _get_factor_loss(self, eps=1e-7):
        f_mean = torch.mean(self.freq_factors, dim=0, keepdim=True)
        f_norm = torch.linalg.norm(f_mean, dim=2, keepdim=True)
        f_loss = (self.freq_factors - f_mean) / (f_norm + eps)
        f_loss = torch.pow(f_loss, 2).sum()
        roi_1_mean = torch.mean(self.roi_1_factors, dim=0, keepdim=True)
        roi_1_norm = torch.linalg.norm(roi_1_mean, dim=2, keepdim=True)
        roi_1_loss = (self.roi_1_factors - roi_1_mean) / (roi_1_norm + eps)
        roi_1_loss = torch.pow(roi_1_loss, 2).sum()
        roi_2_mean = torch.mean(self.roi_2_factors, dim=0, keepdim=True)
        roi_2_norm = torch.linalg.norm(roi_2_mean, dim=2, keepdim=True)
        roi_2_loss = (self.roi_2_factors - roi_2_mean) / (roi_2_norm + eps)
        roi_2_loss = torch.pow(roi_2_loss, 2).sum()
        return f_loss + roi_1_loss + roi_2_loss


    @torch.no_grad()
    def predict_proba(self, features, groups=None, to_numpy=True,
        return_logits=False):
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
        return_logits : bool, optional
            Return unnormalized logits instead of probabilities.

        Returns
        -------
        probs : numpy.ndarray
            Shape: ``[batch, n_classes]``
        """
        check_is_fitted(self, attributes=FIT_ATTRIBUTES)
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
            assert len(setdiff) == 0, f"Found unexpected groups: {setdiff}" \
                    f"Passed to predict: {temp_groups}" \
                    f"Passed to fit: {self.groups_}" 
            new_groups = np.zeros_like(groups)
            group_list = self.groups_.tolist()
            for temp_group in temp_groups:
                new_groups[groups == temp_group] = group_list.index(temp_group)
            groups = new_groups
            # To PyTorch Tensors.
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=FLOAT)
            groups = torch.tensor(groups, dtype=INT).to(self.device)
        logits = []
        i = 0
        while i <= len(features):
            batch_f = features[i:i+self.batch_size].to(self.device)
            batch_g = None if groups is None else groups[i:i+self.batch_size]
            batch_logit = self(batch_f, None, batch_g, None, return_logits=True)
            logits.append(batch_logit)
            i += self.batch_size
        logits = torch.cat(logits, dim=0)
        if return_logits:
            to_return = logits
        else:
            to_return = F.softmax(logits, dim=1) # [b, n_classes]
        if to_numpy:
            return to_return.detach().cpu().numpy()
        return to_return


    @torch.no_grad()
    def predict(self, features, groups=None):
        """
        Predict class labels for the features.

        Parameters
        ----------
        features : numpy.ndarray
            Shape: ``[b,f,r,r]``
        groups : None or numpy.ndarray, optional
            Shape: ``[b]``

        Returns
        -------
        predictions : numpy.ndarray
            Shape: ``[b]``
        """
        # Checks
        assert features.ndim == 4
        assert features.shape[2] == features.shape[3]
        check_is_fitted(self, attributes=FIT_ATTRIBUTES)
        # Feed through model.
        probs = self.predict_proba(features, groups, to_numpy=True)
        predictions = np.argmax(probs, axis=1)
        return self.classes_[predictions]


    @torch.no_grad()
    def score(self, features, labels, groups=None):
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

        Return
        ------
        weighted_accuracy : float
        """
        check_is_fitted(self, attributes=FIT_ATTRIBUTES)
        weights = get_weights(labels, groups, invalid_label=INVALID_LABEL)
        predictions = self.predict(features, groups)
        scores = np.zeros(len(features))
        scores[predictions == labels] = 1.0
        scores = scores * weights
        weighted_accuracy = np.mean(scores)
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
        check_is_fitted(self, attributes=FIT_ATTRIBUTES)
        assert isinstance(factor_num, int)
        assert factor_num >= 0 and factor_num < self.z_dim
        volume = self._get_mean_projection()[factor_num]  # [f,r,r]
        volume = volume.detach().cpu().numpy() # [f,r,r]
        volume = squeeze_triangular_array(volume, dims=(1,2)) # [f,r(r+1)/2]
        return volume.T # [r(r+1)/2,f]


    @torch.no_grad()
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        super_params = super(CpSae, self).get_params(deep=deep)
        params = {
            'reg_strength': self.reg_strength,
            'z_dim': self.z_dim,
            'factor_reg': self.factor_reg,
            'gp_params': self.gp_params,
        }
        params = {**super_params, **params}
        return params


    @torch.no_grad()
    def set_params(self, reg_strength=None, z_dim=None, factor_reg=None,
        gp_params=None, **kwargs):
        """Set the parameters of this estimator."""
        if reg_strength is not None:
            self.reg_strength = reg_strength
        if z_dim is not None:
            self.z_dim = z_dim
        if factor_reg is not None:
            self.factor_reg = factor_reg
        if gp_params is not None:
            self.gp_params = gp_params
        super(CpSae, self).set_params(**kwargs)
        return self



if __name__ == '__main__':
    pass



###