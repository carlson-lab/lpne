"""
CANDECOMP/PARAFAC supervised autoencoder

"""
__date__ = "November 2021"


import numpy as np
import os
from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import torch
from torch.distributions import Categorical, Normal, kl_divergence
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import warnings

from ..utils.utils import get_weights, squeeze_triangular_array

# https://stackoverflow.com/questions/53014306/
if float(torch.__version__[:3]) >= 1.9:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


FLOAT = torch.float32
INT = torch.int64
MAX_LABEL = 1000
EPSILON = 1e-6
FIT_ATTRIBUTES = ['classes_', 'groups_']



class CpSae(torch.nn.Module):

    def __init__(self, reg_strength=1.0, z_dim=32, group_embed_dim=2,
        weight_reg=0.0, kl_factor=1.0, n_iter=10000, lr=1e-3, batch_size=256,
        beta=0.5, device='auto'):
        """
        A supervised autoencoder with nonnegative and variational options.

        Parameters
        ----------
        reg_strength : float, optional
            This controls how much the classifier is regularized. This should
            be positive, and larger values indicate more regularization.
        z_dim : int, optional
            Latent dimension/number of networks.
        weight_reg : float, optional
            Model L2 weight regularization.
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
        super(CpSae, self).__init__()
        # Set parameters.
        self.reg_strength = float(reg_strength)
        self.z_dim = z_dim
        self.group_embed_dim = group_embed_dim
        self.weight_reg = float(weight_reg)
        self.kl_factor = float(kl_factor)
        self.n_iter = n_iter
        self.lr = float(lr)
        self.batch_size = batch_size
        self.beta = float(beta)
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.classes_ = None


    def _initialize(self, n_freqs, n_rois):
        """

        """
        n_groups = len(self.groups_)
        n_classes = len(self.classes_)
        n_features = n_freqs * n_rois**2
        self.group_embed = torch.nn.Parameter(
                torch.randn(n_groups, self.group_embed_dim,
        ))
        self.rec_model_1 = torch.nn.Linear(
                n_features + self.group_embed_dim,
                self.z_dim,
        )
        self.rec_model_2 = torch.nn.Linear(
                n_features + self.group_embed_dim,
                self.z_dim,
        )
        self.freq_factors = torch.nn.Parameter(
                torch.randn(n_groups, self.z_dim, n_freqs),
        )
        self.roi_1_factors = torch.nn.Parameter(
                torch.randn(n_groups, self.z_dim, n_rois),
        )
        self.roi_2_factors = torch.nn.Parameter(
                torch.randn(n_groups, self.z_dim, n_rois),
        )
        self.linear_layer = torch.nn.Linear(self.z_dim, self.z_dim)
        prior_mean = torch.zeros(self.z_dim).to(self.device)
        prior_std = torch.ones(self.z_dim).to(self.device)
        self.prior = Normal(prior_mean, prior_std)
        self.logit_bias = torch.nn.Parameter(torch.zeros(1,n_classes))
        self.to(self.device)



    def fit(self, features, labels, groups, print_freq=100):
        """

        Parameters
        ----------
        features : [b,f,r,r]
        labels :
        groups :
        weights :
        print_freq :
        """
        # Check arguments.
        assert features.ndim == 4
        assert features.shape[2] == features.shape[3]
        assert labels.ndim == 1
        assert groups.ndim == 1
        assert len(features) == len(labels) and len(labels) == len(groups)
        # Initialize.
        weights = get_weights(labels, groups)
        self.classes_, labels = np.unique(labels, return_inverse=True)
        self.groups_, groups = np.unique(groups, return_inverse=True)
        self._initialize(features.shape[1], features.shape[2])
        # NumPy arrays to PyTorch tensors.
        features = torch.tensor(features, dtype=FLOAT).to(self.device)
        labels = torch.tensor(labels, dtype=INT).to(self.device)
        groups = torch.tensor(groups, dtype=INT).to(self.device)
        weights = torch.tensor(weights, dtype=FLOAT).to(self.device)
        sampler_weights = torch.pow(weights, 1.0 - self.beta)
        weights = torch.pow(weights, self.beta)
        # Make some loaders and an optimizer.
        dset = TensorDataset(features, labels, groups, weights)
        sampler = WeightedRandomSampler(
                sampler_weights,
                num_samples=self.batch_size,
                replacement=True,
        )
        loader = DataLoader(
                dset,
                sampler=sampler,
                batch_size=self.batch_size,
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # Train.
        for epoch in range(1,self.n_iter+1):
            epoch_loss = 0.0
            for batch in loader:
                self.zero_grad()
                loss = self(*batch)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            if print_freq is not None and epoch % print_freq == 0:
                print(f"iter {epoch:04d}, loss: {loss:3f}")
        return self


    def _project(self, zs, groups):
        """
        Return factor loss too!

        Parameters
        ----------
        zs : [b,z]
        groups : [b]

        """
        freq_f = F.softplus(self.freq_factors) # [g,z,f]
        roi_1_f = F.softplus(self.roi_1_factors) # [g,z,r]
        roi_2_f = F.softplus(self.roi_2_factors) # [g,z,r]
        freq_loss = torch.var(freq_f, dim=0).mean(dim=1).sum(dim=0)
        roi_loss = torch.var(roi_1_f, dim=0) + torch.var(roi_2_f, dim=0)
        roi_loss = roi_loss.mean(dim=1).sum(dim=0)
        factor_loss = torch.pow(freq_f, 2)
        volume = torch.einsum(
                'bz,bzf,bzr,bzs->bfrs',
                F.softplus(zs),
                freq_f[groups],
                roi_1_f[groups],
                roi_2_f[groups],
        )
        return volume, freq_loss, roi_loss


    @torch.no_grad()
    def _get_mean_projection(self):
        freq_f = F.softplus(self.freq_factors).mean(dim=0) # [z,f]
        roi_1_f = F.softplus(self.roi_1_factors).mean(dim=0) # [z,r]
        roi_2_f = F.softplus(self.roi_2_factors).mean(dim=0) # [z,r]
        volume = torch.einsum(
                'zf,zr,zs->zfrs',
                freq_f,
                roi_1_f,
                roi_2_f,
        )
        return volume


    def forward(self, features, labels, groups, weights):
        """

        Parameters
        ----------
        features : [b,f,r,r]
        """
        # Augment features with group embeddings.
        flat_features = features.view(features.shape[0], -1)
        aug_features = torch.cat(
                [flat_features, self.group_embed[groups]],
                 dim=1,
        )
        # Feed through the recognition network to get latents.
        z_mus = self.rec_model_1(aug_features)
        z_log_stds = self.rec_model_2(aug_features)
        # Make the variational posterior and get a KL from the prior.
        dist = Normal(z_mus, EPSILON + z_log_stds.exp())
        kld = kl_divergence(dist, self.prior).sum(dim=1) # [b]
        # Sample.
        zs = dist.rsample() # [b,z]
        # Project.
        zs = self.linear_layer(zs)
        features_rec, freq_loss, roi_loss = self._project(zs, groups)
        flat_rec = features_rec.view(features.shape[0], -1)
        # Calculate a reconstruction loss.
        rec_loss = torch.mean((flat_features - flat_rec).pow(2), dim=1) # [b]
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
        log_probs = weights * log_probs
        # Regularize the model weights.
        # l2_loss = self.weight_reg * torch.norm(A)
        # TEMP
        l2_loss = freq_loss + roi_loss
        # Combine all the terms into a composite loss.
        loss = rec_loss - log_probs
        loss = loss + self.kl_factor * kld
        loss = torch.mean(loss) + l2_loss
        return loss


    @torch.no_grad()
    def predict_proba(self, features, groups, to_numpy=True, stochastic=False):
        """
        Probability estimates.

        Note
        ----
        * This should be consistent with `self.forward`.

        Parameters
        ----------
        features : numpy.ndarray
        groups :
        to_numpy : bool, optional
        stochastic : bool, optional

        Returns
        -------
        probs : numpy.ndarray
            Shape: [batch, n_classes]
        """
        # Augment features with group embeddings.
        flat_features = features.view(features.shape[0], -1)
        aug_features = torch.cat(
                [flat_features, self.group_embed[groups]],
                 dim=1,
        )
        # Feed through the recognition network to get latents.
        zs = self.rec_model_1(aug_features)
        if stochastic:
            z_log_stds = self.rec_model_2(aug_features)
            dist = Normal(zs, EPSILON + z_log_stds.exp())
            zs = dist.rsample() # [b,z]
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
    def predict(self, features, groups):
        """
        Predict class labels for the features.

        Parameters
        ----------
        features : numpy.ndarray
            Shape: ?
        groups : numpy.ndarray

        Returns
        -------
        predictions : numpy.ndarray
            Shape: [batch]
        """
        # Checks
        assert features.ndim == 4
        assert features.shape[2] == features.shape[3]
        check_is_fitted(self, attributes=FIT_ATTRIBUTES)
        # Feed through model.
        features = torch.tensor(features, dtype=FLOAT).to(self.device)
        groups = torch.tensor(groups, dtype=INT).to(self.device)
        probs = self.predict_proba(features, groups, to_numpy=False)
        predictions = torch.argmax(probs, dim=1)
        return self.classes_[predictions.cpu().numpy()]


    @torch.no_grad()
    def score(self, features, labels, groups):
        """
        Get a class weighted accuracy.

        This is the objective we really care about, which doesn't contain the
        regularization in FA's `forward` method.

        Parameters
        ----------
        features : numpy.ndarray
            Shape: [n,]
        labels : numpy.ndarray
            Shape: [n]
        groups : None or numpy.ndarray
            Shape: [n_datapoints]

        Return
        ------
        weighted_acc : float
        """
        # Derive groups, labels, and weights from labels.
        weights = get_weights(labels, groups)
        predictions = self.predict(features, groups)
        scores = np.zeros(len(features))
        scores[predictions == labels] = 1.0
        scores = scores * weights
        weighted_acc = np.mean(scores)
        return weighted_acc


    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = {
            'reg_strength': self.reg_strength,
            'z_dim': self.z_dim,
            'weight_reg': self.weight_reg,
            'kl_factor': self.kl_factor,
            'n_iter': self.n_iter,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'beta': self.beta,
            'device': self.device,
            'classes_': self.classes_,
            'groups_': self.groups_,
        }
        if deep:
            params['model_state_dict'] = self.state_dict()
        return params


    def set_params(self, reg_strength=None, z_dim=None, weight_reg=None,
        kl_factor=None, n_iter=None, lr=None, batch_size=None, beta=None,
        device=None, classes_=None, groups_=None, model_state_dict=None):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        ...
        """
        if reg_strength is not None:
            self.reg_strength = reg_strength
        if z_dim is not None:
            self.z_dim = z_dim
        if weight_reg is not None:
            self.weight_reg = weight_reg
        if kl_factor is not None:
            self.kl_factor = kl_factor
        if n_iter is not None:
            self.n_iter = n_iter
        if lr is not None:
            self.lr = lr
        if batch_size is not None:
            self.batch_size = batch_size
        if beta is not None:
            self.beta = beta
        if device is not None:
            self.device = device
        if classes_ is not None:
            self.classes_ = classes_
        if groups_ is not None:
            self.groups_ = groups_
        if model_state_dict is not None:
            # n_freqs, n_rois
            assert 'freq_factors' in model_state_dict, \
                    f"'freq_factors' not in {list(model_state_dict.keys())}"
            n_freqs = model_state_dict['freq_factors'].shape[-1]
            assert 'roi_1_factors' in model_state_dict, \
                    f"'roi_1_factors' not in {list(model_state_dict.keys())}"
            n_rois = model_state_dict['roi_1_factors'].shape[-1]
            self._initialize(n_freqs, n_rois)
            self.load_state_dict(model_state_dict)
        return self


    def save_state(self, fn):
        """Save parameters for this estimator."""
        np.save(fn, self.get_params(deep=True))


    def load_state(self, fn):
        """Load and set the parameters for this estimator."""
        self.set_params(**np.load(fn, allow_pickle=True).item())


    @torch.no_grad()
    def get_factor(self, factor_num=0):
        """
        Get a linear factor.

        Parameters
        ----------
        feature_num : int
            Which factor to return. 0 <= `factor_num` < self.z_dim
            Shape: [r(r+1)/2,f]
        """
        check_is_fitted(self, attributes=FIT_ATTRIBUTES)
        assert isinstance(factor_num, int)
        assert factor_num >= 0 and factor_num < self.z_dim
        volume = self._get_mean_projection()[factor_num]  # [f,r,r]
        volume = volume.detach().cpu().numpy()
        volume = squeeze_triangular_array(volume, dims=(1,2))
        return volume.T



if __name__ == '__main__':
    pass



###
