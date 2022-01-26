"""
CANDECOMP/PARAFAC supervised autoencoder with deterministic factors.

"""
__date__ = "November 2021 - January 2022"


import numpy as np
import os
from sklearn.utils.validation import check_is_fitted
import torch
from torch.distributions import Categorical, Normal, kl_divergence
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    pass
import warnings

from lpne import __commit__ as LPNE_COMMIT
from lpne import __version__ as LPNE_VERSION

from ..utils.utils import get_weights, squeeze_triangular_array


# https://stackoverflow.com/questions/53014306/
if float(torch.__version__[:3]) >= 1.9:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


FLOAT = torch.float32
INT = torch.int64
EPSILON = 1e-5
INVALID_LABEL = -1
FIT_ATTRIBUTES = ['classes_', 'groups_', 'iter_']



class CpSae(torch.nn.Module):

    def __init__(self, reg_strength=1.0, z_dim=32, weight_reg=0.0, n_iter=10000,
        lr=1e-3, batch_size=256, beta=0.5, factor_reg=1e-1, log_dir=None,
        n_updates=0, device='auto'):
        """
        A supervised autoencoder using CP-style generative model.

        Parameters
        ----------
        reg_strength : float, optional
            This controls how much we weight the reconstruction loss. This
            should be positive, and larger values indicate more regularization.
        z_dim : int, optional
            Latent dimension/number of networks.
        weight_reg : float, optional
            Model L2 weight regularization.
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
        self.weight_reg = float(weight_reg)
        self.n_iter = n_iter
        self.lr = float(lr)
        self.batch_size = batch_size
        self.beta = float(beta)
        self.factor_reg = float(factor_reg)
        self.log_dir = log_dir
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.n_updates = n_updates
        self.classes_ = None


    def _initialize(self, n_freqs, n_rois, features=None):
        """
        Initialize the network parameters.

        Parameters
        ----------
        n_freqs : int
        n_rois : int
        """
        # Set up TensorBoard.
        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir=self.log_dir)
        self.n_groups = len(self.groups_)
        self.n_classes = len(self.classes_)
        n_features = n_freqs * n_rois**2
        self.rec_model = torch.nn.Linear(n_features, self.z_dim)
        with torch.no_grad():
            self.freq_factors = torch.nn.Parameter(
                    -5 + torch.randn(self.n_groups, self.z_dim, n_freqs),
            )
            self.roi_1_factors = torch.nn.Parameter(
                    -5 + torch.randn(self.n_groups, self.z_dim, n_rois),
            )
            self.roi_2_factors = torch.nn.Parameter(
                    -5 + torch.randn(self.n_groups, self.z_dim, n_rois),
            )
            self.logit_weights = torch.nn.Parameter(
                    -5 * torch.ones(1,self.n_classes),
            )
        self.logit_biases = torch.nn.Parameter(torch.zeros(1,self.n_classes))
        self.to(self.device)


    def _get_H(self, flatten=True):
        """

        Returns
        -------
        H: [g,b,frr] if flatten
           [g,b,f,r,r] otherwise
        """
        freq_f = F.softplus(self.freq_factors) # [g,z,f]
        roi_1_f = F.softplus(self.roi_1_factors) # [g,z,r]
        roi_2_f = F.softplus(self.roi_2_factors) # [g,z,r]
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


    def fit(self, features, labels, groups, print_freq=100, test_freq=1):
        """
        Fit the model to data.

        Parameters
        ----------
        features : numpy.ndarray
            Shape: [b,f,r,r]
        labels : numpy.ndarray
            Shape: [b]
        groups : None or numpy.ndarray
            Shape: [b]
        print_freq : int, optional
        test_freq : int, optional
        """
        # Check arguments.
        assert features.ndim == 4
        assert features.shape[2] == features.shape[3]
        assert labels.ndim == 1
        assert groups.ndim == 1
        assert len(features) == len(labels) and len(labels) == len(groups)
        # Initialize.
        orig_labels = np.copy(labels)
        orig_groups = np.copy(groups)
        weights = get_weights(labels, groups) # NOTE: here with invalid labels?
        idx = np.argwhere(labels == INVALID_LABEL).flatten()
        idx_comp = np.argwhere(labels != INVALID_LABEL).flatten()
        temp_label = np.unique(labels[labels != INVALID_LABEL])[0]
        labels[idx] = temp_label # Mask the labels temporarily.
        self.classes_, labels = np.unique(labels, return_inverse=True)
        labels[idx] = INVALID_LABEL # Unmask the labels.
        assert len(self.classes_) > 1
        self.groups_, groups = np.unique(groups, return_inverse=True)
        assert len(self.groups_) > 1
        self.iter_ = 1
        self._initialize(features.shape[1], features.shape[2], features)
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
                num_samples=len(sampler_weights),
                replacement=True,
        )
        loader = DataLoader(
                dset,
                sampler=sampler,
                batch_size=self.batch_size,
        )
        optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_reg,
        )
        # Train.
        while self.iter_ <= self.n_iter:
            i_loss, i_label_loss, i_rec_loss, i_f_loss = [0.0]*4
            for batch in loader:
                self.zero_grad()
                loss, label_loss, rec_loss, f_loss = self(*batch)
                i_loss += loss.item()
                i_label_loss += label_loss.item()
                i_rec_loss += rec_loss.item()
                i_f_loss += f_loss.item()
                loss.backward()
                optimizer.step()
            if self.log_dir is not None:
                self.writer.add_scalar('train loss', i_loss, self.iter_)
                self.writer.add_scalar('label loss', i_label_loss, self.iter_)
                self.writer.add_scalar('rec loss', i_rec_loss, self.iter_)
                self.writer.add_scalar('f loss', i_f_loss, self.iter_)
            if print_freq is not None and self.iter_ % print_freq == 0:
                print(f"iter {self.iter_:04d}, loss: {i_loss:3f}")
            if test_freq is not None and self.iter_ % test_freq == 0:
                weighted_acc = self.score(
                        features[idx_comp],
                        orig_labels[idx_comp],
                        orig_groups[idx_comp],
                )
                if self.log_dir is not None:
                    self.writer.add_scalar(
                            'weighted accuracy',
                            weighted_acc,
                            self.iter_,
                    )
            self.iter_ += 1
        return self


    def forward(self, features, labels, groups, weights, return_logits=False):
        """
        Calculate a loss.

        Parameters
        ----------
        features : [b,f,r,r]
        labels : [b]
        groups : None or [b]
        weights : [b]

        Returns
        -------
        loss : torch.Tensor
            Shape: []
        """
        if labels is not None:
            nan_mask = torch.isinf(1/(labels - INVALID_LABEL))
            labels[nan_mask] = 0

        # Get latents.
        flat_features = features.view(features.shape[0], -1) # [b,fr^2]
        rec_features = torch.zeros_like(flat_features)
        zs = F.softplus(self.rec_model(flat_features)) # [b,z]

        if groups is None:
            assert self.n_updates == 0
            assert return_logits
            logits = zs[:,:self.n_classes] * F.softplus(self.logit_weights)
            logits = logits + self.logit_biases # [b,c]
            return logits

        # Update latents with multiplicative updates.
        zs = zs.unsqueeze(1) # [b,1,z]
        flat_features = flat_features.unsqueeze(1) # [b,1,fr^2]
        H = self._get_H() # [g,z,fr^2]
        group_oh = F.one_hot(groups, len(self.groups_)) # [b,g]
        group_oh = group_oh.unsqueeze(-1).unsqueeze(-1) # [b,g,1,1]
        H = (group_oh * H.unsqueeze(0)).sum(dim=1) # [b,g,z,fr^2] -> [b,z,fr^2]
        for i in range(self.n_updates):
            # [b,1,fr^2][b,fr^2,z] -> [b,1,z]
            numerator = flat_features @ H.transpose(-1,-2)
            # [b,1,z][b,z,fr^2][b,fr^2,z] -> [b,1,z]
            denominator = zs @ H @ H.transpose(-1,-2) # [b,1,z]
            zs = zs * numerator / denominator
        rec_features = (zs @ H).squeeze(1) # [b,1,z][b,z,fr^2] -> [b,fr^2]
        flat_features = flat_features.squeeze(1) # [b,fr^2]
        rec_loss = torch.mean(torch.abs(flat_features-rec_features), dim=1) #[b]

        # Predict the labels and get weighted label log probabilities.
        zs = zs.squeeze(1) # [b,1,z] -> [b,z]
        logits = zs[:,:self.n_classes] * F.softplus(self.logit_weights)
        logits = logits + self.logit_biases # [b,c]
        if return_logits:
            return logits
        log_probs = Categorical(logits=logits).log_prob(labels) # [b]
        log_probs = weights * log_probs # [b]
        log_probs[nan_mask] = 0.0 # [b]

        # Combine all the terms into a composite loss.
        label_loss = -torch.mean(log_probs) # []
        if torch.isnan(label_loss):
            quit("label_loss NaN")
        rec_loss = self.reg_strength * torch.mean(rec_loss) # []
        if torch.isnan(rec_loss):
            quit("rec_loss NaN")
        factor_loss = self.factor_reg * self._get_factor_loss() # []
        if torch.isnan(factor_loss):
            quit("factor_loss NaN")
        loss = label_loss + rec_loss + factor_loss
        return loss, label_loss, rec_loss, factor_loss


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
    def predict_proba(self, features, groups=None, to_numpy=True):
        """
        Probability estimates.

        Note
        ----
        * This should be consistent with `self.forward`.

        Parameters
        ----------
        features : numpy.ndarray or torch.Tensor
            Shape: [b,f,r,r]
        groups : None or numpy.ndarray
            Shape: [b]
        to_numpy : bool, optional
        stochastic : bool, optional

        Returns
        -------
        probs : numpy.ndarray
            Shape: [batch, n_classes]
        """
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=FLOAT)
        # Figure out group mapping.
        if groups is not None:
            if isinstance(groups, torch.Tensor):
                groups = groups.detach().cpu().numpy()
            # Figure out the group mapping.
            temp_groups = np.unique(groups)
            setdiff = np.setdiff1d(temp_groups, self.groups_, assume_unique=True)
            assert len(setdiff) == 0, f"Found unexpected groups: {setdiff}"
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
        probs = F.softmax(logits, dim=1) # [b, n_classes]
        if to_numpy:
            return probs.detach().cpu().numpy()
        return probs


    @torch.no_grad()
    def predict(self, features, groups=None):
        """
        Predict class labels for the features.

        Parameters
        ----------
        features : numpy.ndarray
            Shape: [b,f,r,r]
        groups : None or numpy.ndarray, optional
            Shape: [b]

        Returns
        -------
        predictions : numpy.ndarray
            Shape: [b]
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
            Shape: [b,f,r,r]
        labels : numpy.ndarray
            Shape: [b]
        groups : None or numpy.ndarray, optional
            Shape: [b]

        Return
        ------
        weighted_accuracy : float
        """
        # Derive groups, labels, and weights from labels.
        weights = get_weights(labels, groups)
        predictions = self.predict(features, groups)
        scores = np.zeros(len(features))
        scores[predictions == labels] = 1.0
        scores = scores * weights
        weighted_accuracy = np.mean(scores)
        return weighted_accuracy


    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = {
            'reg_strength': self.reg_strength,
            'z_dim': self.z_dim,
            'weight_reg': self.weight_reg,
            'n_iter': self.n_iter,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'beta': self.beta,
            'factor_reg': self.factor_reg,
            'n_updates': self.n_updates,
        }
        try:
            params['classes_'] = self.classes_
            params['groups_'] = self.groups_
            params['iter_'] = self.iter_
        except:
            pass
        if deep:
            temp = self.state_dict()
            for key in temp:
                temp[key] = temp[key].to('cpu')
            params['model_state_dict'] = temp
        return params


    def set_params(self, reg_strength=None, z_dim=None, weight_reg=None,
        n_iter=None, lr=None, batch_size=None, beta=None, factor_reg=None,
        n_updates=None, classes_=None, groups_=None, iter_=None,
        model_state_dict=None, **kwargs):
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
        if n_iter is not None:
            self.n_iter = n_iter
        if lr is not None:
            self.lr = lr
        if batch_size is not None:
            self.batch_size = batch_size
        if beta is not None:
            self.beta = beta
        if factor_reg is not None:
            self.factor_reg = factor_reg
        if n_updates is not None:
            self.n_updates = n_updates
        if classes_ is not None:
            self.classes_ = classes_
        if groups_ is not None:
            self.groups_ = groups_
        if iter_ is not None:
            self.iter_ = iter_
        if model_state_dict is not None:
            # n_freqs, n_rois
            assert 'freq_factors' in model_state_dict, \
                    f"'freq_factors' not in {list(model_state_dict.keys())}"
            n_freqs = model_state_dict['freq_factors'].shape[-1]
            assert 'roi_1_factors' in model_state_dict, \
                    f"'roi_1_factors' not in {list(model_state_dict.keys())}"
            n_rois = model_state_dict['roi_1_factors'].shape[-1]
            self._initialize(n_freqs, n_rois)
            for key in model_state_dict:
                model_state_dict[key] = model_state_dict[key].to(self.device)
            self.load_state_dict(model_state_dict)
        return self


    def save_state(self, fn):
        """Save parameters for this estimator."""
        params = self.get_params(deep=True)
        params['__commit__'] = LPNE_COMMIT
        params['__version__'] = LPNE_VERSION
        np.save(fn, params)


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

        Returns
        -------
        factor : numpy.ndarray
            Shape: [r(r+1)/2,f]
        """
        check_is_fitted(self, attributes=FIT_ATTRIBUTES)
        assert isinstance(factor_num, int)
        assert factor_num >= 0 and factor_num < self.z_dim
        volume = self._get_mean_projection()[factor_num]  # [f,r,r]
        volume = volume.detach().cpu().numpy() # [f,r,r]
        volume = squeeze_triangular_array(volume, dims=(1,2)) # [f,r(r+1)/2]
        return volume.T # [r(r+1)/2,f]



if __name__ == '__main__':
    pass



###
