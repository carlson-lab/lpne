"""
Factor Analysis-regularized logistic regression.

"""
__date__ = "June - November 2021"


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

from ..utils.utils import get_weights


# https://stackoverflow.com/questions/53014306/
if float(torch.__version__[:3]) >= 1.9:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


FLOAT = torch.float32
INT = torch.int64
MAX_LABEL = 1000
EPSILON = 1e-6



class FaSae(torch.nn.Module, BaseEstimator):

    def __init__(self, reg_strength=1.0, z_dim=32, weight_reg=0.0,
        nonnegative=True, variational=False, kl_factor=1.0, n_iter=50000,
        lr=1e-3, batch_size=256, beta=0.5, device='auto'):
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
        super(FaSae, self).__init__()
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
        assert isinstance(n_iter, int)
        assert n_iter > 0
        self.n_iter = n_iter
        assert isinstance(lr, (int, float))
        assert lr > 0.0
        self.lr = float(lr)
        assert isinstance(batch_size, int)
        assert batch_size > 0
        self.batch_size = batch_size
        assert isinstance(beta, (int, float))
        assert beta >= 0.0 and beta <= 1.0
        self.beta = float(beta)
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.classes_ = None


    def _initialize(self, n_features):
        """Initialize parameters of the networks before training."""
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
        self.to(self.device)


    def fit(self, features, labels, print_freq=500):
        """
        Train the model on the given dataset.

        Parameters
        ----------
        features : numpy.ndarray
        labels : numpy.ndarray
        n_iter : int, optional
            Number of training epochs.
        lr : float, optional
            Learning rate.
        batch_size : int, optional
        verbose : bool, optional
        print_freq : None or int, optional
        """
        # Check arguments.
        features, labels = check_X_y(features, labels)

        # Derive groups, labels, and weights from labels.
        groups, labels, weights = _derive_groups(labels)

        self.classes_, labels = np.unique(labels, return_inverse=True)
        if features.shape[0] != labels.shape[0]:
            raise ValueError(f"{features.shape}[0] != {labels.shape}[0]")
        if len(features.shape) != 2:
            raise ValueError(f"len({features.shape}) != 2")
        if len(labels.shape) != 1:
            raise ValueError(f"len({labels.shape}) != 1")
        self._initialize(features.shape[1])
        # NumPy arrays to PyTorch tensors.
        features = torch.tensor(features, dtype=FLOAT).to(self.device)
        labels = torch.tensor(labels, dtype=INT).to(self.device)
        weights = torch.tensor(weights, dtype=FLOAT).to(self.device)
        sampler_weights = torch.pow(weights, 1.0 - self.beta)
        weights = torch.pow(weights, self.beta)
        # Make some loaders and an optimizer.
        dset = TensorDataset(features, labels, weights)
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


    def forward(self, features, labels, weights):
        """
        Calculate a loss for the features and labels.

        Parameters
        ----------
        features : torch.Tensor
            Shape: [batch,n_features]
        labels : torch.Tensor
            Shape: [batch]
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
    def predict(self, X):
        """
        Predict class labels for the features.

        Parameters
        ----------
        X : numpy.ndarray
            Features
            Shape: [batch, n_features]

        Returns
        -------
        predictions : numpy.ndarray
            Shape: [batch]
        """
        # Checks
        check_is_fitted(self)
        X = check_array(X)
        # Feed through model.
        X = torch.tensor(X, dtype=FLOAT).to(self.device)
        probs = self.predict_proba(X, to_numpy=False)
        predictions = torch.argmax(probs, dim=1)
        return self.classes_[predictions.cpu().numpy()]


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
        if self.variational:
            # Feed through the recognition network to get latents.
            z_mus = self.rec_model_1(features)
            z_log_stds = self.rec_model_2(features)
            if stochastic:
                # Make the variational posterior and sample.
                dist = Normal(z_mus, EPSILON + z_log_stds.exp())
                zs = dist.rsample() # [b,z]
            else:
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
    def score(self, features, labels):
        """
        Get a class weighted accuracy.

        This is the objective we really care about, which doesn't contain the
        regularization in FA's `forward` method.

        Parameters
        ----------
        features : numpy.ndarray
            Shape: [n_datapoints, n_features]
        labels : numpy.ndarray
            Shape: [n_datapoints]
        weights : None or numpy.ndarray
            Shape: [n_datapoints]

        Return
        ------
        weighted_acc : float
        """
        # Derive groups, labels, and weights from labels.
        groups, labels, weights = _derive_groups(labels)
        predictions = self.predict(features)
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
            'nonnegative': self.nonnegative,
            'variational': self.variational,
            'kl_factor': self.kl_factor,
            'n_iter': self.n_iter,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'beta': self.beta,
            'device': self.device,
            'classes_': self.classes_,
        }
        if deep:
            params['model_state_dict'] = self.state_dict()
        return params


    def set_params(self, reg_strength=None, z_dim=None, weight_reg=None,
        nonnegative=None, variational=None, kl_factor=None, n_iter=None,
        lr=None, batch_size=None, beta=None, device=None, classes_=None,
        model_state_dict=None):
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
        if nonnegative is not None:
            self.nonnegative = nonnegative
        if variational is not None:
            self.variational = variational
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
        if model_state_dict is not None:
            assert 'model.bias' in model_state_dict, \
                    f"'model.bias' not in {list(model_state_dict.keys())}"
            n_features = len(model_state_dict['model.bias'].view(-1))
            self._initialize(n_features)
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
        """
        check_is_fitted(self)
        assert isinstance(factor_num, int)
        assert factor_num >= 0 and factor_num < self.z_dim
        A = self.model.weight[:,factor_num]
        if self.nonnegative:
            A = F.softplus(A)
        return A.detach().cpu().numpy()



def _derive_groups(labels):
    groups = np.array([label // MAX_LABEL for label in labels])
    labels = np.array([label % MAX_LABEL for label in labels])
    weights = get_weights(labels, groups)
    return groups, labels, weights



if __name__ == '__main__':
    """Here's an example using some fake data."""
    raise NotImplementedError
    n = 100 # number of datapoints/windows
    n_features = 100 # total number of LFP features
    n_classes = 3 # number of label types

    # Make some fake data.
    features = np.random.randn(n, n_features)
    labels = np.random.randint(n_classes, size=n)

    # Calculate class weights.
    class_counts = [len(np.argwhere(labels==i)) for i in range(n_classes)]
    print("Class counts:", class_counts)
    class_weights = n / (n_classes * np.array(class_counts))
    print("Class weights:", class_weights)

    # Make the model.
    model = FaSae(
            n_features,
            n_classes,
            class_weights=class_weights,
            weight_reg=0.0,
            nonnegative=True,
            variational=True,
            kl_factor=0.1,
    )

    # Fit the model.
    print("Training model...")
    model.fit(features, labels, epochs=5000, print_freq=250)

    # Make some predictions.
    print("Making predictions...")
    predictions = model.predict(features)
    print("Predictions:")
    print(predictions)

    # Calculate a weighted accuracy.
    weighted_acc = model.score(
            features,
            labels,
            class_weights,
    )
    print("Weighted accuracy on training set:", weighted_acc)

    # Get state.
    params = model.get_params()

    # Make a new model and load the state.
    new_model = FaSae(n_features, n_classes)
    new_model.set_params(params)

    # Calculate a weighted accuracy.
    weighted_acc = new_model.score(
            features,
            labels,
            class_weights,
    )
    print("This should be the same number:", weighted_acc)



###
