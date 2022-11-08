"""
Base model defining the training procedure and some common methods for SAEs

"""
__date__ = "June - July 2022"


import numpy as np
from sklearn.utils.validation import check_is_fitted
import torch
from torch.utils.data import TensorDataset, DataLoader
import warnings

from .. import __commit__ as LPNE_COMMIT
from .. import __version__ as LPNE_VERSION
from .. import INVALID_LABEL
from ..utils.utils import get_weights


FLOAT = torch.float32
INT = torch.int64


class BaseModel(torch.nn.Module):
    def __init__(self, n_iter=50000, batch_size=256, lr=1e-3, device="auto"):
        """

        Parameters
        ----------
        n_iter : int, optional
            Number of epochs to train
        batch_size : int, optional
            DataLoader batch size
        lr : float, optional
            Learning rate
        device : str, optional
            Pytorch device
        """
        super(BaseModel, self).__init__()
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classes_ = None
        self.groups_ = None

    def _initialize(self):
        self.n_groups = len(self.groups_)
        self.n_classes = len(self.classes_)
        self.to(self.device)
        self.iter_ = 1

    def fit(
        self,
        features,
        labels,
        groups=None,
        print_freq=5,
        score_freq=20,
        random_state=None,
    ):
        """
        Train the model on the given dataset.

        Parameters
        ----------
        features : numpy.ndarray
            Shape: [b,f,r,r]
        labels : numpy.ndarray
            Shape: [b]
        groups : None or numpy.ndarray
            Shape: [b]
        print_freq : int or None, optional
            Print loss every ``print_freq`` epochs.
        score_freq : int or None, optional
            Print weighted accuracy every ``score_freq`` epochs.
        random_state : int or None, optional
            A random seed for training. If ``None``, then no seed is set.

        Returns
        -------
        self : BaseModel
            The fitted model
        """
        # Check arguments.
        assert features.ndim == 4
        assert labels.ndim == 1
        assert groups.ndim == 1
        assert len(features) == len(labels) and len(labels) == len(groups)
        # Remove missing data.
        axes = tuple(i for i in range(1, features.ndim))
        idx = np.argwhere(np.isnan(features).sum(axis=axes) == 0).flatten()
        features = features[idx]
        labels = labels[idx]
        if groups is not None:
            groups = groups[idx]
        # Initialize weights, groups, and labels.
        weights = get_weights(labels, groups, invalid_label=INVALID_LABEL)
        idx = np.argwhere(labels == INVALID_LABEL).flatten()
        idx_comp = np.argwhere(labels != INVALID_LABEL).flatten()
        # Mask the labels temporarily, get the classes, and unmask.
        temp_label = np.unique(labels[labels != INVALID_LABEL])[0]
        labels[idx] = temp_label
        self.classes_, labels = np.unique(labels, return_inverse=True)
        labels[idx] = INVALID_LABEL
        assert len(self.classes_) > 1
        # Figure out the groups.
        if groups is None:
            groups = np.zeros(len(features))
        np_groups = np.copy(groups)
        self.groups_, groups = np.unique(groups, return_inverse=True)
        np_labels = np.copy(labels)
        # Set the random seed if one is given.
        if random_state is not None:
            torch.manual_seed(random_state)
        # Initialize the parameters.
        self.features_shape_ = features.shape
        self._initialize()
        # NumPy arrays to PyTorch tensors.
        features = torch.tensor(features, dtype=FLOAT).to(self.device)
        labels = torch.tensor(labels, dtype=INT).to(self.device)
        groups = torch.tensor(groups, dtype=INT).to(self.device)
        weights = torch.tensor(weights, dtype=FLOAT).to(self.device)
        # Make a Dataset, a DataLoader, and an optimizer.
        dset = TensorDataset(features, labels, groups, weights)
        loader = DataLoader(dset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # Train.
        while self.iter_ <= self.n_iter:
            i_loss = 0.0
            for batch in loader:
                self.zero_grad()
                loss = self(*batch)
                i_loss += loss.item()
                loss.backward()
                optimizer.step()
            if print_freq is not None and self.iter_ % print_freq == 0:
                print(f"iter {self.iter_:04d}, loss: {i_loss:.3f}")
            if score_freq is not None and self.iter_ % score_freq == 0:
                weighted_acc = self.score(
                    features[idx_comp],
                    np_labels[idx_comp],
                    np_groups[idx_comp],
                )
                print(f"iter {self.iter_:04d}, acc: {weighted_acc:.3f}")
            self.iter_ += 1
        return self

    @torch.no_grad()
    def reconstruct(self, features):
        """
        Reconstruct the features by sending them round trip through the model.

        Parameters
        ----------
        features : numpy.ndarray
            Shape: ``[b,x]`` or ``[b,f,r,r]``

        Returns
        -------
        rec_features : numpy.ndarray
            Shape: same as ``features``
        """
        check_is_fitted(self, attributes=self.FIT_ATTRIBUTES)
        assert features.ndim in [2, 4]
        flag = features.ndim == 4
        if flag:
            assert features.shape[2] == features.shape[3]
            orig_shape = features.shape
            features = features.reshape(len(features), -1)
        rec_features = []
        i = 0
        while i <= len(features):
            batch_f = features[i : i + self.batch_size]
            batch_f = torch.tensor(batch_f).to(self.device, FLOAT)
            batch_zs = self.get_latents(batch_f)
            batch_rec = self.project_latents(batch_zs)
            rec_features.append(batch_rec.cpu())
            i += self.batch_size
        rec_features = torch.cat(rec_features, dim=0).numpy()
        if flag:
            return rec_features.reshape(orig_shape)
        return rec_features

    def get_params(self, deep=True):
        """Get the parameters of this estimator."""
        params = dict(
            batch_size=self.batch_size,
            lr=self.lr,
            device=self.device,
            model_name=self.MODEL_NAME,
            __commit__=LPNE_COMMIT,
            __version__=LPNE_VERSION,
        )
        try:
            params["classes_"] = self.classes_
            params["groups_"] = self.groups_
            params["iter_"] = self.iter_
            params["features_shape_"] = self.features_shape_
        except:
            pass
        if deep:
            state_dict = self.state_dict()
            if len(state_dict) > 0:
                for key in state_dict:
                    state_dict[key] = state_dict[key].to("cpu")
                params["state_dict"] = state_dict
            try:
                params["optimizer_state_dict"] = self.optimizer_.state_dict()
            except:
                pass
        return params

    def set_params(
        self,
        batch_size=None,
        lr=None,
        n_iter=None,
        device=None,
        classes_=None,
        groups_=None,
        iter_=None,
        features_shape_=None,
        state_dict=None,
        optimizer_state_dict=None,
        **kwargs,
    ):
        """Set the parameters of this estimator."""
        if batch_size is not None:
            self.batch_size = batch_size
        if lr is not None:
            self.lr = lr
        if n_iter is not None:
            self.n_iter = n_iter
        if device is not None:
            if (not torch.cuda.is_available()) and device != "cpu":
                warnings.warn("Loading GPU-trained model as a CPU model.")
                self.device = "cpu"
            else:
                self.device = device
        if features_shape_ is not None:
            self.features_shape_ = features_shape_
        if classes_ is not None:
            self.classes_ = classes_
        if groups_ is not None:
            self.groups_ = groups_
        if iter_ is not None:
            self.iter_ = iter_
        if state_dict is not None or optimizer_state_dict is not None:
            self._initialize()
            if state_dict is not None:
                self.load_state_dict(state_dict)
            if optimizer_state_dict is not None:
                self.optimizer_.load_state_dict(optimizer_state_dict)
        return self

    @torch.no_grad()
    def save_state(self, fn):
        """Save parameters for this estimator."""
        params = self.get_params(deep=True)
        np.save(fn, params)

    @torch.no_grad()
    def load_state(self, fn):
        """Load and set the parameters for this estimator."""
        d = np.load(fn, allow_pickle=True).item()
        if "model_name" in d:
            assert (
                d["model_name"] == self.MODEL_NAME
            ), f"Expected {self.MODEL_NAME}, found {d['model_name']}"
        else:
            warnings.warn("Didn't find field model_name when loading model.")
        self.set_params(**d)


if __name__ == "__main__":
    pass


###
