"""
Base model defining the training procedure

"""
__date__ = "June 2022"


import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    pass

from .. import __commit__ as LPNE_COMMIT
from .. import __version__ as LPNE_VERSION
from .. import INVALID_LABEL
from ..utils.utils import get_weights


FLOAT = torch.float32
INT = torch.int64



class BaseModel(torch.nn.Module):

    def __init__(self, n_iter=50000, batch_size=256, lr=1e-3, device='auto',
        log_dir=None):
        """
        
        Parameters
        ----------
        n_iter : int, optional
        batch_size : int, optional
        lr : float, optional
        device : str, optional
        log_dir : None or str, optional
        
        """
        super(BaseModel, self).__init__()
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.log_dir = log_dir
        self.classes_ = None
        self.groups_ = None


    def _initialize(self):
        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir=self.log_dir)
        self.n_groups = len(self.groups_)
        self.n_classes = len(self.classes_)
        self.to(self.device)
        self.iter_ = 1


    def fit(self, features, labels, groups=None, print_freq=100, score_freq=1):
        """
        Train the model on the given dataset.
        
        Parameters
        ----------
        features : numpy.ndarray
        labels : numpy.ndarray
        groups : None or numpy.ndarray
        print_freq : None or int, optional
        score_freq : None or int, optional

        Returns
        -------
        self : BaseModel
        """
        # Check arguments.
        pass
        # Remove missing data.
        axes = tuple(i for i in range(1,features.ndim))
        idx = np.argwhere(np.isnan(features).sum(axis=axes) == 0).flatten()
        features = features[idx]
        labels = labels[idx]
        if groups is not None:
            groups = groups[idx]
        # Initialize weights, groups, and labels.
        weights = get_weights(labels, groups, invalid_label=INVALID_LABEL)
        idx = np.argwhere(labels == INVALID_LABEL).flatten()
        idx_comp = np.argwhere(labels != INVALID_LABEL).flatten()
        temp_label = np.unique(labels[labels != INVALID_LABEL])[0]
        labels[idx] = temp_label # Mask the labels temporarily.
        self.classes_, labels = np.unique(labels, return_inverse=True)
        labels[idx] = INVALID_LABEL # Unmask the labels.
        assert len(self.classes_) > 1
        if groups is None:
            groups = np.zeros(len(features))
        self.groups_, groups = np.unique(groups, return_inverse=True)
        np_labels = np.copy(labels)
        np_groups = np.copy(groups)
        # Initialize the parameters.
        self._initialize(features.shape)
        # NumPy arrays to PyTorch tensors.
        features = torch.tensor(features, dtype=FLOAT).to(self.device)
        labels = torch.tensor(labels, dtype=INT).to(self.device)
        groups = torch.tensor(groups, dtype=INT).to(self.device)
        weights = torch.tensor(weights, dtype=FLOAT).to(self.device)
        # Make a Dataset, a DataLoader, and an optimizer.
        dset = TensorDataset(features, labels, groups, weights)
        loader = DataLoader(
                dset,
                batch_size=self.batch_size,
                shuffle=True,
        )
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
            if self.log_dir is not None:
                self.writer.add_scalar('train loss', i_loss, self.iter_)
            if print_freq is not None and self.iter_ % print_freq == 0:
                print(f"iter {self.iter_:04d}, loss: {i_loss:3f}")
            if score_freq is not None and self.iter_ % score_freq == 0:
                weighted_acc = self.score(
                        features[idx_comp],
                        np_labels[idx_comp],
                        np_groups[idx_comp],
                )
                if self.log_dir is not None:
                    self.writer.add_scalar(
                            'weighted accuracy',
                            weighted_acc,
                            self.iter_,
                    )
            self.iter_ += 1
        return self


    def get_params(self, deep=True):
        """Get the parameters of this estimator."""
        params = dict(
            batch_size=self.batch_size,
            lr=self.lr,
            device=self.device,
            log_dir=self.log_dir,
            model_name=self.MODEL_NAME,
            __commit__=LPNE_COMMIT,
            __version__=LPNE_VERSION,
        )
        try:
            params['classes_'] = self.classes_
            params['groups_'] = self.groups_
            params['iter_'] = self.iter_
        except:
            pass
        if deep:
            state_dict = self.state_dict()
            if len(state_dict) > 0:
                for key in state_dict:
                    state_dict[key] = state_dict[key].to('cpu')
                params['state_dict'] = state_dict
            try:
                params['optimizer_state_dict'] = self.optimizer_.state_dict()
            except:
                pass
        return params


    def set_params(self, batch_size=None, lr=None, n_iter=None, device=None,
        log_dir=None, classes_=None, groups_=None, iter_=None, state_dict=None,
        optimizer_state_dict=None, **kwargs):
        """Set the parameters of this estimator."""
        if batch_size is not None:
            self.batch_size = batch_size
        if lr is not None:
            self.lr = lr
        if n_iter is not None:
            self.n_iter = n_iter
        if device is not None:
            self.device = device
        if log_dir is not None:
            self.log_dir = log_dir
        if state_dict is not None or optimizer_state_dict is not None:
            self._initialize()
            if state_dict is not None:
                self.load_state_dict(state_dict)
            if optimizer_state_dict is not None:
                self.optimizer_.load_state_dict(optimizer_state_dict)
        if classes_ is not None:
            self.classes_ = classes_
        if groups_ is not None:
            self.groups_ = groups_
        if iter_ is not None:
            self.iter_ = iter_
        return self


    @torch.no_grad()
    def save_state(self, fn):
        """Save parameters for this estimator."""
        params = self.get_params(deep=True)
        np.save(fn, params)


    @torch.no_grad()
    def load_state(self, fn):
        """Load and set the parameters for this estimator."""
        self.set_params(**np.load(fn, allow_pickle=True).item())



if __name__ == '__main__':
    pass


###