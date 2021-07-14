"""
Factor Analysis-regularized logistic regression.

"""
__date__ = "June - July 2021"


import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

FLOAT = torch.float32
INT = torch.int64



class FaSae(torch.nn.Module):

    def __init__(self, n_features, n_classes, reg_strength=1.0, z_dim=20,
        class_weights=None):
        """
        Some notes...

        Attributes
        ----------
        ...

        Parameters
        ----------
        n_features: int
            Total number of features
        n_classes : int
            Number of label classes
        reg_strength : float, optional
            This controls how much the classifier is regularized. This should
            be positive, and larger values indicate more regularization.
        z_dim : int, optional
            Latent dimension/number of networks.
        class_weights : None or numpy.ndarray
            Used to upweight rarer labels. No upweighting is performed if this
            is `None`.
        """
        super(FaSae, self).__init__()
        assert n_classes > 1, f"{n_classes} <= 1"
        assert z_dim >= n_classes, f"{z_dim} < {n_classes}"
        assert reg_strength >= 0.0
        self.trained = False # Has this model been trained yet?
        self.n_features = n_features
        self.reg_strength = reg_strength
        self.n_classes = n_classes
        if class_weights is None:
            self.class_weights = class_weights
        else:
            self.class_weights = torch.tensor(class_weights, dtype=FLOAT)
        self.z_dim = z_dim
        self.recognition_model = torch.nn.Linear(self.n_features, self.z_dim)
        self.model = torch.nn.Linear(self.z_dim, self.n_features)


    def forward(self, features, labels):
        """
        Calculate a loss for the features and labels.

        Parameters
        ----------
        features : torch.Tensor
            Shape: [batch,n_features]
        labels : torch.Tensor
            Shape: [batch]

        Returns
        -------
        loss : torch.Tensor
            Shape: []
        """
        # Feed through the recognition network to get latents.
        zs = self.recognition_model(features)
        # Reconstruct the features.
        features_rec = self.model(zs)
        # Calculate a reconstruction loss.
        rec_loss = torch.mean(features - features_rec, dim=1) # [b]
        # Predict the labels.
        logits = zs[:,:self.n_classes-1]
        zeros = torch.zeros(
                logits.shape[0],
                1,
                dtype=logits.dtype,
                device=logits.device,
        )
        logits = torch.cat([logits, zeros], dim=1)
        log_probs = Categorical(logits=logits).log_prob(labels)
        if self.class_weights is not None:
            weight_vector = self.class_weights[labels]
            log_probs = weight_vector * log_probs
        # Combine all the terms into a loss.
        loss = self.reg_strength * rec_loss - log_probs
        return torch.mean(loss)


    def fit(self, features, labels, epochs=100, lr=1e-3, batch_size=64,
        verbose=True, print_freq=10):
        """
        Train the model on the given dataset.

        Parameters
        ----------
        features : numpy.ndarray
        labels : numpy.ndarray
        epochs : int, optional
            Number of training epochs.
        lr : float, optional
            Learning rate.
        batch_size : int, optional
        verbose : bool, optional
        print_freq : int, optional
        """
        assert not self.trained, "FA model is already trained!"
        self.trained = True
        # To torch tensors.
        features = torch.tensor(features, dtype=FLOAT)
        labels = torch.tensor(labels, dtype=INT)
        # Make some loaders and an optimizer.
        dset = TensorDataset(features, labels)
        loader = DataLoader(dset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # Train.
        for epoch in range(1,epochs+1):
            epoch_loss = 0.0
            for batch in loader:
                self.zero_grad()
                loss = self(*batch)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            if epoch % print_freq == 0 and verbose:
                print(f"epoch {epoch}, loss: {loss}")


    def predict(self, features):
        """
        Predict class labels for the features.

        Parameters
        ----------
        features : numpy.ndarray
            Shape: [batch, n_features]

        Returns
        -------
        predictions : numpy.ndarray
            Shape: [batch]
        """
        # To torch tensors.
        features = torch.tensor(features, dtype=FLOAT)
        with torch.no_grad():
            probs = self.predict_proba(features, to_numpy=False)
            predictions = torch.argmax(probs, dim=1)
        return predictions.cpu().numpy()


    def predict_proba(self, features, to_numpy=True):
        """
        Probability estimates.

        Note
        ----
        * This should be consistent with `self.forward`.

        Parameters
        ----------
        features : numpy.ndarray
            Shape: [batch, n_features]

        Returns
        -------
        probs : numpy.ndarray
            Shape: [batch, n_classes]
        """
        with torch.no_grad():
            # Feed through the recognition network to get latents.
            zs = self.recognition_model(features)
            # Get class predictions.
            logits = zs[:,:self.n_classes-1]
            zeros = torch.zeros(
                    logits.shape[0],
                    1,
                    dtype=logits.dtype,
                    device=logits.device,
            )
            logits = torch.cat([logits, zeros], dim=1)
            probs = F.softmax(logits, dim=1) # [b, n_classes]
        if to_numpy:
            return probs.cpu().numpy()
        return probs


    def score(self, features, labels, class_weights):
        """
        Get a class weighted accuracy.

        This is the objective we really care about, which doesn't contain the
        regularization in FA's `forward` method.

        Parameters
        ----------
        features : numpy.ndarray
            Shape: [n_datapoints, n_features]
        labels : numpoy.ndarray
            Shape: [n_datapoints]
        class_weights : None or numpy.ndarray
            Shape: [n_classes]

        Return
        ------
        weighted_acc : float
        """
        predictions = self.predict(features)
        scores = np.zeros(len(features))
        scores[predictions == labels] = 1.0
        if class_weights is not None:
            scores = scores * class_weights[labels]
        weighted_acc = np.mean(scores)
        return weighted_acc


    def get_params(self):
        """Get parameters for this estimator."""
        return {
            'model_state_dict': self.state_dict(),
            'trained': self.trained,
            'n_features': self.n_features,
            'reg_strength': self.reg_strength,
            'n_classes': self.n_classes,
            'class_weights': self.class_weights,
            'z_dim': self.z_dim,
		}


    def set_params(self, params):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        params : dict
        """
        assert self.n_features == params['n_features']
        assert self.n_classes == params['n_classes']
        assert self.z_dim == params['z_dim']
        self.trained = params['trained']
        self.reg_strength = params['reg_strength']
        self.class_weights = params['class_weights']
        self.load_state_dict(params['model_state_dict'])


    def save_state(self, fn):
        """Save parameters for this estimator."""
        np.save(fn, self.get_params())


    def load_state(self, fn):
        """Load and set the parameters for this estimator."""
        self.set_params(np.load(f, allow_pickle=True).item())



if __name__ == '__main__':
    """Here's an example using some fake data."""
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
    model = FaSae(n_features, n_classes, class_weights=class_weights)

    # Fit the model.
    print("Training model...")
    model.fit(features, labels)

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
