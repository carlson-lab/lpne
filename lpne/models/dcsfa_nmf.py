"""
Hunter's supervised autoencoder model

"""
__date__ = "September - October 2022"


import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import warnings


from .nmf_base import NmfBase


class DcsfaNmf(NmfBase):
    """
    dCSFA-NMF model

    Parameters
    ----------
    n_components : int, optional
        number of networks to learn or latent dimensionality
        NOTE: the sklearn convention is for all parameters to have default
        values.
    device : ``{'cuda','cpu','auto'}``, optional
        Torch device. Defaults to ``'auto'``.
    n_intercepts : int, optional
        Number of unique intercepts for logistic regression
        (sometimes mouse specific intercept is helpful). Defaults to ``1``.
        NOTE: can this be detected automatically, given a groups argument?
    n_sup_networks : int, optional
        Number of networks that will be supervised
        ``0 < n_sup_networks < n_components``. Defaults to ``1``.
    optim_name : ``{'SGD','Adam','AdamW'}``, optional
        torch.optim algorithm to use in . Defaults to ``'AdamW'``.
    recon_loss : ``{'IS', 'MSE'}``, optional
        Reconstruction loss function. Defaults to ``'MSE'``.
    recon_weight : float, optional
        Importance weight for the reconstruction. Defaults to ``1.0``.
    sup_weight : float, optional
        Importance weight for the supervision. Defaults to ``1.0``.
    sup_recon_weight : float, optional
        Importance weight for the reconstruction of the supervised component.
        Defaults to ``1.0``.
    use_deep_encoder : bool, optional
        NOTE: I've changed the name to use underscores
        Whether to use a deep or linear encoder. Defaults to ``True``.
    h : int, optional
        Hidden layer size for the deep encoder. Defaults to ``256``.
    sup_recon_type : ``{'Residual', 'All'}``, optional
        Which supervised component reconstruction loss to use. Defaults to
        ``'Residual'``. ``'Residual'`` estimates network scores optimal for
        reconstruction and penalizes deviation of the real scores from those
        values. ``'All'`` evaluates the reconstruction loss of the supervised
        network reconstruction against all features.
    feature_groups : ``None`` or list of int, optional
        Indices of the divisions of feature types. Defaults to ``None``.
    group_weights : ``None`` or list of floats, optional
        Weights for each of the feature types. Defaults to ``None``.
    fixed_corr : ``None`` or list of str, optional
        List the same length as ``n_sup_networks`` indicating correlation
        constraints for the network. Defaults to ``None``. ``'positive'``
        constrains a supervised network to have a positive correlation between
        score and label. ``'negative'`` constrains a supervised network to have
        a negative correlation between score and label. ``'n/a'`` applies no
        constraint, meaning the supervised network can be positive or
        negatively correlated.
    momentum : float, optional
        Momentum value if optimizer is ``'SGD'``. Defaults to ``0.9``.
    lr : float, optional
        Learning rate for the optimizer. Defaults to ``1e-3``.
    sup_smoothness_weight : float, optional
        Encourages smoothness for the supervised network. Defaults to ``1.0``.
    save_folder : str, optional
        Location to save the best pytorch model parameters. Defaults to
        ``'~'``. NOTE: this seems like a bad default. Maybe a ``save_filename``
        parameter would be better.
    verbose : bool, optional
        Activates or deactivates print statements. Defaults to ``False``.
    """

    def __init__(
        self,
        n_components=32,
        device="auto",
        n_intercepts=1,
        n_sup_networks=1,
        optim_name="AdamW",
        recon_loss="MSE",
        recon_weight=1.0,
        sup_weight=1.0,
        sup_recon_weight=1.0,
        use_deep_encoder=True,
        h=256,
        sup_recon_type="Residual",
        feature_groups=None,
        group_weights=None,
        fixed_corr=None,
        momentum=0.9,
        lr=1e-3,
        sup_smoothness_weight=1.0,
        save_folder="~",
        verbose=False,
    ):
        """ """
        super(DcsfaNmf, self).__init__(
            n_components,
            device,
            n_sup_networks,
            fixed_corr,
            recon_loss,
            recon_weight,
            sup_recon_type,
            sup_recon_weight,
            sup_smoothness_weight,
            feature_groups,
            group_weights,
            verbose,
        )
        self.n_intercepts = n_intercepts
        self.optim_name = optim_name
        self.optim_alg = self.get_optim(optim_name)
        self.pred_loss_f = nn.BCELoss
        self.recon_weight = recon_weight
        self.sup_weight = sup_weight
        self.use_deep_encoder = use_deep_encoder
        self.h = h
        self.momentum = momentum
        self.lr = lr
        self.sup_smoothness_weight = sup_smoothness_weight
        self.save_folder = save_folder

    def _initialize(self, dim_in):
        """
        Initializes encoder and NMF parameters using the input dimensionality

        Parameters
        ----------
        dim_in : int
            Total number of features
        """
        # NOTE: the sklearn convention is for attributes assigned after the
        # __init__ function to have trailing underscores. It doesn't look like
        # this attribute is every used, though, so maybe it should be removed.
        self.dim_in_ = dim_in
        # NOTE: I would change this to: super(dCSFA_NMF, self)._initialize()
        # to make it more readable, renaming the _initialize_NMF to just
        # _initialize.
        self._initialize_NMF(dim_in)
        if self.use_deep_encoder:
            # NOTE: It's easier to read nested structures that look like this,
            # with the closing parenthesis on the same level as the opening.
            # NOTE: I've also changed Encoder to encoder to match conventions.
            self.encoder = nn.Sequential(
                nn.Linear(dim_in, self.h),
                nn.BatchNorm1d(self.h),
                nn.LeakyReLU(),
                nn.Linear(self.h, self.n_components),
                nn.Softplus(),
            )
        else:
            # NOTE: this can also be a Sequential object too.
            self.encoder = nn.Sequential(
                nn.Linear(dim_in, self.n_components),
                nn.Softplus(),
            )
        # Initialize logistic regression parameters.
        # NOTE: these could larger torch tensors instead of lists.
        self.phi_list = nn.ParameterList(
            [nn.Parameter(torch.randn(1)) for _ in range(self.n_sup_networks)]
        )
        self.beta_list = nn.ParameterList(
            [
                nn.Parameter(torch.randn(self.n_intercepts, 1))
                for _ in range(self.n_sup_networks)
            ]
        )
        self.to(self.device)

    def instantiate_optimizer(self):
        """
        Create an optimizer.

        Returns
        -------
        optimizer : torch.optim.Optimizer
            Torch optimizer
        """
        if self.optim_name == "SGD":
            optimizer = self.optim_alg(
                self.parameters(),
                lr=self.lr,
                momentum=self.momentum,
            )
        else:
            optimizer = self.optim_alg(self.parameters(), lr=self.lr)
        return optimizer

    def get_all_class_predictions(self, X, s, intercept_mask, avg_intercept):
        """
        Get predictions for every supervised network.

        Parameters
        ----------
        X : torch.Tensor
            Input features
            Shape: ``[batch_size,n_features]``
        s : torch.Tensor
            latent embeddings
            Shape: ``[batch_size,n_components]``
        intercept_mask : torch.Tensor
            One-hot encoded mask for logistic regression intercept terms
            Shape: ``[batch_size,n_intercepts]``
        avg_intercept : bool
            Whether to average all intercepts together.

        Returns
        -------
        y_pred_proba : torch.Tensor
            Predictions
            Shape: ``[batch_size, n_sup_networks]``
        """
        if intercept_mask is None and avg_intercept is False:
            warnings.warn(
                f"Intercept mask cannot be none and avg_intercept False... "
                f"Averaging Intercepts",
            )
            avg_intercept = True
        # Get predictions for each class.
        # NOTE: it seems like this could be vectorized
        y_pred_list = []
        for sup_net in range(self.n_sup_networks):
            if self.n_intercepts == 1:
                # NOTE: use torch.sigmoid instead of torch.nn.Sigmoid
                # NOTE: squeeze() without a dimension argument could result in
                # unexpected behavior if there's a batch dimension of 1
                # somewhere.
                y_pred_proba = torch.sigmoid(
                    s[:, sup_net].view(-1, 1) * self.get_phi(sup_net)
                    + self.beta_list[sup_net],
                ).squeeze()
            elif self.n_intercepts > 1 and not avg_intercept:
                y_pred_proba = torch.sigmoid(
                    s[:, sup_net].view(-1, 1) * self.get_phi(sup_net)
                    + intercept_mask @ self.beta_list[sup_net],
                ).squeeze()
            else:
                intercept_mask = (
                    torch.ones(X.shape[0], self.n_intercepts).to(self.device)
                    / self.n_intercepts
                )
                y_pred_proba = torch.sigmoid(
                    s[:, sup_net].view(-1, 1) * self.get_phi(sup_net)
                    + intercept_mask @ self.beta_list[sup_net]
                ).squeeze()
            y_pred_list.append(y_pred_proba.view(-1, 1))
        # Concatenate predictions into a single matrix [n_samples,n_tasks]
        y_pred_proba = torch.cat(y_pred_list, dim=1)
        return y_pred_proba

    def get_embedding(self, X):
        """
        Map features to latents

        Parameters
        ----------
        X : torch.Tensor
            Input features
            Shape: ``[batch_size,n_features]``

        Returns
        -------
        s : torch.Tensor
            Latent embeddings (scores)
            Shape: ``[batch_size,n_components]``
        """
        # NOTE: this is a one-liner because both encoders are Sequential
        # objects
        return self.encoder(X)

    def get_phi(self, sup_net=0):
        """
        Return the logistic regression coefficient correspoding to ``sup_net``.

        # NOTE: Raises before Parameters and Returns

        Raises
        ------
        * ``ValueError`` if ``self.fixed_corr[sup_net]`` is not in
          ``{'n/a', 'positive', 'negative'}``. This should be caught at
          initialization. # NOTE: it's better to catch this a initialization.

        Parameters
        ----------
        sup_net : int, optional
            Which supervised network you would like to get a coefficient for.
            Defaults to ``0``.

        Returns
        -------
        phi : torch.Tensor
            The coefficient that has either been returned raw, or through a
            positive or negative softplus(phi).
        """
        fixed_corr_str = self.fixed_corr[sup_net].lower()
        if fixed_corr_str == "n/a":
            return self.phi_list[sup_net]
        elif fixed_corr_str == "positive":
            # NOTE: use nn.functional instead of nn here.
            return F.softplus(self.phi_list[sup_net])
        elif fixed_corr_str == "negative":
            return -1 * F.softplus(self.phi_list[sup_net])
        else:
            # NOTE: spit out an informative error
            raise ValueError(f"Unsupported fixed_corr value: {fixed_corr_str}")

    def forward(
        self, X, y, task_mask, pred_weight, intercept_mask=None, avg_intercept=False
    ):
        """
        dCSFA-NMF forward pass

        Parameters
        ----------
        X : torch.Tensor
            Input Features
            Shape: ``[batch_size,n_features]``
        y : torch.Tensor
            Ground truth labels
            Shape: ``[batch_size,n_sup_networks]``
        task_mask : torch.Tensor
            Per window mask for whether or not predictions should be counted
            Shape: ``[batch_size,n_sup_networks]``
        pred_weight : torch.Tensor
            Per window classification importance weighting
            Shape: ``[batch_size,1]``
        intercept_mask : ``None`` or torch.Tensor, optional
            Window specific intercept mask. Defaults to ``None``.
            Shape: ``[batch_size,n_intercepts]``
        avg_intercept : bool, optional
            Whether or not to average intercepts. This is used in evaluation.
            Defaults to ``False``.

        Returns
        -------
        recon_loss (torch.Tensor):
            ``recon_weight*full_recon_loss + sup_recon_weight*sup_recon_loss``
        pred_loss (torch.Tensor):
            ``sup_weight * BCELoss()``
        """
        # Get the scores from the encoder
        s = self.get_embedding(X)

        # Get the reconstruction losses
        recon_loss = self.NMF_decoder_forward(X, s)

        # Get predictions
        y_pred = self.get_all_class_predictions(
            X,
            s,
            intercept_mask,
            avg_intercept,
        )
        pred_loss_f = self.pred_loss_f(weight=pred_weight)
        pred_loss = self.sup_weight * pred_loss_f(
            y_pred * task_mask,
            y * task_mask,
        )

        # recon loss and pred loss are left seperate so only recon_loss can be
        # applied for encoder pretraining
        # NOTE: consider adding a ``pretraining`` parameter to this method
        # instead of passing two losses.
        return recon_loss, pred_loss

    @torch.no_grad()
    def transform(self, X, intercept_mask=None, avg_intercept=True, return_npy=True):
        """
        Transform method to return reconstruction, predictions, and projections

        Parameters
        ----------
        X (torch.Tensor): Input Features
            Shape: ``[batch_size,n_features]``
        intercept_mask (torch.Tensor, optional): window specific intercept
            mask. Defaults to None.
            Shape: ``[batch_size,n_intercepts]``
        avg_intercept (bool, optional): Whether or not to average intercepts -
            used in evaluation. Defaults to False.
        return_npy (bool, optional): Whether or not to convert to numpy arrays.
            Defaults to True.

        Returns
        -------
        X_recon (torch.Tensor) : Full reconstruction of input features
            Shape: ``[n_samples,n_features]``
        y_pred (torch.Tensor) : All task predictions
            Shape: ``[n_samples,n_sup_networks]``
        s (torch.Tensor) : Network activation scores
            Shape: ``[n_samples,n_components]``
        """
        if not torch.is_tensor(X):
            X = torch.Tensor(X).float().to(self.device)

        s = self.get_embedding(X)
        X_recon = self.get_all_comp_recon(s)
        y_pred = self.get_all_class_predictions(
            X,
            s,
            intercept_mask,
            avg_intercept,
        )
        if return_npy:
            s = s.detach().cpu().numpy()
            X_recon = X_recon.detach().cpu().numpy()
            y_pred = y_pred.detach().cpu().numpy()
        return X_recon, y_pred, s

    def pretrain_encoder(
        self,
        X,
        y,
        y_pred_weights,
        task_mask,
        intercept_mask,
        sample_weights,
        n_pre_epochs=100,
        batch_size=128,
    ):
        """
        Pretrain the encoder.

        Parameters
        ----------
        X (torch.Tensor): Input Features
            Shape: ``[n_samples,n_features]``
        y (torch.Tensor): ground truth labels
            Shape: ``[n_samples,n_sup_networks]``
        task_mask (torch.Tensor):
            per window mask for whether or not predictions should be counted
            Shape: ``[n_samples,n_sup_networks]``
        y_pred_weight (torch.Tensor):
            per window classification importance weighting
            Shape: ``[n_samples,1]``
        intercept_mask (torch.Tensor, optional):
            window specific intercept mask. Defaults to None.
            Shape: ``[n_samples,n_intercepts]``
        sample_weights (torch.Tensor):
            Gradient Descent sampling weights.
            Shape: ``[n_samples,1]
        n_pre_epochs (int,optional):
            number of epochs for pretraining the encoder. Defaults to 100
        batch_size (int,optional):
            batch size for pretraining. Defaults to 128
        """
        # Freeze the decoder
        self.W_nmf.requires_grad = False
        # Load arguments onto device
        X = torch.Tensor(X).float().to(self.device)
        y = torch.Tensor(y).float().to(self.device)
        y_pred_weights = torch.Tensor(y_pred_weights).float().to(self.device)
        task_mask = torch.Tensor(task_mask).long().to(self.device)
        intercept_mask = torch.Tensor(intercept_mask).to(self.device)
        sample_weights = torch.Tensor(sample_weights).to(self.device)
        # Create a Dataset.
        # dset = TensorDataset(X,y,y_pred_weights,task_mask,intercept_mask)
        # NOTE: I changed the order to match ``self.forward``
        dset = TensorDataset(X, y, task_mask, y_pred_weights, intercept_mask)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        loader = DataLoader(dset, batch_size=batch_size, sampler=sampler)
        # Instantiate Optimizer
        optimizer = self.instantiate_optimizer()
        # Define iterator
        if self.verbose:
            epoch_iter = tqdm(range(n_pre_epochs))
        else:
            epoch_iter = range(n_pre_epochs)
        for epoch in epoch_iter:
            r_loss = 0.0
            # NOTE: don't unpack everything if you don't need to:
            for batch in loader:
                optimizer.zero_grad()
                recon_loss, _ = self.forward(*batch)
                recon_loss.backward()
                optimizer.step()
                r_loss += recon_loss.item()
            if self.verbose:
                # NOTE: python f-strings are more concise than ``.format``
                avg_r = r_loss / len(loader)
                epoch_iter.set_description(
                    f"Encoder Pretrain Epoch: {epoch}, Recon Loss: {avg_r:.6}",
                )
        # Unfreeze the NMF decoder.
        self.W_nmf.requires_grad = True

    def fit(
        self,
        X,
        y,
        y_pred_weights=None,
        task_mask=None,
        intercept_mask=None,
        y_sample_groups=None,
        n_epochs=100,
        n_pre_epochs=100,
        nmf_max_iter=100,
        batch_size=128,
        lr=1e-3,
        pretrain=True,
        verbose=False,
        X_val=None,
        y_val=None,
        y_pred_weights_val=None,
        task_mask_val=None,
        best_model_name="dCSFA-NMF-best-model.pt",
    ):
        """
        Fit the model.

        NOTE: isn't ``lr`` a model parameter? epochs and pre epochs could be
        too.

        Parameters
        ----------
        X (np.ndarray): Input Features
            Shape: ``[n_samples, n_features]``
        y (np.ndarray): ground truth labels
            Shape: ``[n_samples, n_sup_networks]``
        y_pred_weights (np.ndarray, optional):
            supervision window specific importance weights. Defaults to
            ``None``.
            Shape: ``[n_samples,1]``
        task_mask (np.ndarray, optional):
            identifies which windows should be trained on which tasks. Defaults
            to ``None``.
            Shape: ``[n_samples,n_sup_networks]``
        intercept_mask (np.ndarray, optional):
            One-hot Mask for group specific intercepts in the logistic
            regression model. Defaults to None.
            Shape: ``[n_samples,n_intercepts]``
        y_sample_groups (_type_, optional):
            groups for creating sample weights - each group will be sampled
            evenly. Defaults to None.
            Shape: ``[n_samples,1]``
        n_epochs (int, optional):
            number of training epochs. Defaults to 100.
        n_pre_epochs (int, optional):
            number of pretraining epochs. Defaults to 100.
        nmf_max_iter (int, optional):
            max iterations for NMF pretraining solver. Defaults to 100.
        batch_size (int, optional):
            batch size for gradient descent. Defaults to 128.
        lr (_type_, optional):
            learning rate for gradient descent. Defaults to 1e-3.
        pretrain (bool, optional):
            whether or not to pretrain the generative model. Defaults to True.
        verbose (bool, optional):
            activate or deactivate print statements. Defaults to False.
        X_val (np.ndarray, optional):
            Validation Features for checkpointing. Defaults to None.
            Shape: ``[n_val_samples,n_features]``
        y_val (np.ndarray, optional):
            Validation Labels for checkpointing. Defaults to None.
            Shape: ``[n_val_samples,n_sup_networks]``
        y_pred_weights_val (np.ndarray, optional):
            window specific classification weights. Defaults to None.
            Shape: ``[n_val_samples,1]``
        task_mask_val (np.ndarray, optional):
            validation task relevant window masking. Defaults to None.
            Shape: ``[n_val_samples,n_sup_networks]``
        best_model_name (str, optional):
            save file name for the best model. Must end in ".pt". Defaults to
            ``'dCSFA-NMF-best-model.pt'``.

        Returns
        -------
        self : dCSFA_NMF
            The fitted model. NOTE: sklearn convention
        """
        # Initialize model parameters.
        self._initialize(X.shape[1])

        # Establish loss histories.
        self.training_hist = []  # tracks average overall loss
        self.recon_hist = []  # tracks training data mse
        self.pred_hist = []  # tracks training data aucs

        # Globaly activate/deactivate print statements.
        self.verbose = verbose
        self.lr = lr

        # Fill default values
        if intercept_mask is None:
            intercept_mask = np.ones((X.shape[0], self.n_intercepts))
        if task_mask is None:
            task_mask = np.ones(y.shape)
        if y_pred_weights is None:
            y_pred_weights = np.ones((y.shape[0], 1))

        # Fill sampler parameters.
        if y_sample_groups is None:
            y_sample_groups = np.ones((y.shape[0]))
            samples_weights = y_sample_groups
        else:
            class_sample_counts = np.array(
                [
                    np.sum(y_sample_groups == group)
                    for group in np.unique(y_sample_groups)
                ],
            )
            weight = 1.0 / class_sample_counts
            samples_weights = np.array(
                [weight[t] for t in y_sample_groups.astype(int)],
            ).squeeze()
            samples_weights = torch.Tensor(samples_weights)

        # Pretrain the model.
        if pretrain:
            self.pretrain_NMF(X, y, nmf_max_iter)
            self.pretrain_encoder(
                X,
                y,
                y_pred_weights,
                task_mask,
                intercept_mask,
                samples_weights,
                n_pre_epochs,
                batch_size,
            )

        # Send training arguments to Tensors.
        X = torch.Tensor(X).float().to(self.device)
        y = torch.Tensor(y).float().to(self.device)
        y_pred_weights = torch.Tensor(y_pred_weights).float().to(self.device)
        task_mask = torch.Tensor(task_mask).long().to(self.device)
        intercept_mask = torch.Tensor(intercept_mask).long().to(self.device)
        samples_weights = torch.Tensor(samples_weights).to(self.device)

        # If validation data is provided, set up the tensors.
        if X_val is not None and y_val is not None:
            assert best_model_name.split(".")[-1] == "pt", (
                f"Save file `{self.save_folder + best_model_name}` must be "
                f"of type .pt"
            )
            self.best_model_name = best_model_name
            self.best_val_recon = 1e8
            self.best_val_avg_auc = 0.0
            self.val_recon_hist = []
            self.val_pred_hist = []

            if task_mask_val is None:
                task_mask_val = np.ones(y_val.shape)

            if y_pred_weights_val is None:
                y_pred_weights_val = np.ones((y_val[:, 0].shape[0], 1))

            X_val = torch.Tensor(X_val).float().to(self.device)
            y_val = torch.Tensor(y_val).float().to(self.device)
            task_mask_val = torch.Tensor(task_mask_val).long().to(self.device)
            y_pred_weights_val = (
                torch.Tensor(
                    y_pred_weights_val,
                )
                .float()
                .to(self.device)
            )

        # Instantiate the dataloader and optimizer.
        dset = TensorDataset(X, y, task_mask, y_pred_weights, intercept_mask)
        sampler = WeightedRandomSampler(samples_weights, len(samples_weights))
        loader = DataLoader(dset, batch_size=batch_size, sampler=sampler)
        optimizer = self.instantiate_optimizer()

        # Define the training iterator.
        if self.verbose:
            print("Beginning Training")
            epoch_iter = tqdm(range(n_epochs))
        else:
            epoch_iter = range(n_epochs)

        # Training loop.
        for epoch in epoch_iter:
            epoch_loss = 0.0
            recon_e_loss = 0.0
            pred_e_loss = 0.0

            for batch in loader:
                self.train()
                optimizer.zero_grad()
                recon_loss, pred_loss = self.forward(*batch)
                # Weighting happens inside of the forward call
                loss = recon_loss + pred_loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                recon_e_loss += recon_loss.item()
                pred_e_loss += pred_loss.item()
            self.training_hist.append(epoch_loss / len(loader))
            with torch.no_grad():
                self.eval()
                X_recon, y_pred, _ = self.transform(
                    X,
                    y,
                    intercept_mask,
                    return_npy=False,
                )
                training_mse_loss = nn.MSELoss()(X_recon, X).item()
                training_auc_list = []
                for sup_net in range(self.n_sup_networks):
                    temp_mask = task_mask[:, sup_net].detach().cpu().numpy()
                    auc = roc_auc_score(
                        y.detach().cpu().numpy()[temp_mask == 1, sup_net],
                        y_pred.detach().cpu().numpy()[temp_mask == 1, sup_net],
                    )
                    training_auc_list.append(auc)
                self.recon_hist.append(training_mse_loss)
                self.pred_hist.append(training_auc_list)

                # If validation data is present, collect performance metrics.
                if X_val is not None and y_val is not None:
                    X_recon_val, y_pred_val, _ = self.transform(
                        X_val,
                        y_val,
                        return_npy=False,
                    )
                    validation_mse_loss = nn.MSELoss()(
                        X_recon_val,
                        X_val,
                    ).item()
                    validation_auc_list = []
                    for sup_net in range(self.n_sup_networks):
                        temp_mask = task_mask_val[:, sup_net].detach().cpu().numpy()
                        auc = roc_auc_score(
                            y_val.detach().cpu().numpy()[temp_mask == 1, sup_net],
                            y_pred_val.detach().cpu().numpy()[temp_mask == 1, sup_net],
                        )
                        validation_auc_list.append(auc)

                    self.val_recon_hist.append(validation_mse_loss)
                    self.val_pred_hist.append(validation_auc_list)

                    cond_1 = validation_mse_loss < self.best_val_recon
                    cond_2 = np.mean(validation_auc_list) > self.best_val_avg_auc

                    if cond_1 and cond_2:
                        self.best_epoch = epoch
                        self.best_val_recon = validation_mse_loss
                        self.best_val_aucs = validation_auc_list
                        # NOTE: os.path.join does this a bit better:
                        torch.save(
                            self.state_dict(),
                            self.save_folder + self.best_model_name,
                        )

                    if self.verbose:
                        # NOTE: this string can be defined in the file header
                        # to save some space.
                        epoch_iter.set_description(
                            "Epoch: {}, Best Epoch: {}, Best Val MSE: {:.6}, Best Val by Window ROC-AUC {}, current MSE: {:.6}, current AUC: {}".format(
                                epoch,
                                self.best_epoch,
                                self.best_val_recon,
                                self.best_val_aucs,
                                validation_mse_loss,
                                validation_auc_list,
                            )
                        )
                else:
                    # NOTE: ssame here
                    epoch_iter.set_description(
                        "Epoch: {}, Current Training MSE: {:.6}, Current Training by Window ROC-AUC: {}".format(
                            epoch, training_mse_loss, training_auc_list
                        )
                    )

        if self.verbose:
            print(
                "Saving the last epoch with training MSE: {:.6} and AUCs:{}".format(
                    training_mse_loss, training_auc_list
                )
            )
        # Last epoch is saved in case the saved 'best model' doesn't make sense
        # NOTE: is this necessary?
        torch.save(
            self.state_dict(),
            self.save_folder + "dCSFA-NMF-last-epoch.pt",
        )

        if X_val is not None and y_val is not None:
            if self.verbose:
                print(
                    "Loaded the best model from Epoch: {} with MSE: {:.6} and AUCs: {}".format(
                        self.best_epoch, self.best_val_recon, self.best_val_aucs
                    )
                )
            self.load_state_dict(torch.load(self.save_folder + self.best_model_name))
        return self

    def load_last_epoch(self):
        """
        Loads model parameters of the last epoch in case the last epoch is
        preferable to the early stop checkpoint conditions. This will be
        visible in the training printout if self.verbose=True
        """
        self.load_state_dict(
            torch.load(self.save_folder + "dCSFA-NMF-last-epoch.pt"),
        )

    def reconstruct(self, X, component=None):
        """
        Gets full or partial reconstruction.

        Parameters
        ----------
        X : numpy.ndarray
            Input Features
            Shape: ``[n_samples,n_features]``
        component : int, optional
            identifies which component to use for reconstruction

        Returns
        -------
        X_recon : numpy.ndarray
            Full recon if component=None, else, recon for network[comopnent]
        """
        X_recon, _, s = self.transform(X)
        if component is not None:
            X_recon = self.get_comp_recon(s, component)
        return X_recon

    def predict_proba(self, X, return_scores=False):
        """
        Returns prediction probabilities

        Parameters
        ----------
        X : numpy.ndarray
            Input Features
            Shape: ``[n_samples,n_features]``
        return_scores (bool, optional):
            Whether or not to include the projections. Defaults to False.

        Returns
        -------
        y_pred_proba (numpy.ndarray): predictions
            Shape: ``[n_samples,n_sup_networks]``
        s (numpy.ndarray): supervised network activation scores
            Shape: ``[n_samples,n_components]
        """
        _, y_pred, s = self.transform(X)

        if return_scores:
            return y_pred, s
        else:
            return y_pred

    def predict(self, X, return_scores=False):
        """
        Return binned predictions.


        Parameters
        ----------
        X : numpy.ndarray
            Input Features
        return_scores : bool, optional
            Whether or not to include the projections. Defaults to ``False``.

        Returns
        -------
        y_pred_proba : numpy.ndarray
            Predictions in {0,1}
            Shape: ``[n_samples,n_sup_networks]``
        s : numpy.ndarray
            supervised network activation scores
            Shape: ``[n_samples,n_components]
        """
        _, y_pred, s = self.transform(X)

        if return_scores:
            return y_pred > 0.5, s
        else:
            return y_pred > 0.5

    def project(self, X):
        """
        Get projections

        Parameters
        ----------
        X : numpy.ndarray
            Input Features
            Shape: ``[n_samples,n_features]
        return_scores : bool, optional
            Whether or not to include the projections. Defaults to False.

        Returns
        -------
        s : numpy.ndarray
            supervised network activation scores
            Shape: ``[n_samples,n_components]
        """
        _, _, s = self.transform(X)
        return s

    def score(self, X, y, groups=None, return_dict=False):
        """
        Gets a list of task AUCs either by group or for all samples. Can return
        a dictionary with AUCs for each group with each group label as a key.

        Parameters
        ----------
        X : numpy.ndarray
            Input Features
            Shape: ``[n_samples,n_features]
        y : numpy.ndarray
            Ground Truth Labels
            Shape: ``[n_samples,n_sup_networks]``
        groups : numpy.ndarray, optional
            per window group assignment labels. Defaults to None.
            Shape: ``[n_samples,1]``
        return_dict : bool, optional
            Whether or not to return a dictionary with values for each group. Defaults to False.

        Returns
        -------
        score_results: numpy.ndarray or dict
            Array or dictionary of results either as the mean performance of all groups,
            the performance of all samples, or a dictionary of results for each group.
        """
        _, y_pred, _ = self.transform(X)
        if groups is not None:
            auc_dict = {}
            for group in np.unique(groups):
                auc_list = []
                for sup_net in range(self.n_sup_networks):
                    auc = roc_auc_score(y[:, sup_net], y_pred[:, sup_net])
                    auc_list.append(auc)
                auc_dict[group] = auc_list
            if return_dict:
                score_results = auc_dict
            else:
                auc_array = np.vstack([auc_dict[key] for key in np.unique(groups)])
                score_results = np.mean(auc_array, axis=0)
        else:
            auc_list = []
            for sup_net in range(self.n_sup_networks):
                auc = roc_auc_score(y[:, sup_net], y_pred[:, sup_net])
                auc_list.append(auc)
            score_results = np.array(auc_list)
        return score_results


if __name__ == "__main__":
    pass


###
