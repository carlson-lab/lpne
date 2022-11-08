"""
A simple grid search cross validation model.

"""
__date__ = "December 2021 - November 2022"


from itertools import product
import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupShuffleSplit
from sklearn.utils.validation import check_is_fitted
import torch


FIT_ATTRIBUTES = ["best_estimator_", "best_params_", "best_score_"]


class GridSearchCV:
    """
    A simple grid search cross validation model.

    Parameters
    ----------
    model : BaseModel
        Model
    param_grid : dict
        Maps names of model parameters (strings) to lists of values. These
        are passed to the ``model.set_params`` method.
    cv : int, optional
        Number of folds to estimate the performance of each parameter set.
    test_size : int, optional
        Passed to ``GroupShuffleSplit``. Defaults to ``2``.
    cv_seed : int, optional
        Passed to ``GroupedShuffleSplit`` or ``StratifiedKFold``. Defaults to ``42``.
    training_seed : int, optional
        Passed to ``BaseModel.fit``. Defaults to ``42``.
    """

    def __init__(
        self, model, param_grid, cv=3, test_size=2, cv_seed=42, training_seed=42
    ):
        # NOTE: HERE!
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.test_size = test_size
        self.cv_seed = cv_seed
        self.training_seed = training_seed

    def fit(self, features, labels, groups, print_freq=None, score_freq=None):
        """
        Fit the model to data.

        These parameters are passed to ``BaseModel.fit``.

        If ``groups`` is ``None``, a label-stratified K-fold is use for model
        selection. Otherwise, a group shuffle used.

        Parameters
        ----------
        features : numpy.ndarray
            Shape: ``[b,f,r,r]``
        labels : numpy.ndarray
            Shape: ``[b]``
        groups : None or numpy.ndarray
            Shape: ``[b]``
        print_freq : int or None, optional
            Print loss every ``print_freq`` epochs.
        score_freq : int or None, optional
            Print weighted accuracy every ``score_freq`` epochs.
        """
        # Split data into folds.
        if groups is None:
            groups = np.zeros(len(features))
            # Balance folds by class.
            def get_cv_gen():
                skf = StratifiedKFold(n_splits=self.cv, random_state=self.cv_seed)
                return enumerate(skf.split(features, labels))

        else:
            # Make sure groups aren't in multiple folds.
            def get_cv_gen():
                skf = GroupShuffleSplit(
                    test_size=self.test_size,
                    n_splits=self.cv,
                    random_state=self.cv_seed,
                )
                return enumerate(skf.split(features, labels, groups))

        best_score = -np.inf
        best_params = None
        # For each parameter setting...
        param_names = sorted(list(self.param_grid.keys()))
        gen = list(product(*[self.param_grid[param] for param in param_names]))
        for param_num, param_setting in enumerate(gen):
            params = dict(zip(param_names, param_setting))
            scores = []
            print("Parameter setting:", params)
            # For each fold...
            for cv_num, (train_idx, test_idx) in get_cv_gen():
                # Set the parameters, fit the model, and score the model.
                self.model.set_params(**params)
                self.model.fit(
                    features[train_idx],
                    labels[train_idx],
                    groups[train_idx],
                    print_freq=print_freq,
                    score_freq=score_freq,
                    random_state=self.training_seed,
                )
                model_score = self.model.score(
                    features[test_idx],
                    labels[test_idx],
                    groups[test_idx],
                    warn=False,
                )
                print(
                    f"Param {param_num} cv {cv_num} score {model_score:.4f}",
                    flush=True,
                )
                scores.append(model_score)
            model_score = np.mean(scores)
            print(f"Param {param_num} score {model_score:.4f}", flush=True)
            if model_score > best_score:
                best_score = model_score
                best_params = params
        # Retrain using the best found parameters.
        self.model.set_params(**best_params)
        self.model.fit(
            features,
            labels,
            groups,
            print_freq=print_freq,
            score_freq=score_freq,
        )
        self.best_estimator_ = self.model
        self.best_params_ = best_params
        self.best_score_ = best_score

    def predict(self, features, groups, *args, **kwargs):
        """
        Predict labels using the best found estimator.

        Parameters
        ----------
        features : numpy.ndarray
            Shape: ``[b,f,r,r]``
        groups : None or numpy.ndarray
            Shape: ``[b]``

        Returns
        -------
        predicted_labels : numpy.ndarray
            Shape: ``[b]``
        """
        check_is_fitted(self, attributes=FIT_ATTRIBUTES)
        return self.best_estimator_.predict(features, groups, *args, **kwargs)

    def score(self, features, labels, groups, *args, **kwargs):
        """
        Score the predictions using the best found estimator.

        Parameters
        ----------
        features : numpy.ndarray
            Shape: ``[b,f,r,r]``
        labels : numpy.ndarray
            Shape: ``[b]``
        groups : None or numpy.ndarray
            Shape: ``[b]``

        Returns
        -------
        score : float
            Weighted accuracy
        """
        check_is_fitted(self, attributes=FIT_ATTRIBUTES)
        return self.best_estimator_.score(
            features,
            labels,
            groups,
            *args,
            **kwargs,
        )

    @torch.no_grad()
    def save_state(self, fn):
        """Save parameters for the best estimator."""
        check_is_fitted(self, attributes=FIT_ATTRIBUTES)
        params = self.best_estimator_.get_params(deep=True)
        np.save(fn, params)


if __name__ == "__main__":
    pass


###
