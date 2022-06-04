"""
A simple grid search cross validation model.

TO DO
-----
* make compatible with BaseModel
"""
__date__ = "December 2021 - March 2022"


import numpy as np
from sklearn.model_selection import StratifiedKFold



class GridSearchCV:

    def __init__(self, model, param_grid, cv=3):
        """
        A simple grid search cross validation model.

        Parameters
        ----------
        model : CpSae or FaSae
            Model
        param_grid : dict
            Maps names of model parameters (strings) to lists of values. These
            are passed to the ``model.set_params`` method.
        cv : int, optional
            Number of folds to estimate the performance of each parameter set.
        """
        self.model = model
        self.param_grid = param_grid
        self.cv = cv


    def fit(self, features, labels, groups, print_freq=5):
        """
        Fit the model to data.

        These parameters are passed to ``BaseModel.fit``.

        Parameters
        ----------
        features : numpy.ndarray
            Shape: ``[b,f,r,r]``
        labels : numpy.ndarray
            Shape: ``[b]``
        groups : numpy.ndarray
            Shape: ``[b]``
        print_freq : int, optional
            Print loss every ``print_freq`` epochs.
        """
        # Split data into folds.
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True)
        skf_groups = 1000*groups + labels

        best_score = -np.inf
        best_params = None

        # For each parameter setting...
        param_names = list(self.param_grid.keys())
        gen = list(product(*[self.param_grid[param] for param in param_names]))
        for param_num, param_setting in enumerate(gen):
            params = dict(zip(param_names, param_setting))
            scores = []
            # For each fold...
            cv_gen = enumerate(skf.split(features, skf_groups))
            for cv_num, (train_idx, test_idx) in cv_gen:
                # Set the parameters, fit the model, and score the model.
                self.model.set_params(**params)
                self.model.fit(
                        features[train_idx],
                        labels[train_idx],
                        groups[train_idx],
                        print_freq=print_freq,
                )
                model_score = self.model.score(
                        features[test_idx],
                        labels[test_idx],
                        groups[test_idx],
                )
                print(f"Param {param_num} cv {cv_num} score {model_score}")
                scores.append(model_score)
            model_score = np.mean(scores)
            print(f"Param {param_num} score {model_score}")
            if model_score > best_score:
                best_score = model_score
                best_params = params
        # Retrain using the best found parameters.
        self.model.set_params(**best_params)
        self.model.fit(features, labels, groups, print_freq=print_freq)
        self.best_estimator_ = self.model
        self.best_params_ = best_params
        self.best_score_ = best_score


    def predict(self, features, groups):
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
        return self.best_estimator_.predict(features, groups)


    def score(self, features, labels, groups):
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
        return self.best_estimator_.score(features, labels, groups)



if __name__ == '__main__':
    pass



###
