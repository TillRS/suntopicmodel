"""
SUN Topic Model

    suntopic : Class for suntopic model

Authors: Till R. Saenger, ORFE Princeton

[Citation TBD]

"""

from __future__ import annotations

import logging
import logging.config

import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
from tqdm import tqdm
from joblib import Parallel, delayed
from matplotlib import rc
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from sun_topicmodel.snmf import SNMF


class suntopic(SNMF):
    """
    suntopic(data, num_bases)


    """

    def __init__(self, Y, X, alpha, num_bases, random_state=None):
        """
        Initialize the suntopic class.
        data : array_like, shape (_num_samples, _data_dimension)
        num_bases : int, specifies the number of topics to model
        """

        if alpha < 0 or alpha > 1:
            msg = "Alpha must be in [0,1]"
            raise ValueError(msg)
        if num_bases < 1:
            msg = "Number of bases must be at least 1"
            raise ValueError(msg)
        if num_bases > X.shape[1]:
            msg = "Number of bases must be less than the dimensionality of X.shape[1]"
            raise ValueError(msg)
        if num_bases != int(num_bases):
            msg = "Number of bases must be an integer"
            raise ValueError(msg)

        if len(Y) != X.shape[0]:
            msg = "Y and X must have the same number of samples"
            raise ValueError(msg)

        def setup_logging():
            self._logger = logging.getLogger("suntopic")
            # Add console handler and set level to DEBUG
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            # Create formatter
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            # Add formatter to handler
            ch.setFormatter(formatter)
            # Add handler to logger
            self._logger.addHandler(ch)

        setup_logging()

        self.Y = Y
        self.X = X
        self.num_bases = num_bases
        self.alpha = alpha
        self._niter = 0

        data = np.hstack(
            (np.sqrt(alpha) * X, np.sqrt(1 - alpha) * np.array(Y).reshape(-1, 1))
        )
        np.hstack((np.sqrt(alpha) * X, np.sqrt(1 - alpha) * np.array(Y).reshape(-1, 1)))
        self.data = data
        self.model = SNMF(data, num_bases, random_state=random_state)
        self.model.random_state = random_state

    def fit(
        self, niter=100, verbose=False, compute_w=True, compute_h=True, compute_err=True, standardize = True
    ):
        """
        Fit the suntopic model to the data.
        """
        self._niter = niter

        self.model.factorize(
            niter=niter,
            verbose=verbose,
            compute_w=compute_w,
            compute_h=compute_h,
            compute_err=compute_err,
        )

        if standardize:
            S = np.diag(np.std(self.model.W, axis=0))
            self.model.W = np.dot(self.model.W, np.linalg.inv(S))
            self.model.H = np.dot(S, self.model.H)


    def predict(
        self,
        X_new,
        return_topics=False,
        niter=100,
        random_state=None,
        verbose=False,
        compute_err=False,
        compute_topic_err = True,
        topic_err_tol = 1e-3,
        cvxpy = False,
        solver = 'ECOS',
    ):
        """
        Predict the response variable for new data X_new.
        """
        # Reshape singular observation to matrix
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)

        if X_new.shape[1] != self.X.shape[1]:
            msg = "X_new.shape[1] must be equal to X.shape[1]"
            raise ValueError(msg)

        if not hasattr(self.model, "H"):
            msg = "Model has not been fitted yet. Call fit() first."
            raise ValueError(msg)
        
        if solver not in ['ECOS', 'SCS']:
            msg = "Solver must be either 'ECOS' or 'SCS'"
            raise ValueError(msg)

        data_new = np.sqrt(self.alpha) * X_new

        if cvxpy:
            # use cvxpy to predict
            H = self.model.H[:,:-1]
            W = cp.Variable((data_new.shape[0], self.num_bases), nonneg=True)

            objective = cp.Minimize(cp.norm(data_new - W @ H, 'fro'))
            problem = cp.Problem(objective)
            if solver == 'SCS':
                problem.solve(solver=cp.SCS)
            elif solver == 'ECOS':
                problem.solve(solver=cp.ECOS)

            W_pred = W.value
            Y_pred = np.dot(W_pred, self.model.H[:, -1])
        
        else:
            # use SNMF to predict
            self._model_pred = SNMF(data_new, self.num_bases, random_state=random_state)
            self._model_pred.H = self.model.H[:,:-1]

            self._model_pred.factorize(
                niter=niter, 
                verbose=verbose, 
                compute_h=False, 
                compute_err=compute_err, 
                compute_topic_err=compute_topic_err, 
                topic_err_tol=topic_err_tol
            )

            W_pred = self._model_pred.W
            Y_pred = np.dot(W_pred, self.model.H[:, -1])

        Y_pred /= np.sqrt(1 - self.alpha)
        if return_topics is False:
            return Y_pred
        return Y_pred, W_pred

    def get_topics(self):
        """
        Get the topics from the suntopic model.
        """
        return self.model.W

    def get_coefficients(self):
        """
        Get the coefficients from the suntopic model.
        """
        return self.model.H

    def get_top_docs_idx(self, topic, n_docs=10):
        """
        Get the index of the top n documents for a given topic.
        """
        if not hasattr(self.model, "W"):
            msg = "Model has not been fitted yet. Call fit() first."
            raise ValueError(msg)

        if n_docs < 1:
            msg = "Number of iterations must be at least 1"
            raise ValueError(msg)
        if n_docs != int(n_docs):
            msg = "Number of iterations must be an integer"
            raise ValueError(msg)
        if n_docs > self.model.H.shape[1]:
            msg = "Number of top documents must be less than the total number of documents"
            raise ValueError(msg)

        if topic < 0 or topic >= self.model.W.shape[1]:
            msg = "Topic index out of bounds"
            raise ValueError(msg)

        return np.argsort(self.model.W[:, topic])[::-1][:n_docs]

    def summary(self):
        """
        Print a summary of the suntopic model.
        """

        print("Suntopic Model Summary")
        print("=" * 50)
        print("Number of topics: ", self.num_bases)
        print("Alpha: ", self.alpha)
        print("Data shape: ", self.data.shape)
        print("Number of iterations of model fit: ", self._niter)
        print("Random initialization state: ", self.model.random_state)
        print(
            "Frobenius norm error: ",
            np.linalg.norm(self.data - self.model.W @ self.model.H),
        )
        print(
            "In-sample MSE: ",
            mean_squared_error(self.Y, np.dot(self.model.W, self.model.H[:, -1])),
        )
        print("Prediction coefficients: ", self.model.H[:, -1])
        return

    def save(self, filename):
        """
        Save the suntopic model to a file.
        """
        np.savez(
            filename,
            Y=self.Y,
            X=self.X,
            W=self.model.W,
            H=self.model.H,
            alpha=self.alpha,
            random_state=str(self.model.random_state),
        )

    @staticmethod
    def load(filename):
        """
        Load a suntopic model from a file.
        """
        npzfile = np.load(filename)

        if npzfile["random_state"] == "None":
            loaded_model = suntopic(
                Y=npzfile["Y"],
                X=npzfile["X"],
                alpha=npzfile["alpha"],
                num_bases=npzfile["W"].shape[1],
            )
        else:
            loaded_model = suntopic(
                Y=npzfile["Y"],
                X=npzfile["X"],
                alpha=npzfile["alpha"],
                num_bases=npzfile["W"].shape[1],
                random_state=int(npzfile["random_state"]),
            )

        loaded_model.model.W = npzfile["W"]
        loaded_model.model.H = npzfile["H"]
        return loaded_model

    def hyperparam_cv(
        self,
        alpha_range,
        num_bases_range,
        cv_folds,
        random_state=None,
        parallel=False,
        verbose=False,
        niter=100,
        cvxpy=True,
        pred_niter=100,
        compute_topic_err=False,
        topic_err_tol=1e-2
    ):
        """
        Calculate the cross-validated mean squared error for different values of alpha and num_bases.
        """
        for alpha in alpha_range:
            if alpha < 0 or alpha > 1:
                msg = "Alpha must be in [0,1]"
                raise ValueError(msg)

        for num_bases in num_bases_range:
            if num_bases < 1:
                msg = "Each number of bases must be at least 1"
                raise ValueError(msg)
            if num_bases > self.X.shape[1]:
                msg = "Each number of bases must be less than the dimensionality of X.shape[1]"
                raise ValueError(msg)
            if num_bases != int(num_bases):
                msg = "Each number of bases must be an integer"
                raise ValueError(msg)

        if cv_folds < 2:
            msg = "Number of folds must be at least 2"
            raise ValueError(msg)
        if cv_folds != int(cv_folds):
            msg = "Number of folds must be an integer"
            raise ValueError(msg)
        if cv_folds > len(self.Y):
            msg = "Number of folds must be less than the number of samples"
            raise ValueError(msg)

        if verbose:
            self._logger.setLevel(logging.INFO)
        else:
            self._logger.setLevel(logging.ERROR)

        self.cv_alpha_range = alpha_range
        self.cv_num_base_range = num_bases_range
        self.cv_folds = cv_folds
        self.cv_errors = np.full(
            (len(alpha_range), len(num_bases_range), cv_folds), None
        )
        self.cv_random_state = random_state

        self._cv_kf = KFold(
            n_splits=cv_folds, random_state=self.cv_random_state, shuffle=True
        )
        self.cv_errors = (
            np.ones((len(num_bases_range), len(alpha_range), cv_folds)) * np.nan
        )

        def predict_Y_mse(
            self,
            k,
            alpha,
            num_bases,
            train_index,
            test_index,
            random_state=self.cv_random_state,
            niter=niter,
        ):
            model = suntopic(
                Y=self.Y[train_index],
                X=self.X[train_index],
                alpha=alpha,
                num_bases=num_bases,
                random_state=random_state,
            )
            model.fit(niter=niter, verbose=False)
            Y_pred = model.predict(
                self.X[test_index], 
                random_state=random_state, 
                cvxpy=cvxpy,
                niter=pred_niter, 
                compute_err=False, 
                compute_topic_err=compute_topic_err,
                topic_err_tol=topic_err_tol
            )
            mse = mean_squared_error(self.Y[test_index], Y_pred)
            self._logger.info(
                "Alpha: %s, Num bases: %s, Fold: %s, MSE: %s", alpha, num_bases, k, mse
            )
            # currently does not work for parallel
            return mse

        total_iterations = len(alpha_range) * len(num_bases_range) * cv_folds

        if parallel is False:
            with tqdm(total=total_iterations, desc="Cross-Validation Progress") as pbar:
                for i, num_bases in enumerate(num_bases_range):
                    for j, alpha in enumerate(alpha_range):
                        for k, (train_index, test_index) in enumerate(self._cv_kf.split(self.Y)):
                            self.cv_errors[i, j, k] = predict_Y_mse(
                                self, k, alpha, num_bases, train_index, test_index
                            )
                            pbar.update(1)
        else:
            # Sequentially loop over alpha_ranges and parallelize across topic_range
            num_cores = -1 if len(alpha_range) > 1 else 1  # Use all available cores
            results = Parallel(n_jobs=num_cores)(
                delayed(predict_Y_mse)(
                    self, k, alpha, num_bases, train_index, test_index
                )
                for i, num_bases in enumerate(num_bases_range)
                for j, alpha in enumerate(alpha_range)
                for k, (train_index, test_index) in enumerate(self._cv_kf.split(self.Y))
            )
            # Assign results back to cv_errors array
            idx = 0
            with tqdm(total=total_iterations, desc="Cross-Validation Progress") as pbar:
                for i in range(len(num_bases_range)):
                    for j in range(len(alpha_range)):
                        for k in range(cv_folds):
                            self.cv_errors[i, j, k] = results[idx]
                            idx += 1
                            pbar.update(1)

    def cv_summary(self, top_hyperparam_combinations=3):
        """
        Print a summary of the cross-validation runs of suntopic models.
        """

        if hasattr(self, "cv_errors") is False:
            msg = "Cross-validation errors have not been computed yet. Call hyperparam_cv() first."
            raise ValueError(msg)

        mean_cv_errors = np.mean(self.cv_errors, axis=2)
        min_idx = np.unravel_index(
            np.argsort(mean_cv_errors, axis=None), mean_cv_errors.shape
        )

        print("Cross-Validation Summary")
        print("=" * 50)
        print("Alpha candidate values: ", self.cv_alpha_range)
        print("Number of topics: ", self.cv_num_base_range)
        print("Number of folds: ", self.cv_folds)
        print("CV Random state: ", self.cv_random_state)
        print("=" * 50)
        for i in range(top_hyperparam_combinations):
            print(
                f"Top {i+1} hyperparam combinations - num_bases: {self.cv_num_base_range[min_idx[0][i]]:.2f}, alpha: {self.cv_alpha_range[min_idx[1][i]]:.2f}, MSE: {mean_cv_errors[min_idx[0][i], min_idx[1][i]]:.4f}"
            )
        return

    def cv_mse_plot(
        self,
        figsize=(10, 6),
        title="Cross-Validation MSE",
        return_plot=False,
        benchmark=None,
    ):
        """
        Return plot of cross-validation errors.
        """
        if hasattr(self, "cv_errors") is False:
            msg = "Cross-validation errors have not been computed yet. Call hyperparam_cv() first."
            raise ValueError(msg)

        rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
        plt.rcParams["text.usetex"] = True

        mean_cv_errors = np.mean(self.cv_errors, axis=2)

        fig, ax = plt.subplots(figsize=figsize)
        for i, num_bases in enumerate(self.cv_num_base_range):
            ax.plot(
                self.cv_alpha_range,
                mean_cv_errors[i],
                label=f"{num_bases} topics",
                marker="o",
            )
        ax.set_xlabel("alpha")
        ax.set_ylabel("MSE")
        ax.set_title(title)
        ax.legend()
        if benchmark is not None:
            ax.axhline(y=benchmark, color="red", linestyle="--", label="Benchmark")
            ax.legend()
        if return_plot:
            return fig
        plt.show()
        return None