"""
SUN Topic Model

    SunTopic : Class for SunTopic model

Authors: Till R. Saenger, ORFE Princeton

This class implements the SUN Topic Model, a semi-supervised topic model that incorporates a response variable into the topic modeling process.
This model is first introduced in "AutoPersuade: A Framework for Evaluating and Explaining Persuasive Arguments".

The model is based on the Convex and Semi-Nonnegative Matrix Factorization (SNMF) algorithm by Ding et al. [1] and its implementation in the PyMF library [2].
"""

from __future__ import annotations

import logging
import logging.config

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib import rc
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm

from suntm.snmf import SNMF
from suntm.utils import setup_logging


class SunTopic(SNMF):
    """
    SunTopic Class for the SUN Topic Model.

    """

    def __init__(self, Y, X, alpha, num_bases, random_state=None):
        """
        Initialize the Suntopic class.

        Parameters
        ----------
        X : np.ndarray
            Input data array of shape (n, d) where `n` is the number of samples and
            `d` is the dimension of each data sample.
        Y : np.ndarray
            Response variable array of shape (n,) where `n` is the number of samples.
        alpha : float
            Specifies the weight of the response variable. Must be a float in the range (0, 1).
        num_bases : int
            Specifies the number of topics to model.
        random_state : int
            Seed for the random number generator, to ensure reproducibility.
        """

        if alpha < 0 or alpha > 1:
            message = "Alpha must be in [0,1]"
            raise ValueError(message)
        if num_bases < 1:
            message = "Number of bases must be at least 1"
            raise ValueError(message)
        if num_bases > X.shape[1]:
            message = (
                "Number of bases must be less than the dimensionality of X.shape[1]"
            )
            raise ValueError(message)
        if num_bases != int(num_bases):
            message = "Number of bases must be an integer"
            raise ValueError(message)

        if len(Y) != X.shape[0]:
            message = "Y and X must have the same number of samples"
            raise ValueError(message)

        setup_logging(self)

        self.Y = Y
        self.X = X
        self.num_bases = num_bases
        self.alpha = alpha
        self._niter = 0
        self.cv_values = {
            "alpha_range": None,
            "num_base_range": None,
            "folds": None,
            "errors": None,
            "kf": None,
            "random_state": random_state,
        }

        X_scaled = np.sqrt(alpha) * X  # Scaling matrix X by the square root of alpha
        Y_scaled = np.sqrt(1 - alpha) * np.array(Y).reshape(
            -1, 1
        )  # Scaling and reshaping Y
        data = np.hstack((X_scaled, Y_scaled))
        self.data = data
        self.model = SNMF(data, num_bases, random_state=random_state)
        self.model.random_state = random_state

    def fit(
        self,
        niter=100,
        verbose=False,
        compute_w=True,
        compute_h=True,
        compute_err=True,
        standardize=True,
    ):
        """
        Fit the SunTopic model to the data.

        Parameters
        ----------
        niter : int
            Number of iterations for the model fitting.
        verbose : bool
            Whether to print progress messages.
        compute_w : bool
            Whether to update the W matrix.
        compute_h : bool
            Whether to update the H matrix.
        compute_err : bool
            Whether to compute the Frobenius norm error.
        standardize : bool
            Whether to standardize the W and H matrices after fitting.
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
        compute_topic_err=False,
        topic_err_tol=1e-3,
        cvxpy=False,
        solver="SCS",
    ):
        """
        Predict the response variable for new data X_new.

        Parameters
        ----------
        X_new : np.ndarray
            New data array of shape (n, d) where `n` is the number of samples and
            `d` is the dimension of each data sample.
        return_topics : bool
            Whether to return the topics.
        niter : int
            Number of iterations for the prediction.
        random_state : int
            Seed for the random initialization of the inferred W.
        verbose : bool
            Whether to print progress messages.
        compute_err : bool
            Whether to compute the Frobenius norm error.
        compute_topic_err : bool
            Whether to compute the topic error.
        topic_err_tol : float
            Early stopping tolerance for topic error.
        cvxpy : bool
            Whether to use cvxpy for prediction instead of closed form updating steps.
        solver : str
            Solver to use for cvxpy optimization. Must be either 'ECOS' or 'SCS'.
        """
        # Reshape singular observation to matrix
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)

        if X_new.shape[1] != self.X.shape[1]:
            message = "X_new.shape[1] must be equal to X.shape[1]"
            raise ValueError(message)

        if not hasattr(self.model, "H"):
            message = "Model has not been fitted yet. Call fit() first."
            raise ValueError(message)

        if solver not in ["ECOS", "SCS"]:
            message = "Solver must be either 'ECOS' or 'SCS'"
            raise ValueError(message)

        data_new = np.sqrt(self.alpha) * X_new

        if cvxpy:
            # use cvxpy to predict
            H = self.model.H[:, :-1]
            W = cp.Variable((data_new.shape[0], self.num_bases), nonneg=True)

            objective = cp.Minimize(cp.norm(data_new - W @ H, "fro"))
            problem = cp.Problem(objective)
            if solver == "SCS":
                problem.solve(solver=cp.SCS)
            elif solver == "ECOS":
                problem.solve(solver=cp.ECOS)

            W_pred = W.value
            Y_pred = np.dot(W_pred, self.model.H[:, -1])

        else:
            # use SNMF to predict
            self._model_pred = SNMF(data_new, self.num_bases, random_state=random_state)
            self._model_pred.H = self.model.H[:, :-1]

            self._model_pred.factorize(
                niter=niter,
                verbose=verbose,
                compute_h=False,
                compute_err=compute_err,
                compute_topic_err=compute_topic_err,
                topic_err_tol=topic_err_tol,
            )

            W_pred = self._model_pred.W
            Y_pred = np.dot(W_pred, self.model.H[:, -1])

        Y_pred /= np.sqrt(1 - self.alpha)
        if return_topics is False:
            return Y_pred
        return Y_pred, W_pred

    def get_topics(self):
        """
        Get the topics from the SunTopic model.

        Returns
        -------
        np.ndarray
            The topics matrix W.
        """
        return self.model.W

    def get_coefficients(self):
        """
        Get the coefficients from the SunTopic model.

        Returns
        -------
        np.ndarray
            The coefficients matrix H.
        """
        return self.model.H

    def get_top_docs_idx(self, topic, n_docs=10):
        """
        Get the index of the top n documents for a given topic.

        Parameters
        ----------
        topic : int
            Index of the topic.
        n_docs : int
            Number of top documents to return.

        Returns
        -------
        np.ndarray
            Array of indices of the top documents.
        """
        if not hasattr(self.model, "W"):
            message = "Model has not been fitted yet. Call fit() first."
            raise ValueError(message)

        if n_docs < 1:
            message = "Number of top documents must be at least 1"
            raise ValueError(message)
        if n_docs != int(n_docs):
            message = "Number of top documents must be an integer"
            raise ValueError(message)
        if n_docs > self.model.H.shape[1]:
            message = "Number of top documents must be less than the total number of documents"
            raise ValueError(message)

        if topic < 0 or topic >= self.model.W.shape[1]:
            message = "Topic index out of bounds"
            raise ValueError(message)

        topic_weights = self.model.W[:, topic]
        top_indices_desc = np.argsort(topic_weights)[::-1]
        return top_indices_desc[:n_docs]

    def summary(self):
        """
        Print a summary of the SunTopic model.
        """
        predicted_values = np.dot(self.model.W, self.model.H[:, -1]) / np.sqrt(
            1 - self.alpha
        )
        in_sample_mse = mean_squared_error(self.Y, predicted_values)

        print(
            f"""
            SunTopic Model Summary
            {'=' * 50}
            Number of topics: {self.num_bases}
            Alpha: {self.alpha}
            Data shape: {self.data.shape}
            Number of iterations of model fit: {self._niter}
            Random initialization state: {self.model.random_state}
            Frobenius norm error: {np.linalg.norm(self.data - self.model.W @ self.model.H)}
            In-sample MSE: {in_sample_mse}
            Prediction coefficients: {self.model.H[:, -1]}
            """  # noqa: T201
        )

    def save(self, filename):
        """
        Save the SunTopic model to an npz file.

        Parameters
        ----------
        filename : str
            Name of the file to save the model.

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
        Load a SunTopic model from an npz file.

        Parameters
        ----------
        filename : str
            Name of the file to load the model from.

        """
        npzfile = np.load(filename)

        if npzfile["random_state"] == "None":
            loaded_model = SunTopic(
                Y=npzfile["Y"],
                X=npzfile["X"],
                alpha=npzfile["alpha"],
                num_bases=npzfile["W"].shape[1],
            )
        else:
            loaded_model = SunTopic(
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
        topic_err_tol=1e-2,
    ):
        """
        Perform cross-validation for different alpha and num_bases values.

        Parameters
        ----------
        alpha_range : list
            Range of alpha values for cross-validation.
        num_bases_range : list
            Range of number of bases for cross-validation.
        cv_folds : int
            Number of folds for cross-validation.
        random_state : int, optional
            Seed for the random number generator.
        parallel : bool, optional
            Whether to parallelize the cross-validation process.
        verbose : bool, optional
            Whether to print progress messages.
        niter : int, optional
            Number of iterations for model fitting.
        cvxpy : bool, optional
            Whether to use cvxpy for prediction.
        pred_niter : int, optional
            Number of iterations for predictio, used when cvxpy is False.
        compute_topic_err : bool, optional
            Whether to compute topic error.
        topic_err_tol : float, optional
            Early stopping tolerance for topic error, used when compute_topic_err is True.
        """
        for alpha in alpha_range:
            if alpha < 0 or alpha > 1:
                message = "Alpha must be in [0,1]"
                raise ValueError(message)

        for num_bases in num_bases_range:
            if (
                num_bases < 1
                or num_bases > self.X.shape[1]
                or num_bases != int(num_bases)
            ):
                message = (
                    "Each number of bases must be an integer between 1 and X.shape[1]"
                )
                raise ValueError(message)

        if cv_folds < 2 or cv_folds != int(cv_folds) or cv_folds > len(self.Y):
            message = (
                "Number of folds must be an integer between 2 and the number of samples"
            )
            raise ValueError(message)

        if verbose:
            self._logger.setLevel(logging.INFO)
        else:
            self._logger.setLevel(logging.ERROR)

        self.cv_values.update(
            {
                "alpha_range": alpha_range,
                "num_base_range": num_bases_range,
                "folds": cv_folds,
                "random_state": random_state,
            }
        )

        self.cv_values["kf"] = KFold(
            n_splits=cv_folds, random_state=self.cv_values["random_state"], shuffle=True
        )
        self.cv_values["errors"] = (
            np.ones((len(num_bases_range), len(alpha_range), cv_folds)) * np.nan
        )

        total_iterations = len(alpha_range) * len(num_bases_range) * cv_folds

        if parallel is False:
            with tqdm(total=total_iterations, desc="Cross-Validation Progress") as pbar:
                for i, num_bases in enumerate(num_bases_range):
                    for j, alpha in enumerate(alpha_range):
                        for k, (train_index, test_index) in enumerate(
                            self.cv_values["kf"].split(self.Y)
                        ):
                            self.cv_values["errors"][i, j, k] = _predict_Y_mse(
                                self.Y,
                                self.X,
                                self.cv_values["random_state"],
                                alpha=alpha,
                                num_bases=num_bases,
                                train_index=train_index,
                                test_index=test_index,
                                niter=niter,
                                pred_niter=pred_niter,
                                cvxpy=cvxpy,
                                compute_topic_err=compute_topic_err,
                                topic_err_tol=topic_err_tol,
                            )
                            pbar.update(1)
        else:
            num_cores = (
                -1 if len(alpha_range) > 1 else 1
            )  # Use all available cores if more than 1 alpha

            try:
                with tqdm(
                    total=total_iterations, desc="Cross-Validation Progress"
                ) as pbar:
                    results = Parallel(n_jobs=num_cores)(
                        delayed(_predict_Y_mse)(
                            self.Y,
                            self.X,
                            self.cv_values["random_state"],
                            alpha=alpha,
                            num_bases=num_bases,
                            train_index=train_index,
                            test_index=test_index,
                            niter=niter,
                            pred_niter=pred_niter,
                            cvxpy=cvxpy,
                            compute_topic_err=compute_topic_err,
                            topic_err_tol=topic_err_tol,
                        )
                        for i, num_bases in enumerate(num_bases_range)
                        for j, alpha in enumerate(alpha_range)
                        for k, (train_index, test_index) in enumerate(
                            self.cv_values["kf"].split(self.Y)
                        )
                    )
            except Exception as e:
                message = f"Parallel execution failed: {e}"
                raise RuntimeError(message)

            # Ensure results is properly assigned before accessing it
            if results:
                # Assign results back to cv_values["errors"] array
                idx = 0
                for i in range(len(num_bases_range)):
                    for j in range(len(alpha_range)):
                        for k in range(cv_folds):
                            self.cv_values["errors"][i, j, k] = results[idx]
                            idx += 1
            else:
                message = "No results returned from Parallel execution"
                raise RuntimeError(message)

    def cv_summary(self, top_hyperparam_combinations=3):
        """
        Print a summary of the cross-validation runs of SunTopic models.

        Parameters
        ----------
        top_hyperparam_combinations : int
            Number of top hyperparameter combinations to print.
        """

        if self.cv_values["random_state"] is None:
            message = "Cross-validation errors have not been computed yet. \
Call hyperparam_cv() first."
            raise ValueError(message)

        mean_cv_errors = np.mean(self.cv_values["errors"], axis=2)
        min_idx = np.unravel_index(
            np.argsort(mean_cv_errors, axis=None), mean_cv_errors.shape
        )

        print(
            f"""
            Cross-Validation Summary
            {'=' * 50}
            Alpha candidate values: {self.cv_values["alpha_range"]}
            Number of topics: {self.cv_values["num_base_range"]}
            Number of folds: {self.cv_values["folds"]}
            CV Random state: {self.cv_values["random_state"]}
            {'=' * 50}
            """  # noqa: T201
        )

        for i in range(top_hyperparam_combinations):
            print(
                f"Top {i+1} hyperparam combinations - num_bases: {self.cv_values['num_base_range'][min_idx[0][i]]:.2f},\
alpha: {self.cv_values['alpha_range'][min_idx[1][i]]:.2f}, \
MSE: {mean_cv_errors[min_idx[0][i], min_idx[1][i]]:.4f}"
            )  # noqa: T201

    def cv_mse_plot(
        self,
        figsize=(10, 6),
        title="Cross-Validation MSE",
        return_plot=False,
        benchmark=None,
    ):
        """
        Return plot of cross-validation errors.

        Parameters
        ----------
        figsize : tuple
            Figure size.
        title : str
            Title of the plot.
        return_plot : bool
            Whether to return the plot.
        benchmark : float
            Benchmark value to plot as a horizontal dashed line.

        """
        if self.cv_values["random_state"] is None:
            message = "Cross-validation errors have not been computed yet. \
Call hyperparam_cv() first."
            raise ValueError(message)

        rc("font", family="serif", serif=["Computer Modern"])
        plt.rcParams["text.usetex"] = True

        mean_cv_errors = np.mean(self.cv_values["errors"], axis=2)

        fig, ax = plt.subplots(figsize=figsize)
        for i, num_bases in enumerate(self.cv_values["num_base_range"]):
            ax.plot(
                self.cv_values["alpha_range"],
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


def _predict_Y_mse(
    Y,
    X,
    cv_random_state,
    alpha,
    num_bases,
    train_index,
    test_index,
    niter,
    pred_niter,
    cvxpy,
    compute_topic_err,
    topic_err_tol,
):
    """
    Fuction for cross-validation predictions as part of the hyperparam_cv method.
    Implemented separately to allow for parallel execution.
    Predict the response variable and compute the mean squared error.

    Parameters
    ----------
    Y : np.ndarray
        Response variable array of shape (n,) where `n` is the number of samples.
    X : np.ndarray
        Input data array of shape (n, d) where `n` is the number of samples and
        `d` is the dimension of each data sample.
    cv_random_state : int
        Seed for the random number generator.
    alpha : float
        Specifies the weight of the response variable. Must be a float in the range (0, 1).
    num_bases : int
        Specifies the number of topics to model.
    train_index : np.ndarray
        Indices of the training data.
    test_index : np.ndarray
        Indices of the test data.
    niter : int
        Number of iterations for the model fitting.
    pred_niter : int
        Number of iterations for the prediction.
    cvxpy : bool
        Whether to use cvxpy for prediction.
    compute_topic_err : bool
        Whether to compute the topic error.
    topic_err_tol : float
        Early stopping tolerance for topic error.

    Returns
    -------
    float
        Mean squared error of the prediction.
    """
    model = SunTopic(
        Y=Y[train_index],
        X=X[train_index],
        alpha=alpha,
        num_bases=num_bases,
        random_state=cv_random_state,
    )
    model.fit(niter=niter, verbose=False, compute_err=False)
    Y_pred = model.predict(
        X[test_index],
        random_state=cv_random_state,
        cvxpy=cvxpy,
        niter=pred_niter,
        compute_err=False,
        compute_topic_err=compute_topic_err,
        topic_err_tol=topic_err_tol,
    )
    return mean_squared_error(Y[test_index], Y_pred)
