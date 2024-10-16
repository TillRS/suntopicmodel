"""
Base class for all SNMF class
which we use for semi-nonnegative matrix factorization from [1].

Authors: Till R. Saenger, ORFE Princeton

Notes:

This implementation builds directly on an unsupported implementation of
Christian Thurau (https://github.com/pzoccante/pymf/blob/master/pymf/)


[1] Ding, C., Li, T. and Jordan, M.. Convex and Semi-Nonnegative Matrix Factorizations.
IEEE Trans. on Pattern Analysis and Machine Intelligence 32(1), 45-55.
"""

from __future__ import annotations

import logging
import logging.config

import numpy as np

from suntm.utils import setup_logging

__all__ = ["PyMFBase"]
_EPS = np.finfo(float).eps


class PyMFBase:
    """
    PyMF Base Class. Does nothing useful apart from providing some basic methods.
    """

    # some small value
    _EPS = _EPS

    def __init__(self, data, num_bases, random_state=None, **kwargs):
        """
        Initilaize the PyMFBase class.
        data : array_like, shape (_num_samples, _data_dimension)
        num_bases : int, specifies the number of topics to model
        random_state : int, seed for random number generator
        """

        setup_logging(self)

        # set variables
        self.data = data
        # Check if data is a vector (1-dimensional array)
        if self.data.ndim == 1:
            # Convert it to a 2D array
            self.data = self.data.reshape(1, -1)
        self._num_bases = num_bases
        self.random_state = random_state
        self._num_samples, self._data_dimension = self.data.shape

        # initialize W and H
        self._init_w()
        self._init_h()

    def _init_h(self):
        """Initialize H matrix."""
        message = "The method _init_h() must be implemented in the subclass."
        raise NotImplementedError(message)

    def _init_w(self):
        """Initialize W matrix."""
        message = "The method _init_w() must be implemented in the subclass."
        raise NotImplementedError(message)

    def _update_h(self):
        """Update H matrix."""
        message = "The method _update_h() must be implemented in the subclass."
        raise NotImplementedError(message)

    def _update_w(self):
        """Update W matrix."""
        message = "The method _update_w() must be implemented in the subclass."
        raise NotImplementedError(message)

    def _converged(self, i):
        """
        If the optimization of the approximation is below the machine precision,
        return True.

        Parameters
        ----------
            i   : index of the update step

        Returns
        -------
            converged : boolean
        """
        derr = np.abs(self.ferr[i] - self.ferr[i - 1]) / self._num_samples
        return derr < self._EPS

    def factorize(
        self,
        niter=100,
        verbose=False,
        compute_w=True,
        compute_h=True,
        compute_err=False,
        compute_topic_err=False,
        topic_err_tol=10**-2,
    ):
        """Factorize s.t. WH = data

        Parameters
        ----------
        niter : int
                number of iterations.
        verbose : bool
                print some extra information to stdout.
        compute_h : bool
                iteratively update values for H.
        compute_w : bool
                iteratively update values for W.
        compute_err : bool
                compute Frobenius norm |data-WH| after each update and store
                it to .ferr[k]. Can be omitted for speed.
        compute_topic_err : bool
                compute the L2 norm for each row of W and W_old, and get the maximum difference.
                Can be omitted for speed.
        topic_err_tol : float
                tolerance for the maximum difference of the L2 norm of W and W_old before stopping the iteration.

        Updated Values
        --------------
        .W : updated values for W.
        .H : updated values for H.
        .ferr : Frobenius norm |data-WH| for each iteration.
        """

        if verbose:
            self._logger.setLevel(logging.INFO)
        else:
            self._logger.setLevel(logging.ERROR)

        # create W and H if they don't already exist
        # -> any custom initialization to W,H should be done before
        if not hasattr(self, "W"):
            self._init_w()

        if not hasattr(self, "H"):
            self._init_h()

        if compute_err:
            self.ferr = np.zeros(niter)

        if compute_topic_err:
            self.topic_err = np.zeros(niter)

        for i in range(niter):
            if compute_h:
                self._update_h()

            if compute_w:
                W_old = self.W.copy()
                self._update_w()

            if compute_err:
                self.ferr[i] = np.linalg.norm(self.data - np.dot(self.W, self.H), "fro")
                self._logger.info("FN: %s (%s / %s)", self.ferr[i], i + 1, niter)
            if compute_topic_err:
                # Calculate the L2 norm for each row of W and W_old, and get the maximum difference
                # self.topic_err[i] = np.max(np.abs(self.W - W_old))
                self.topic_err[i] = np.max(np.linalg.norm(self.W - W_old, axis=1))
                self._logger.info(
                    "W (Max Doc Update): %s (%s / %s)", self.topic_err[i], i + 1, niter
                )
            else:
                self._logger.info("Iteration: (%s, %s)", i + 1, niter)

            # check if the err is not changing anymore
            if i > 1 and compute_err and self._converged(i):
                self.ferr = self.ferr[:i]
                break

            if i > 1 and compute_topic_err and self.topic_err[i] < topic_err_tol:
                self.topic_err = self.topic_err[:i]
                break
