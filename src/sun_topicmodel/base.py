"""
Base class for all PyMF classes
which we use for convex and semi-nonnegative matrix factorization.

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

__all__ = ["PyMFBase"]
_EPS = np.finfo(float).eps


class PyMFBase:
    """
    PyMF Base Class. Does nothing useful apart from poviding some basic methods.
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

        def setup_logging():
            # create logger
            self._logger = logging.getLogger("pymf")

            # add ch to logger
            if len(self._logger.handlers) < 1:
                # create console handler and set level to debug
                ch = logging.StreamHandler()
                ch.setLevel(logging.DEBUG)
                # create formatter
                formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

                # add formatter to ch
                ch.setFormatter(formatter)

                self._logger.addHandler(ch)

        setup_logging()

        # set variables
        self.data = data
        self._num_bases = num_bases
        self.random_state = random_state
        self._num_samples, self._data_dimension = self.data.shape

    def _init_h(self):
        """Overwrite for initializing H."""

    def _init_w(self):
        """Overwrite for initializing W."""

    def _update_h(self):
        """Overwrite for updating H."""

    def _update_w(self):
        """Overwrite for updating W."""

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
        show_progress=False,
        compute_w=True,
        compute_h=True,
        compute_err=True,
    ):
        """Factorize s.t. WH = data

        Parameters
        ----------
        niter : int
                number of iterations.
        show_progress : bool
                print some extra information to stdout.
        compute_h : bool
                iteratively update values for H.
        compute_w : bool
                iteratively update values for W.
        compute_err : bool
                compute Frobenius norm |data-WH| after each update and store
                it to .ferr[k]. Can be omitted for speed.

        Updated Values
        --------------
        .W : updated values for W.
        .H : updated values for H.
        .ferr : Frobenius norm |data-WH| for each iteration.
        """

        if show_progress:
            self._logger.setLevel(logging.INFO)
        else:
            self._logger.setLevel(logging.ERROR)

        # create W and H if they don't already exist
        # -> any custom initialization to W,H should be done before
        if not hasattr(self, "W") and compute_w:
            self._init_w()

        if not hasattr(self, "H") and compute_h:
            self._init_h()

        if compute_err:
            self.ferr = np.zeros(niter)

        for i in range(niter):
            if compute_h:
                self._update_h()

            if compute_w:
                self._update_w()

            if compute_err:
                self.ferr[i] = np.linalg.norm(self.data - np.dot(self.W, self.H), "fro")
                self._logger.info(f"FN: {self.ferr[i]} ({i + 1} / {niter})")
            else:
                self._logger.info(f"Iteration: ({i + 1} , {niter})")

            # check if the err is not changing anymore
            if i > 1 and compute_err and self._converged(i):
                self.ferr = self.ferr[:i]
                break

    # Define a save method to export self.W and self.H
    def save(self, filename):
        """Save the factorization to a file.

        Parameters
        ----------
        filename : str
            name of the file to save to.
        """
        np.savez(filename, W=self.W, H=self.H, ferr=self.ferr)

    @staticmethod
    def load(filename):
        """Load a factorization from a file.

        Parameters
        ----------
        filename : str
            name of the file to load from.
        """
        npzfile = np.load(filename)
        loaded_model = PyMFBase(
            data=npzfile["W"] @ npzfile["H"], num_bases=npzfile["W"].shape[1]
        )
        loaded_model.W = npzfile["W"]
        loaded_model.H = npzfile["H"]
        return loaded_model
