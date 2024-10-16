"""
Semi Non-negative Matrix Factorization.

    SNMF(NMF) : Class for semi non-negative matrix factorization

[1] Ding, C., Li, T. and Jordan, M.. Convex and Semi-Nonnegative Matrix Factorizations.
IEEE Trans. on Pattern Analysis and Machine Intelligence 32(1), 45-55.

Authors: Till R. Saenger, ORFE Princeton

Note: This implementation builds directly on an unsupported implementation of
Christian Thurau (https://github.com/pzoccante/pymf/blob/master/pymf/)

"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from suntm.base import PyMFBase

__all__ = ["SNMF"]


class SNMF(PyMFBase):
    """
    SNMF(data, num_bases)

    Semi Nonnegative Matrix Factorization. Factorize a data matrix into two
    matrices s.t. F = | data - W*H | is minimal. For Semi-NMF only H is
    constrained to non-negativity.

    Parameters
    ----------
    data : array_like, shape (_data_dimension, _num_samples)
        the input data
    num_bases: int, optional
        Number of bases to compute (column rank of W and row rank of H).
        4 (default)

    Attributes
    ----------
    W : "data_dimension x num_bases" matrix of basis vectors
    H : "num bases x num_samples" matrix of coefficients
    ferr : frobenius norm (after calling .factorize())

    """

    def _init_h(self):
        """Initialize H to random values [0,1]."""
        self.H = np.zeros((self._num_bases, self._data_dimension))

    def _init_w(self):
        """Initialize W using k-means ++"""
        self.W = np.zeros((self._num_samples, self._num_bases))
        if self.data.shape[0] >= self._num_bases:
            km = KMeans(
                n_clusters=self._num_bases,
                random_state=self.random_state,
                n_init="auto",
                init="k-means++",
            ).fit(self.data)
            assign = km.labels_
            self._logger.info("SNMF - Initial Assignment: %s", assign)

            num_i = np.zeros(self._num_bases)
            for i in range(self._num_bases):
                num_i[i] = len(np.where(assign == i)[0])

            self.W[range(len(assign)), assign] = 1.0
            self.W += np.ones((self._num_samples, self._num_bases)) * 0.2

        else:
            self.W = np.ones((self._num_samples, self._num_bases)) * 0.2

    def _update_h(self):
        H1 = np.dot(self.W.T, self.W)
        H2 = np.dot(self.W.T, self.data)
        self.H = np.linalg.solve(H1, H2)

    def _update_w(self):
        def separate_positive(m):
            return (np.abs(m) + m) / 2.0

        def separate_negative(m):
            return (np.abs(m) - m) / 2.0

        # check dimensionality, for predictions, data will have one column less
        if self.data.shape[1] == self.H.shape[1]:
            XH = np.dot(self.data, self.H.T)
        else:
            XH = np.dot(self.data, self.H[:, :-1].T)

        HH = np.dot(self.H, self.H.T)
        HH_pos = separate_positive(HH)
        HH_neg = separate_negative(HH)

        XH_pos = separate_positive(XH)
        W1 = XH_pos + np.dot(self.W, HH_neg)

        XH_neg = separate_negative(XH)
        W2 = (XH_neg + np.dot(self.W, HH_pos)) + 10**-9

        self.W *= np.sqrt(W1 / W2)
