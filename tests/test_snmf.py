from __future__ import annotations

import numpy as np
import pytest
from sklearn.utils._param_validation import InvalidParameterError

from suntm.snmf import SNMF

# Create a random number generator
rng = np.random.default_rng(seed=42)


@pytest.fixture()
def sample_data():
    return rng.random((10, 20))  # Sample data of shape (10, 20)


# Define generic test cases
def test_initialization(sample_data):
    # Test initialization of SNMF instance
    model = SNMF(sample_data, num_bases=5)
    assert model.data.shape == sample_data.shape
    assert model._num_bases == 5
    assert model._data_dimension == 20
    assert model._num_samples == 10


def test_factorize(sample_data):
    # Test factorize method of SNMF instance
    model = SNMF(sample_data, num_bases=4)
    model.factorize(niter=10)
    assert model.W.shape == (10, 4)
    assert model.H.shape == (4, 20)


def test_initialization_with_random_state(sample_data):
    # Test initialization of SNMF instance with random_state
    model = SNMF(sample_data, num_bases=4, random_state=42)
    assert model.random_state == 42


def test_initialization_with_invalid_random_state(sample_data):
    # Test initialization of SNMF instance with invalid random_state
    with pytest.raises(InvalidParameterError):
        SNMF(sample_data, num_bases=4, random_state=-1)


def test_initialization_with_invalid_num_bases(sample_data):
    # Test initialization of SNMF instance with invalid num_bases
    with pytest.raises(InvalidParameterError):
        SNMF(sample_data, num_bases=0)


# define specific test cases
rng = np.random.default_rng(seed=42)
data_test = rng.standard_normal((5, 5))
W_init = np.array(
    [
        [1.2, 0.2, 0.2],
        [0.2, 1.2, 0.2],
        [0.2, 0.2, 1.2],
        [0.2, 1.2, 0.2],
        [1.2, 0.2, 0.2],
    ]
)
W_final = np.array(
    [
        [1.52717716, 0.4118178, 0.41608436],
        [0.38007206, 1.13281955, 0.08401745],
        [0.20492652, 0.18779632, 1.05447671],
        [0.119735, 1.237688, 0.64700313],
        [0.72035853, 0.0406315, 0.04210814],
    ]
)
H_final = np.array(
    [
        [0.15267809, -0.90650513, 0.93239813, 0.18052979, -1.23454713],
        [-1.24915809, 0.16639575, -0.73022645, -0.00441065, -0.35860456],
        [1.04916613, 0.75847681, -0.11758287, 1.14149388, 0.72544398],
    ]
)


def test_initialization_with_specific_data():
    model_test = SNMF(data_test, num_bases=3, random_state=44, compute_err=True)
    assert np.allclose(model_test.W, W_init)


def test_specfic_data():
    model_test = SNMF(data_test, num_bases=3, random_state=44, compute_err=True)
    model_test.factorize(niter=10)
    assert np.allclose(model_test.W, W_final)
    assert np.allclose(model_test.H, H_final)
