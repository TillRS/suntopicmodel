from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.append("..")
from sun_topicmodel.snmf import SNMF


# Define fixtures or setup functions if necessary
@pytest.fixture()
def sample_data():
    # Example sample data
    return np.random.rand(10, 20)  # Sample data of shape (10, 20)


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
    model = SNMF(sample_data, num_bases=4, random_state=-1)
    with pytest.raises(ValueError):
        model.factorize(niter=10)


def test_initialization_with_invalid_num_bases(sample_data):
    # Test initialization of SNMF instance with invalid num_bases
    model = SNMF(sample_data, num_bases=0)
    with pytest.raises(ValueError):
        model.factorize(niter=10)


def test_save_load(sample_data):
    # Test save and load methods of SNMF instance
    model = SNMF(sample_data, num_bases=4)
    model.factorize(niter=10)
    model.save(filename="test_model.npz")
    loaded_model = SNMF.load(filename="test_model.npz")
    # delete the file
    os.remove("test_model.npz")
    assert np.allclose(model.W, loaded_model.W)
    assert np.allclose(model.H, loaded_model.H)


# define specific test cases
np.random.seed(44)
data_test = np.random.randn(5, 5)
W_init = np.array(
    [
        [0.2, 0.2, 1.2],
        [1.2, 0.2, 0.2],
        [0.2, 1.2, 0.2],
        [0.2, 1.2, 0.2],
        [1.2, 0.2, 0.2],
    ]
)
W_final = np.array(
    [
        [0.26131357, 0.28294198, 1.10650968],
        [1.53183718, 0.48374749, 0.14029453],
        [0.56818264, 1.08108117, 0.07414763],
        [0.05217432, 1.1717574, 0.47405955],
        [0.57225118, 0.02169945, 0.37475691],
    ]
)
H_final = np.array(
    [
        [-1.33936537, 0.78980205, -0.04690679, -0.47313419, 0.55653641],
        [0.25285725, 0.03530502, -0.77982843, 1.25854171, -0.75134136],
        [-0.38705062, 1.01099344, 1.24087478, -1.73020804, -1.26286445],
    ]
)


def test_initialization_with_specific_data():
    model_test = SNMF(data_test, num_bases=3, random_state=44)
    model_test.factorize(niter=0)
    assert np.allclose(model_test.W, W_init)


def test_specfic_data():
    model_test = SNMF(data_test, num_bases=3, random_state=44)
    model_test.factorize(niter=100)
    assert np.allclose(model_test.W, W_final)
    assert np.allclose(model_test.H, H_final)
