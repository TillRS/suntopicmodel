from __future__ import annotations

import numpy as np
import pytest

from sun_topicmodel import PyMFBase

# Create a random number generator
rng = np.random.default_rng(seed=42)


@pytest.fixture()
def sample_data():
    return rng.random((10, 20))  # Sample data of shape (10, 20)


# Define generic test cases
def test_initialization(sample_data):
    # Test initialization of PyMFBase instance
    model = PyMFBase(sample_data, num_bases=5)
    assert model.data.shape == sample_data.shape
    assert model._num_bases == 5
    assert model._data_dimension == 20
    assert model._num_samples == 10


def test_initialization_with_random_state(sample_data):
    # Test initialization of PyMFBase instance with random_state
    model = PyMFBase(sample_data, num_bases=4, random_state=42)
    assert model.random_state == 42


def test_initialization_with_invalid_data():
    # Test initialization of PyMFBase instance with invalid data
    with pytest.raises(ValueError):
        PyMFBase(rng.random((10, 20, 30)), num_bases=4)
