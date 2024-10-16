from __future__ import annotations

import numpy as np
import pytest

from suntm.base import PyMFBase

# Create a random number generator
rng = np.random.default_rng(seed=42)


@pytest.fixture()
def sample_data():
    return rng.random((10, 20))  # Sample data of shape (10, 20)


# Define generic test cases
def test_initialization(sample_data):
    # Test initialization of PyMFBase instance
    with pytest.raises(NotImplementedError):
        PyMFBase(sample_data, num_bases=4)
