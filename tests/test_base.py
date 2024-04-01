import numpy as np
import sys
import os
sys.path.append("..")
from sun_topicmodel.base import PyMFBase
import pytest


# Define fixtures or setup functions if necessary
@pytest.fixture
def sample_data():
    # Example sample data
    return np.random.rand(10, 20)  # Sample data of shape (10, 20)

# Define generic test cases
def test_initialization(sample_data):
    # Test initialization of PyMFBase instance
    model = PyMFBase(sample_data, num_bases=5)
    assert model.data.shape == sample_data.shape
    assert model._num_bases == 5
    assert model._data_dimension == 10
    assert model._num_samples == 20

# def test_residual(sample_data):
#     # Test residual method of PyMFBase instance
#     model = PyMFBase(sample_data, num_bases=5)
#     model.factorize(niter=50)
#     residual = model.residual()
#     assert isinstance(residual, float)
#     assert 0 <= residual <= 100

def test_factorize(sample_data):
    # Test factorize method of PyMFBase instance
    model = PyMFBase(sample_data, num_bases=4)
    model.factorize(niter=10)
    assert model.W.shape == (10, 4)
    assert model.H.shape == (4, 20)

def test_save_load(sample_data):
    # Test save and load methods of PyMFBase instance
    model = PyMFBase(sample_data, num_bases=4)
    model.factorize(niter=10)
    model.save(filename="test_model.npz")
    loaded_model = PyMFBase.load(filename="test_model.npz")
    # delete the file
    os.remove("test_model.npz")
    assert np.allclose(model.W, loaded_model.W)
    assert np.allclose(model.H, loaded_model.H)

def test_initialization_with_random_state(sample_data):
    # Test initialization of PyMFBase instance with random_state
    model = PyMFBase(sample_data, num_bases=4, random_state=42)
    assert model.random_state == 42


# Define test cases for exceptions
def test_initialization_with_invalid_random_state(sample_data):
    # Test initialization of PyMFBase instance with invalid random_state
    with pytest.raises(ValueError):
        model = PyMFBase(sample_data, num_bases=4, random_state=-1)
        model.factorize(niter=10)

def test_initialization_with_invalid_data(sample_data):
    # Test initialization of PyMFBase instance with invalid data
    with pytest.raises(ValueError):
        model = PyMFBase(np.random.rand(10, 20, 30), num_bases=4)
        
def test_initialization_with_invalid_num_bases(sample_data):
    # Test initialization of PyMFBase instance with invalid num_bases
    with pytest.raises(ValueError):
        model = PyMFBase(sample_data, num_bases=0)
        model.factorize(niter=10)


# Define test cases for specific data samples
np.random.seed(23)
test_data = np.random.rand(4, 5) 
W_test_data = np.array(
    [0.51739788, 0.9470626,
 0.76555976, 0.28249584,
 0.22114536, 0.68632209,
 0.1672392,  0.39254247]
).reshape(4, 2)

def test_initialization_with_specific_data():
    # Test initialization of PyMFBase instance with specific data
    model = PyMFBase(test_data, num_bases=2, random_state=23)
    assert model.data.shape == test_data.shape
    assert model._num_bases == 2
    assert model._data_dimension == 4
    assert model._num_samples == 5
    model.factorize(niter=100)
    assert model.W.shape == (4, 2)
    assert model.H.shape == (2, 5)
    print(model.W)
    assert np.allclose(model.W, W_test_data)

    