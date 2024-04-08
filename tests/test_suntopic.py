from __future__ import annotations

import os

import numpy as np
import pytest

from sun_topicmodel import suntopic


@pytest.fixture()
def sample_X():
    return np.random.rand(10, 20)

@pytest.fixture()
def sample_Y():
    return np.random.rand(10)

def test_initialization(sample_X, sample_Y, alpha = 0.5, num_bases = 5):
    # Test initialization of suntopic instance
    model = suntopic(sample_Y, sample_X, alpha= alpha, num_bases=5)
    assert np.allclose(model.data, np.hstack((np.sqrt(alpha) * sample_X, 
                                          np.sqrt(1 - alpha) * np.array(sample_Y).reshape(-1, 1))))
    assert model.num_bases == 5
    assert model.alpha == 0.5
    assert model.Y.shape == sample_Y.shape
    assert model.X.shape == sample_X.shape

@pytest.mark.parametrize("alpha", [-0.1, 2])
def test_initialization_with_invalid_alpha(sample_X, sample_Y, alpha):
    # Test initialization of suntopic instance with invalid alpha
    with pytest.raises(ValueError):
        suntopic(sample_Y, sample_X, alpha=alpha, num_bases=5)

@pytest.mark.parametrize("num_bases", [0, 21, 10.5])
def test_initialization_with_invalid_num_bases(sample_X, sample_Y, num_bases):
    # Test initialization of suntopic instance with invalid num_bases
    with pytest.raises(ValueError):
        suntopic(sample_Y, sample_X, alpha=0.5, num_bases=num_bases)

def test_initialization_with_invalid_num_samples(sample_X, sample_Y):
    # Test initialization of suntopic instance with invalid num_samples
    with pytest.raises(ValueError):
        suntopic(sample_Y, sample_X[:5], alpha= 0.5, num_bases=5)   

def test_fit(sample_X, sample_Y):
    # Test fit method of suntopic instance
    model = suntopic(sample_Y, sample_X, alpha= 0.5, num_bases=4)
    model.fit(niter=10)
    assert model.model.W.shape == (10, 4)
    assert model.model.H.shape == (4, 21)

@pytest.mark.parametrize("random_state", [None,21])
def test_save_load(sample_X, sample_Y, random_state):
    # Test save and load methods of suntopic instance
    model = suntopic(sample_Y, sample_X, alpha= 0.5, num_bases=4, random_state=random_state)
    model.fit(niter=10)
    model.save(filename="test_model.npz")
    loaded_model = suntopic.load(filename="test_model.npz")
    assert np.allclose(model.data, loaded_model.data)
    assert model.num_bases == loaded_model.num_bases
    assert model.alpha == loaded_model.alpha
    assert np.allclose(model.Y, loaded_model.Y)
    assert np.allclose(model.X, loaded_model.X)




# def test_predict(sample_X, sample_Y):
#     # Test predict method of suntopic instance
#     model = suntopic(sample_Y, sample_X, alpha= 0.5, num_bases=4)
#     model.fit(niter=10)
#     X_new = np.random.rand(5, 20)
#     Y_pred = model.predict(X_new)
#     assert Y_pred.shape == (5,)
