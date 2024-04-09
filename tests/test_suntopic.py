from __future__ import annotations

import os

import numpy as np
import pytest

from sun_topicmodel import suntopic

sample_size = 10
data_dim = 20

@pytest.fixture()
def sample_X():
    return np.random.rand(sample_size, data_dim)

@pytest.fixture()
def sample_Y():
    return np.random.rand(sample_size)

def test_initialization(sample_X, sample_Y, alpha = 0.5, num_bases = 5):
    # Test initialization of suntopic instance
    model = suntopic(sample_Y, sample_X, alpha= alpha, num_bases=num_bases)
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
    assert model.model.W.shape == (sample_size, 4)
    assert model.model.H.shape == (4, data_dim + 1)
    assert (model.get_coefficients() == model.model.H).all()
    assert (model.get_topics() == model.model.W).all()

@pytest.mark.parametrize("random_state", [None,21])
def test_save_load(sample_X, sample_Y, random_state):
    # Test save and load methods of suntopic instance
    model = suntopic(sample_Y, sample_X, alpha= 0.5, num_bases=4, random_state=random_state)
    model.fit(niter=10)
    model.save(filename="test_model.npz")
    loaded_model = suntopic.load(filename="test_model.npz")
    os.remove("test_model.npz")
    assert np.allclose(model.data, loaded_model.data)
    assert model.num_bases == loaded_model.num_bases
    assert model.alpha == loaded_model.alpha
    assert np.allclose(model.Y, loaded_model.Y)
    assert np.allclose(model.X, loaded_model.X)

def test_predict_and_infer(sample_X, sample_Y):
    # Test predict method of suntopic instance
    model = suntopic(sample_Y, sample_X, alpha= 0.5, num_bases=4)
    model.fit(niter=10)
    X_new = np.random.rand(5, data_dim)
    Y_pred = model.predict(X_new)
    assert Y_pred.shape == (5,)
    Y_pred, W_pred = model.predict(X_new, return_topics=True)
    assert Y_pred.shape == (5,)
    assert W_pred.shape == (5,4)

def test_predict_with_invalid_X_new(sample_X, sample_Y):
    # Test predict method of suntopic instance with invalid X_new
    model = suntopic(sample_Y, sample_X, alpha= 0.5, num_bases=4)
    model.fit(niter=10)
    X_new = np.random.rand(5, 21)
    with pytest.raises(ValueError):
        model.predict(X_new)

def test_predict_without_fit(sample_X, sample_Y):
    # Test predict method of suntopic instance without fit
    model = suntopic(sample_Y, sample_X, alpha= 0.5, num_bases=4)
    X_new = np.random.rand(5, 20)
    with pytest.raises(ValueError):
        model.predict(X_new)

def test_predict_with_invalid_return_topics(sample_X, sample_Y):
    # Test predict method of suntopic instance with invalid return_topics
    model = suntopic(sample_Y, sample_X, alpha= 0.5, num_bases=4)
    model.fit(niter=10)
    X_new = np.random.rand(5, 22)
    with pytest.raises(ValueError):
        model.predict(X_new, return_topics=1)


def test_summary(sample_X, sample_Y):
    # Test summary method of suntopic instance
    model = suntopic(sample_Y, sample_X, alpha= 0.5, num_bases=4)
    model.fit(niter=10)
    assert model.summary() is None

def test_summary_without_fit(sample_X, sample_Y):
    # Test summary method of suntopic instance without fit
    model = suntopic(sample_Y, sample_X, alpha= 0.5, num_bases=4)
    with pytest.raises(ValueError):
        model.summary()

def test_hyperparam_cv():
    Y = np.random.rand(100)
    X = np.random.rand(100, 200)
    # Test find_alpha_cv method of suntopic instance
    model = suntopic(Y, X, alpha= 0.5, num_bases=4)
    model.hyperparam_cv(alpha_range=[0.1, 0.5, 0.9], num_bases_range=[2, 4, 6], cv_folds=3, random_state=21, niter=10)
    assert model.cv_errors.shape == (3, 3, 3)
    assert model.cv_alpha == [0.1, 0.5, 0.9]
    assert model.cv_num_bases == [2, 4, 6]
    assert model.cv_folds == 3
    assert ((0<= model.cv_errors) & (model.cv_errors <= 1)).all()
    cv_errors = model.cv_errors

    model_parallel = suntopic(Y, X, alpha= 0.5, num_bases=4)
    model_parallel.hyperparam_cv(alpha_range=[0.1, 0.5, 0.9], num_bases_range=[2, 4, 6], cv_folds=3, random_state=21, niter=10, parallel=True)
    assert model_parallel.cv_errors.shape == (3, 3, 3)
    assert model_parallel.cv_alpha == [0.1, 0.5, 0.9]
    assert model_parallel.cv_num_bases == [2, 4, 6]
    assert model_parallel.cv_folds == 3
    assert ((0<= model_parallel.cv_errors) & (model_parallel.cv_errors <= 1)).all()
    assert cv_errors.all() == model_parallel.cv_errors.all()


def test_cv_summary_without_fit(sample_X, sample_Y):
    # Test summary method of suntopic instance without fit
    model = suntopic(sample_Y, sample_X, alpha= 0.5, num_bases=4)
    with pytest.raises(ValueError):
        model.cv_summary() 