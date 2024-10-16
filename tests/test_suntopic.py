from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from suntm.suntopic import SunTopic

sample_size = 10
data_dim = 20

# Create a random number generator
rng = np.random.default_rng(seed=42)


@pytest.fixture()
def sample_X():
    return rng.random((sample_size, data_dim))  # Sample data of shape (10, 20)


@pytest.fixture()
def sample_Y():
    return rng.random(sample_size)


def test_initialization(sample_X, sample_Y, alpha=0.5, num_bases=5):
    # Test initialization of SunTopic instance
    model = SunTopic(sample_Y, sample_X, alpha=alpha, num_bases=num_bases)
    assert np.allclose(
        model.data,
        np.hstack(
            (
                np.sqrt(alpha) * sample_X,
                np.sqrt(1 - alpha) * np.array(sample_Y).reshape(-1, 1),
            )
        ),
    )
    assert model.num_bases == 5
    assert model.alpha == 0.5
    assert model.Y.shape == sample_Y.shape
    assert model.X.shape == sample_X.shape


@pytest.mark.parametrize("alpha", [-0.1, 2])
def test_initialization_with_invalid_alpha(sample_X, sample_Y, alpha):
    # Test initialization of SunTopic instance with invalid alpha
    with pytest.raises(ValueError, match=re.escape("Alpha must be in [0,1]")):
        SunTopic(sample_Y, sample_X, alpha=alpha, num_bases=5)


@pytest.mark.parametrize(
    ("num_bases", "expected_message"),
    [
        (0, "Number of bases must be at least 1"),
        (21, "Number of bases must be less than the dimensionality of X.shape[1]"),
        (10.5, "Number of bases must be an integer"),
    ],
)
def test_initialization_with_invalid_num_bases(
    sample_X, sample_Y, num_bases, expected_message
):
    # Test initialization of SunTopic instance with invalid num_bases
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        SunTopic(sample_Y, sample_X, alpha=0.5, num_bases=num_bases)


def test_initialization_with_invalid_num_samples(sample_X, sample_Y):
    # Test initialization of SunTopic instance with invalid num_samples
    with pytest.raises(
        ValueError, match="Y and X must have the same number of samples"
    ):
        SunTopic(sample_Y, sample_X[:5], alpha=0.5, num_bases=5)


def test_fit(sample_X, sample_Y):
    # Test fit method of SunTopic instance
    model = SunTopic(sample_Y, sample_X, alpha=0.5, num_bases=4)
    model.fit(niter=10)
    assert model.model.W.shape == (sample_size, 4)
    assert model.model.H.shape == (4, data_dim + 1)
    assert (model.get_coefficients() == model.model.H).all()
    assert (model.get_topics() == model.model.W).all()


def test_fit_standardization(sample_X, sample_Y):
    # Test fit method of SunTopic instance with standardization
    model = SunTopic(sample_Y, sample_X, alpha=0.5, num_bases=4)
    model.fit(niter=10, standardize=True)
    # Ensure that the standard deviation is approximately 1 after standardization
    coefficients_std = np.std(model.get_topics(), axis=0)  # Added () to method call
    assert np.allclose(
        coefficients_std, 1, atol=1e-1
    ), "Standardized coefficients should have a standard deviation of approximately 1"


def test_fit_no_standardization(sample_X, sample_Y):
    # Test fit method of SunTopic instance without standardization
    model = SunTopic(sample_Y, sample_X, alpha=0.5, num_bases=4)
    model.fit(niter=10, standardize=False)
    # Ensure that the standard deviation is not approximately 1 when standardization is off
    coefficients_std = np.std(model.get_topics(), axis=0)  # Added () to method call
    assert not np.allclose(
        coefficients_std, 1, atol=1e-1
    ), "Non-standardized coefficients should not have a standard deviation close to 1"


def test_get_top_docs_idx(sample_X, sample_Y):
    # Test get_top_docs method of SunTopic instance
    model = SunTopic(sample_Y, sample_X, alpha=0.5, num_bases=4)
    model.fit(niter=10)
    with pytest.raises(ValueError, match="Topic index out of bounds"):
        model.get_top_docs_idx(topic=10, n_docs=10)
    with pytest.raises(ValueError, match="Number of top documents must be an integer"):
        model.get_top_docs_idx(topic=1, n_docs=10.5)
    with pytest.raises(ValueError, match="Number of top documents must be at least 1"):
        model.get_top_docs_idx(topic=1, n_docs=-5)
    top_docs = model.get_top_docs_idx(topic=1, n_docs=10)
    assert len(top_docs) == 10
    assert np.allclose(top_docs, np.argsort(model.model.W[:, 1])[::-1][:10])


@pytest.mark.parametrize("random_state", [None, 21])
def test_save_load(sample_X, sample_Y, random_state):
    # Test save and load methods of SunTopic instance
    model = SunTopic(
        sample_Y, sample_X, alpha=0.5, num_bases=4, random_state=random_state
    )
    model.fit(niter=10)
    model.save(filename="test_model.npz")
    loaded_model = SunTopic.load(filename="test_model.npz")
    Path("test_model.npz").unlink()
    assert np.allclose(model.data, loaded_model.data)
    assert model.num_bases == loaded_model.num_bases
    assert model.alpha == loaded_model.alpha
    assert np.allclose(model.Y, loaded_model.Y)
    assert np.allclose(model.X, loaded_model.X)


def test_predict_and_infer(sample_X, sample_Y):
    # Test predict method of SunTopic instance
    model = SunTopic(sample_Y, sample_X, alpha=0.5, num_bases=4)
    model.fit(niter=10)
    X_new = rng.random((5, data_dim))
    Y_pred = model.predict(X_new)
    assert Y_pred.shape == (5,)
    Y_pred, W_pred = model.predict(X_new, return_topics=True)
    assert Y_pred.shape == (5,)
    assert W_pred.shape == (5, 4)


def test_predict_with_invalid_X_new(sample_X, sample_Y):
    # Test predict method of SunTopic instance with invalid X_new
    model = SunTopic(sample_Y, sample_X, alpha=0.5, num_bases=4)
    model.fit(niter=10)
    X_new = rng.random((5, 21))
    with pytest.raises(
        ValueError, match=re.escape("X_new.shape[1] must be equal to X.shape[1]")
    ):
        model.predict(X_new)


def test_predict_with_cvxpy(sample_X, sample_Y):
    # Test predict method of SunTopic instance with cvxpy solver
    model = SunTopic(sample_Y, sample_X, alpha=0.5, num_bases=4)
    model.fit(niter=10)
    X_new = rng.random((5, data_dim))
    Y_pred = model.predict(X_new, cvxpy=True)
    assert Y_pred.shape == (5,)


def test_predict_with_invalid_solver(sample_X, sample_Y):
    # Test predict method of SunTopic instance with invalid solver
    model = SunTopic(sample_Y, sample_X, alpha=0.5, num_bases=4)
    model.fit(niter=10)
    X_new = rng.random((5, data_dim))
    with pytest.raises(
        ValueError, match=re.escape("Solver must be either 'ECOS' or 'SCS'")
    ):
        model.predict(X_new, cvxpy=True, solver="invalid_solver")


def test_predict_with_single_X_new(sample_X, sample_Y):
    # Test predict method of SunTopic instance with invalid X_new
    model = SunTopic(sample_Y, sample_X, alpha=0.5, num_bases=4)
    model.fit(niter=10)
    X_new = rng.random((5, 20))
    Y_pred = model.predict(X_new[0, :])
    assert Y_pred.shape == (1,)


def test_summary(sample_X, sample_Y):
    # Test summary method of SunTopic instance
    model = SunTopic(sample_Y, sample_X, alpha=0.5, num_bases=4)
    model.fit(niter=10)
    assert model.summary() is None


def test_hyperparam_cv():
    Y = rng.random(100)
    X = rng.random((100, 200))
    # Test find_alpha_cv method of SunTopic instance
    model = SunTopic(Y, X, alpha=0.5, num_bases=4)
    model.hyperparam_cv(
        alpha_range=[0.1, 0.5, 0.9],
        num_bases_range=[2, 4, 6],
        cv_folds=3,
        random_state=21,
        niter=10,
    )
    assert model.cv_values["errors"].shape == (3, 3, 3)
    assert model.cv_values["alpha_range"] == [0.1, 0.5, 0.9]
    assert model.cv_values["num_base_range"] == [2, 4, 6]
    assert model.cv_values["folds"] == 3
    assert ((model.cv_values["errors"] >= 0) & (model.cv_values["errors"] <= 1)).all()

    model_parallel = SunTopic(Y, X, alpha=0.5, num_bases=4)
    model_parallel.hyperparam_cv(
        alpha_range=[0.1, 0.5, 0.9],
        num_bases_range=[2, 4, 6],
        cv_folds=3,
        random_state=21,
        niter=10,
        parallel=True,
    )
    assert model_parallel.cv_values["errors"].shape == (3, 3, 3)
    assert model_parallel.cv_values["alpha_range"] == [0.1, 0.5, 0.9]
    assert model_parallel.cv_values["num_base_range"] == [2, 4, 6]
    assert model_parallel.cv_values["folds"] == 3
    assert (
        (model_parallel.cv_values["errors"] >= 0)
        & (model_parallel.cv_values["errors"] <= 1)
    ).all()
    assert model.cv_values["errors"].all() == model_parallel.cv_values["errors"].all()


def test_cv_summary_without_fit(sample_X, sample_Y):
    # Test summary method of SunTopic instance without fit
    model = SunTopic(sample_Y, sample_X, alpha=0.5, num_bases=4)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cross-validation errors have not been computed yet. \
Call hyperparam_cv() first."
        ),
    ):
        model.cv_summary()


def test_cv_mse_plot():
    # Test cv_mse_plot method of SunTopic instance
    Y = rng.random(100)
    X = rng.random((100, 200))
    model = SunTopic(Y, X, alpha=0.5, num_bases=4)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cross-validation errors have not been computed yet. \
Call hyperparam_cv() first."
        ),
    ):
        model.cv_mse_plot()
    model.hyperparam_cv(
        alpha_range=[0.1, 0.5, 0.9],
        num_bases_range=[2, 4, 6],
        cv_folds=3,
        random_state=21,
        niter=10,
        parallel=False,
    )
    fig = model.cv_mse_plot(return_plot=True)
    assert fig is not None
