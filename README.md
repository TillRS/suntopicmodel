# SUN_TopicModel
Python implementation of the SUpervised Nonnegative matrix factorization (SUN) topic model for topic discovery and effect estimation from "AutoPersuade: A Framework for Evaluating and Explaining Persuasive Arguments".



## Installation
To install this package, clone the repository and install it using pip.
The current setup requires manual installation of the dependencies or using the `suntopic_env.yml` to create an environment.
```bash
git clone https://github.com/TillRS/SUN_TopicModel
cd SUN_TopicModel
pip install .
```

## Usage
For a sample use of the package, please refer to the replication code of the AuoPersaude paper, avalilable here: https://github.com/TillRS/AutoPersuade.
For a simpler demonstartion, refer to the `sample.ipynb` notebook.

## Structure
#### Package:
- **`__init__.py`:** initializes the package.
- **`base.py`:** Sets up the base class `PyMFBase` for matrix factorization following the implementation of [1].
- **`snmf.py`:** Defines the semi-nonnegative matrix factorization class `SNMF` of [2] building on the implementation of [1].
- **`suntopic.py`:** Defines the topic model class using the `SNMF` class including the features:
    - Model Initialization: Allows for flexible configuration of the alpha parameter and the number of topics.
    - Fit Functionality: Implements the fitting of SNMF models using an iterative approach.
    - Cross-Validation for Hyperparameter Tuning: Provides support for cross-validation on both alpha and num_bases
    - Save/Load Functionality: Models can be saved to disk and reloaded for later use.
    - Prediction: Capable of predicting outcomes for new data points with or without topic assignment.
    - Topic Summarization: Summarize the model and visualize cross-validation results.
    - Parallel Processing: Supports parallel processing for cross-validation.

The main parameters of the topic model are:
- `Y` (numpy.ndarray): The target vector (e.g., labels or outcomes).
- `X` (numpy.ndarray): Input data matrix of document embeddings where rows represent samples and columns represent features.
- `alpha` (float, optional): A weighting parameter between `X` and `Y`. Default is 0.5. Must be within the range [0, 1].
- `num_bases` (int, optional): The number of basis vectors (topics). Default is 5. Must be a positive integer
- `random_state` (int, optional): A random seed for reproducibility. Default is None.

#### Testing
Pytest test functions for the different class objects are collected in:
- **`test_base.py`:** Tests for base matrix factorization class.
- **`test_snmf.py`:** Tests for semi-nonnegative matrix factorization implementation.
- **`test_tuntopic.py`:** Tests for topic modeling functionalities.

## Dependencies and setup
- Relevant dependencies are included in `pyproject.toml`.
- The CI setup is included in `.github/workflows/ci.yml`.


## Acknowledgements

This implementation builds directly on the relevant parts of [1]:
- https://github.com/cthurau/pymf.

The semi-nonnegative matrix factorization approach was developed by [2]:
- C. H. Q. Ding, T. Li and M. I. Jordan, "Convex and Semi-Nonnegative Matrix Factorizations," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 32, no. 1, pp. 45-55, Jan. 2010, doi: 10.1109/TPAMI.2008.277.



This repository is currently work in progress and **not ready** for final users.
