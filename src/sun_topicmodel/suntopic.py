"""
SUN Topic Model

    suntopic : Class for suntopic model

Authors: Till R. Saenger, ORFE Princeton

[1] Saenger, Hinck, Grimmer, and Stewart. AutoPersuade: 

"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from sun_topicmodel.snmf import SNMF




class suntopic:
    """
    suntopic(data, num_bases)


    """

    def __init__(self, Y, X, alpha, num_bases, random_state=None, **kwargs):
        """
        Initialize the suntopic class.
        data : array_like, shape (_num_samples, _data_dimension)
        num_bases : int, specifies the number of topics to model
        """

        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be in [0,1]")
        if num_bases < 1:
            raise ValueError("Number of bases must be at least 1")
        if num_bases > X.shape[1]:
            raise ValueError("Number of bases must be less than the dimensionality of X.shape[1]")
        if num_bases != int(num_bases):
            raise ValueError("Number of bases must be an integer")

        if len(Y) != X.shape[0]:
            raise ValueError("Y and X must have the same number of samples")

        self.Y = Y
        self.X = X
        self.num_bases = num_bases
        self.alpha = alpha

        data = np.hstack((np.sqrt(alpha) * X, np.sqrt(1 - alpha) * np.array(Y).reshape(-1, 1)))
        self.data = data
        self.model = SNMF(data, num_bases)
        self.model.random_state = random_state

    def fit(self,
            niter=100,
            show_progress=False,
            compute_w=True,
            compute_h=True,
            compute_err=True):
        """
        Fit the suntopic model to the data.
        """
        self.model.factorize(niter=niter,
                                show_progress=show_progress,
                                compute_w=compute_w,
                                compute_h=compute_h,
                                compute_err=compute_err)

    def predict(self, 
                X_new,
                return_topics=False,
                niter=100,
                random_state=None,
                show_progress=False,
                compute_err=True):
        """
        Predict the response variable for new data X_new.
        """
        if X_new.shape[1] != self.X.shape[1]:
            raise ValueError("X_new.shape[1] must be equal to X.shape[1]")
        
        if not hasattr(self.model, "H"):
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        data_new = np.sqrt(self.alpha) * X_new
        self._model_pred = SNMF(data_new, 
                                self.num_bases,
                                random_state=random_state)
        self._model_pred.H = self.model.H[:,:-1]

        self._model_pred.factorize(
                                niter=niter,
                                show_progress=show_progress,
                                compute_h=False,
                                compute_err=compute_err)
        
        Y_pred = np.dot(self._model_pred.W, self.model.H[:,-1])
        Y_pred /= np.sqrt(1 - self.alpha)
        if return_topics is False:
            return Y_pred
        else:
            return Y_pred, self._model_pred.W

    def get_topics(self):
        """
        Get the topics from the suntopic model.
        """
        return self.model.W

    def get_coefficients(self):
        """
        Get the coefficients from the suntopic model.
        """
        return self.model.H
    
    def summary(self):
        """
        Print a summary of the suntopic model.
        """

        if not hasattr(self.model, "H"):
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        print("Suntopic Model Summary")
        print("="*50)
        print("Number of topics: ", self.num_bases)
        print("Alpha: ", self.alpha)
        print("Data shape: ", self.data.shape)
        print("Model: ", self.model)
        print("Random initialization state: ", self.model.random_state)
        # print("Frobenius norm error: ", self.model.ferr)
        print("Prediction coefficients: ", self.model.H[:,-1])
        # print("Topics: ", self.model.W)
        # print("Coefficients: ", self.model.H)
        
        return None

    def save(self, filename):
        """
        Save the suntopic model to a file.
        """
        np.savez(filename, 
                 Y = self.Y, 
                 X = self.X, 
                 W = self.model.W, 
                 H = self.model.H, 
                 alpha = self.alpha, 
                 random_state = str(self.model.random_state))

    @staticmethod
    def load(filename):
        """
        Load a suntopic model from a file.
        """
        npzfile = np.load(filename)
        
        if npzfile["random_state"] == "None":
            loaded_model = suntopic(Y=npzfile["Y"], 
                                X=npzfile["X"], 
                                alpha=npzfile["alpha"], 
                                num_bases=npzfile["W"].shape[1])
        else:
            loaded_model = suntopic(Y=npzfile["Y"], 
                                X=npzfile["X"], 
                                alpha=npzfile["alpha"], 
                                num_bases=npzfile["W"].shape[1],
                                random_state=int(npzfile["random_state"]))
            
        loaded_model.model.W = npzfile["W"]
        loaded_model.model.H = npzfile["H"]
        return loaded_model
    
    def hyperparam_cv(self, alpha_range, num_bases_range, 
                      cv_folds, random_state = None, parallel = False, 
                      niter=100):
        """
        Calculate the cross-validated mean squared error for different values of alpha and num_bases.
        """
        for alpha in alpha_range:
            if alpha < 0 or alpha > 1:
                raise ValueError("Alpha must be in [0,1]")
            
        for num_bases in num_bases_range:
            if num_bases < 1:
                raise ValueError("Each number of bases must be at least 1")
            if num_bases > self.X.shape[1]:
                raise ValueError("Each number of bases must be less than the dimensionality of X.shape[1]")
            if num_bases != int(num_bases):
                raise ValueError("Each number of bases must be an integer")
        
        if cv_folds < 2:
            raise ValueError("Number of folds must be at least 2")
        if cv_folds != int(cv_folds):
            raise ValueError("Number of folds must be an integer")
        if cv_folds > len(self.Y):
            raise ValueError("Number of folds must be less than the number of samples")
        
        self.cv_alpha = alpha_range
        self.cv_num_bases = num_bases_range
        self.cv_folds = cv_folds
        self.cv_errors = np.full((len(alpha_range), len(num_bases_range), cv_folds), None)
        self.cv_random_state = random_state
        
        kf = KFold(n_splits=cv_folds, random_state=self.cv_random_state, shuffle=True)
        self.cv_errors = np.ones((len(num_bases_range), len(alpha_range), cv_folds))*np.nan

        def predict_Y_mse(alpha, num_bases, train_index, test_index, random_state=self.cv_random_state):
            model = suntopic(Y=self.Y[train_index], 
                            X=self.X[train_index], 
                            alpha=alpha, 
                            num_bases=num_bases, 
                            random_state=random_state)
            model.fit(niter=niter, show_progress=False)
            Y_pred = model.predict(self.X[test_index], random_state=random_state)
            return mean_squared_error(self.Y[test_index], Y_pred)

        if parallel == False:
            for i, num_bases in enumerate(num_bases_range):
                for j, alpha in enumerate(alpha_range):
                    for k, (train_index, test_index) in enumerate(kf.split(self.Y)):
                        self.cv_errors[i,j,k]= predict_Y_mse(alpha, num_bases, train_index, test_index)
        else: 
            # Sequentially loop over alpha_ranges and parallelize across topic_range
            num_cores = -1 if len(alpha_range) > 1 else 1 # Use all available cores
            results = Parallel(n_jobs=num_cores)(
                delayed(predict_Y_mse)(alpha, num_bases, train_index, test_index)
                for i, num_bases in enumerate(num_bases_range)
                for j, alpha in enumerate(alpha_range)
                for k, (train_index, test_index) in enumerate(kf.split(self.Y))
            )
            # Assign results back to cv_errors array
            idx = 0
            for i in range(len(num_bases_range)):
                for j in range(len(alpha_range)):
                    for k in range(cv_folds):
                        self.cv_errors[i, j, k] = results[idx]
                        idx += 1
            
            return self.cv_errors

    def cv_summary(self):
        """
        Print a summary of the cross-validation runs of suntopic models.
        """

        if hasattr(self, "cv_errors") is False:
            raise ValueError("Cross-validation errors have not been computed yet. Call hyperparam_cv() first.")

        print("Cross-Validation Summary")
        print("="*50)
        print("Alpha candidate values: ", self.cv_alpha)
        print("Number of topics: ", self.cv_num_bases)
        print("Number of folds: ", self.cv_folds)
        print("CV Random state: ", self.cv_random_state)
        print("MSE Errors: ", self.cv_errors)
        
        return None
