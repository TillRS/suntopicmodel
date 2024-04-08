"""
SUN Topic Model

    suntopic : Class for suntopic model

Authors: Till R. Saenger, ORFE Princeton

[1] Saenger, Hinck, Grimmer, and Stewart. AutoPersuade: 

"""

from __future__ import annotations

import numpy as np
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

    def get_model(self):
        """
        Get the suntopic model.
        """
        return self.model
    
    def summary(self):
        """
        Print a summary of the suntopic model.
        """
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