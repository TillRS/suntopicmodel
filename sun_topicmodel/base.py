"""
Base class for all PyMF classes 
which we use for convex and semi-nonnegative matrix factorization.

Authors: Till R. Saenger, ORFE Princeton 
 

Notes:

This implementation builds directly on an unsupported implementation of 
Christian Thurau (https://github.com/pzoccante/pymf/blob/master/pymf/)

    
[1] Ding, C., Li, T. and Jordan, M.. Convex and Semi-Nonnegative Matrix Factorizations.
IEEE Trans. on Pattern Analysis and Machine Intelligence 32(1), 45-55.
"""

import numpy as np
import logging
import logging.config
import scipy.sparse
from sklearn.cluster import KMeans

__all__ = ["PyMFBase", "PyMFBase3", "eighk", "cmdet", "simplex"]
_EPS = np.finfo(float).eps

class PyMFBase():
    """
    PyMF Base Class. Does nothing useful apart from poviding some basic methods.
    """
    # some small value
   
    _EPS = _EPS
    
    def __init__(self, data, num_bases=4, random_state = None, **kwargs):
        """
        """
        
        def setup_logging():
            # create logger       
            self._logger = logging.getLogger("pymf")
       
            # add ch to logger
            if len(self._logger.handlers) < 1:
                # create console handler and set level to debug
                ch = logging.StreamHandler()
                ch.setLevel(logging.DEBUG)
                # create formatter
                formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        
                # add formatter to ch
                ch.setFormatter(formatter)

                self._logger.addHandler(ch)

        setup_logging()
        
        # set variables
        self.data = data       
        self._num_bases = num_bases 
        self.random_state = random_state            
      
        # initialize H and W to random values
        # TILL: note - this def of dimensions vs num_samples can be confusing!
        self._data_dimension, self._num_samples = self.data.shape
        # self._num_samples, self._data_dimension = self.data.shape
        

    # def residual(self):
    #     """ Returns the residual in % of the total amount of data

    #     Returns
    #     -------
    #     residual : float
    #     """
    #     res = np.sum(np.abs(self.data - np.dot(self.W, self.H)))
    #     total = 100.0*res/np.sum(np.abs(self.data))
    #     return total
        
    def frobenius_norm(self):
        """ Frobenius norm (||data - WH||) of a data matrix and a low rank
        approximation given by WH. Minimizing the Fnorm ist the most common
        optimization criterion for matrix factorization methods.

        Returns:
        -------
        frobenius norm: F = ||data - WH||

        """
        # check if W and H exist
        if hasattr(self,'H') and hasattr(self,'W'):
            if scipy.sparse.issparse(self.data):
                tmp = self.data[:,:] - (self.W * self.H)
                tmp = tmp.multiply(tmp).sum()
                err = np.sqrt(tmp)
            else:
                err = np.sqrt( np.sum((self.data[:,:] - np.dot(self.W, self.H))**2 ))            
        else:
            err = None

        return err
        
    def _init_w(self):
        """ Initalize W to random values [0,1].
        """
        # add a small value, otherwise nmf and related methods get into trouble as 
        # they have difficulties recovering from zero.
        np.random.seed(self.random_state)
        self.W = np.random.random((self._data_dimension, self._num_bases)) + 10**-4
        
        
    def _init_h(self):
        """ Initalize H to random values [0,1].
        """
        # np.random.seed(self.random_state+1)
        # self.H = np.random.random((self._num_bases, self._num_samples)) + 10**-4
        
        # initialize using k-means ++
        self.H = np.zeros((self._num_bases, self._num_samples))            
        km = KMeans(n_clusters = self._num_bases, 
                    random_state = self.random_state,
                    n_init='auto',
                    init='k-means++').fit(self.data.T) # need to transpose this relative to the original code       
        assign = km.labels_
        
        num_i = np.zeros(self._num_bases)
        for i in range(self._num_bases):
            num_i[i] = len(np.where(assign == i)[0])

        self.H.T[range(len(assign)), assign] = 1.0                
        self.H += 0.2*np.ones((self._num_bases, self._num_samples))
        
    def _update_h(self):
        """ Overwrite for updating H.
        """
        pass

    def _update_w(self):
        """ Overwrite for updating W.
        """
        pass

    def _converged(self, i):
        """ 
        If the optimization of the approximation is below the machine precision,
        return True.

        Parameters
        ----------
            i   : index of the update step

        Returns
        -------
            converged : boolean 
        """
        derr = np.abs(self.ferr[i] - self.ferr[i-1])/self._num_samples
        if derr < self._EPS:
            return True
        else:
            return False

    def factorize(self, niter=100, show_progress=False, 
                  compute_w=True, compute_h=True, compute_err=True):
        """ Factorize s.t. WH = data
        
        Parameters
        ----------
        niter : int
                number of iterations.
        show_progress : bool
                print some extra information to stdout.
        compute_h : bool
                iteratively update values for H.
        compute_w : bool
                iteratively update values for W.
        compute_err : bool
                compute Frobenius norm |data-WH| after each update and store
                it to .ferr[k].
        
        Updated Values
        --------------
        .W : updated values for W.
        .H : updated values for H.
        .ferr : Frobenius norm |data-WH| for each iteration.
        """
        
        if show_progress:
            self._logger.setLevel(logging.INFO)
        else:
            self._logger.setLevel(logging.ERROR)        
        
        # create W and H if they don't already exist
        # -> any custom initialization to W,H should be done before
        if not hasattr(self,'W') and compute_w:
            self._init_w()
               
        if not hasattr(self,'H') and compute_h:
            self._init_h()                   
        
        # Computation of the error can take quite long for large matrices,
        # thus we make it optional.
        if compute_err:
            self.ferr = np.zeros(niter)
             
        for i in range(niter):
            if compute_w:
                self._update_w()

            if compute_h:
                self._update_h()                                        
         
            if compute_err:                 
                self.ferr[i] = self.frobenius_norm()                
                self._logger.info('FN: %s (%s/%s)'  %(self.ferr[i], i+1, niter))
            else:                
                self._logger.info('Iteration: (%s/%s)'  %(i+1, niter))
           

            # check if the err is not changing anymore
            if i > 1 and compute_err:
                if self._converged(i):
                    # adjust the error measure
                    self.ferr = self.ferr[:i]
                    break


    # Define a save method to export self.W and self.H
    def save(self, filename):
        """ Save the factorization to a file. 

        Parameters
        ----------
        filename : str
            name of the file to save to.
        """
        np.savez(filename, W=self.W, H=self.H, ferr=self.ferr)


    @staticmethod
    def load(filename):
        """ Load a factorization from a file. 

        Parameters
        ----------
        filename : str
            name of the file to load from.
        """
        npzfile = np.load(filename)
        loaded_model = PyMFBase(data=npzfile['W'] @ npzfile['H'], num_bases=npzfile['W'].shape[1])
        loaded_model.W = npzfile['W']
        loaded_model.H = npzfile['H']
        # If 'ferr' is saved and needs to be loaded, uncomment the following line
        # loaded_model.ferr = npzfile['ferr']
        return loaded_model