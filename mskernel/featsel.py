import numpy as np
from future.utils import with_metaclass
from abc import ABCMeta, abstractmethod

from mskernel.fdr import bh, fisher
from mskernel import util

import mskernel.poly as poly
import mskernel.msboot
import mskernel.msbootsel

class FeatureSelection(with_metaclass(ABCMeta, object)):
    """
        Abstract class for various feature selection
        algorithms.
    """
    def test(self, X, Y, n_select, alpha, seed):
        """
            Returns signifcant features.
        """
        raise NotImplementedError()

def region(y, scaling, index, k):
    bp = np.sum(y[:,index] <= 0)/y.shape[0]
    return bp

def select(x, scaling, index, k):
    """
    Select the largest k values
    x : n_bootstrap x n_dim
    """
    sel_vars = np.argpartition(x, -k, axis=1)[:,-k:]
    isin = np.apply_along_axis(lambda x: np.isin(x, index), 1, sel_vars)
    return np.sum(isin)/x.shape[0]

class MultiSel(FeatureSelection):
    def __init__(self, dist, params=True):
        """
           dist: dist function of type Estimator
        """
        self.dist = dist
        self.linear_model = params

    def test(self, X, Y, args=None, alpha=0.05, seed=5, plot=False):
        n_select = args
        n_samples = X.shape[0]
        n_dim = X.shape[1]

        ## Compute mean and variance of covariance matrix
        params = self.dist.compute(X,Y), \
            self.dist.compute_cov(X,Y), \
            self.dist.n_estimates, \
            n_samples

        ## Multiscale bootstrap
        ms = mskernel.msbootsel.Multiscale([X,Y],
                region,
                select,
                n_scalings=10,
                params=params,
                linear_model=self.linear_model)

        ## Select n_select lowest scores
        sel_vars = np.argpartition(params[0], -n_select, axis=0)[-n_select:]

        pvals = []
        stat = np.sqrt(self.dist.n_estimates(n_samples)/params[1].diagonal()) \
                * params[0]

        for j,index in enumerate(sel_vars):
            sparams = [index, n_select]
            pval = ms.pvalue(sparams,
                    seed=seed+j+4,
                    plot=plot,
                    stat=stat[index])
            pvals.append(pval) 

        rej = np.any(bh(pvals, alpha))
        pvals = np.array(pvals)
        results ={ 
                'sel_vars': sel_vars,
                'h0_rejs': pvals < alpha,
                'pvals': pvals,
                'rej': rej,
                }
        return results

    def __repr__(self):
        return "MultiscaleSelection"

    def __str__(self):
        return "MultiscaleSelection"


class PolySel(FeatureSelection):
    def __init__(self, dist, params=None):
        self.dist = dist

    def test(self, X, Y, args=None, alpha=0.05, seed=5, plot=False):
        n_select = args
        n_samples = X.shape[0]
        n_dim = X.shape[1]

        ## Compute mean and variance of covariance matrix
        z, cov = \
            np.sqrt(self.dist.n_estimates(n_samples)) * self.dist.compute(X,Y), \
            self.dist.compute_cov(X,Y)
        
        sel_vars, A, b = poly.selection(z, n_select)

        h0_rejs=[]
        for index in sel_vars:
            # Define eta
            eta = np.zeros(n_dim)
            eta[index] = 1
            stat = np.dot(eta,z)
            ppf,params = poly.psi_inf(A, b, eta, np.zeros(n_dim), cov, z)
            thres = ppf(1-alpha)
            h0_rejs.append(thres < stat)
        #rej = fisher(pvals,alpha)
        #rej = np.any(pvals< alpha)
        results ={ 
                'sel_vars': sel_vars,
                'h0_rejs': np.array(h0_rejs),
                }
        return results

    def __repr__(self):
        return "PolyhedralSelection"
    def __str__(self):
        return "PolyhedralSelection"


