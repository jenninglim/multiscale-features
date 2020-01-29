import numpy as np
from future.utils import with_metaclass
from abc import ABCMeta, abstractmethod
from sklearn.linear_model import LinearRegression
import math

# TODO: Perhaps add linear tangent line at point sigma=1

class Model(with_metaclass(ABCMeta, object)):
    """
    Abstract class for models used to extrapolate to sigma^2=x
    """
    def f(self, x):
        raise NotImplementedError()    

class LinearModel(Model):
    def __init__(self, sigma2s, phis,):
        """
        Fit a linear model such that phis = b_0 + b_1 * sigma2s
        """
        self.model = LinearRegression().fit(sigma2s, phis)
        self.aic = aic(sse(np.array(phis), self.f(sigma2s)[:,0]),
                len(phis), 2)

    def f(self, x):
        intercept = self.model.intercept_
        coefs = self.model.coef_[0]
        return intercept + coefs * x

class PolyModel(Model):
    def __init__(self, sigma2s,phis, degree=2):
        assert(degree > 0)
        poly_feats = []
        self.degree = degree
        for i in range(degree):
            poly_feats.append(np.array(sigma2s) ** (i+1))
        self.model = LinearRegression().fit(np.hstack(poly_feats), phis)
        self.aic = aic(sse(np.array(phis), self.f(sigma2s)[:,0]),
                len(phis), degree + 1)
    def dnf(self, x, n):
        coef = self.model.coef_[n-1:]
        if n == 0:
            return self.f(x)
        else:
            return np.poly1d(np.polyder(np.hstack((self.model.intercept_, self.model.coef_))[::-1], n))(x)

    def f(self, x):
        intercept = self.model.intercept_
        coefs = self.model.coef_
        acc = intercept
        for i in range(self.degree):
            acc+=coefs[i]*x**(i+1)
        return acc

class TaylModel(Model):
    def __init__(self, sigma2s,phis, degree=5):
        self.pmodel = PolyModel(sigma2s,phis, degree=degree)
        self.degree = degree
        self.aic = self.pmodel.aic
    
    def f(self,x, k=2, sigma=3):
        acc = 0
        for j in range(k):
            acc += (x-sigma) ** j/ math.factorial(j) *  self.pmodel.dnf(sigma, j)
        return acc
        """
        return self.pmodel.dnf(sigma,1)*x +\
                self.pmodel.dnf(sigma,0) - \
                self.pmodel.dnf(sigma,1)*sigma
        """

def aic(sse, n_samples, predictors):
    return n_samples * np.log(sse/n_samples) + 2 * (predictors + 1)

def sse(true, predicted):
    assert(true.shape == predicted.shape)
    return np.sum(np.square(true-predicted))
