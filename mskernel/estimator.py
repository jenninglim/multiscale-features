import numpy as np
from future.utils import with_metaclass
from abc import ABCMeta, abstractmethod

class Estimator(with_metaclass(ABCMeta, object)):
    """
        Abstract class for various estimators.
    """
    def compute(self, X, Y):
        """
        Compute statistic for sample from X ~ P and Y ~ Q.
        """
        raise NotImplementedError()    

    def compute_var(self ,X ,Y ):
        raise NotImplementedError()

    def compute_cov(self ,X ,Y ):
        raise NotImplementedError()
