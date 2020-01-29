"""Module containing kernel related classes"""

__author__ = 'wittawat'

from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.signal as sig
from mskernel import util

class Kernel(object):
    """Abstract class for kernels"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def eval(self, X1, X2):
        """Evalute the kernel on data X1 and X2 """
        pass

    @abstractmethod
    def pair_eval(self, X, Y):
        """Evaluate k(x1, y1), k(x2, y2), ..."""
        pass

class KHoPoly(Kernel):
    """Homogeneous polynomial kernel of the form
    (x.dot(y))**d
    """
    def __init__(self, degree):
        assert degree > 0
        self.degree = degree

    def eval(self, X1, X2):
        return X1.dot(X2.T)**self.degree

    def pair_eval(self, X, Y):
        return np.sum(X1*X2, 1)**self.degree

    def __str__(self):
        return 'KHoPoly(d=%d)'%self.degree



class KLinear(Kernel):
    def eval(self, X1, X2):
        return X1.dot(X2.T)

    def pair_eval(self, X, Y):
        return np.sum(X*Y, 1)

    def __str__(self):
        return "KLinear()"

class KGauss(Kernel):

    def __init__(self, sigma2):
        assert sigma2 > 0, 'sigma2 must be > 0. Was %s'%str(sigma2)
        self.sigma2 = sigma2

    def eval(self, X1, X2):
        """
        Evaluate the Gaussian kernel on the two 2d numpy arrays.

        Parameters
        ----------
        X1 : n1 x d numpy array
        X2 : n2 x d numpy array

        Return
        ------
        K : a n1 x n2 Gram matrix.
        """
        (n1, d1) = X1.shape
        (n2, d2) = X2.shape
        assert d1==d2, 'Dimensions of the two inputs must be the same'
        D2 = np.sum(X1**2, 1)[:, np.newaxis] - 2*X1.dot(X2.T) + np.sum(X2**2, 1)
        K = np.exp(-D2/self.sigma2)
        return K

    def pair_eval(self, X, Y):
        """
        Evaluate k(x1, y1), k(x2, y2), ...

        Parameters
        ----------
        X, Y : n x d numpy array

        Return
        -------
        a numpy array with length n
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert n1==n2, 'Two inputs must have the same number of instances'
        assert d1==d2, 'Two inputs must have the same dimension'
        D2 = np.sum( (X-Y)**2, 1)
        Kvec = np.exp(-D2/self.sigma2)
        return Kvec

    def __str__(self):
        return "KGauss(%.3f)"%self.sigma2



class KTriangle(Kernel):
    """
    A triangular kernel defined on 1D. k(x, y) = B_1((x-y)/width) where B_1 is the 
    B-spline function of order 1 (i.e., triangular function).
    """

    def __init__(self, width):
        assert width > 0, 'width must be > 0'
        self.width = width

    def eval(self, X1, X2):
        """
        Evaluate the triangular kernel on the two 2d numpy arrays.

        Parameters
        ----------
        X1 : n1 x 1 numpy array
        X2 : n2 x 1 numpy array

        Return
        ------
        K : a n1 x n2 Gram matrix.
        """
        (n1, d1) = X1.shape
        (n2, d2) = X2.shape
        assert d1==1, 'd1 must be 1'
        assert d2==1, 'd2 must be 1'
        diff = (X1-X2.T)/self.width
        K = sig.bspline( diff , 1)
        return K

    def pair_eval(self, X, Y):
        """
        Evaluate k(x1, y1), k(x2, y2), ...

        Parameters
        ----------
        X, Y : n x 1 numpy array

        Return
        -------
        a numpy array with length n
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert d1==1, 'd1 must be 1'
        assert d2==1, 'd2 must be 1'
        diff = (X-Y)/self.width
        Kvec = sig.bspline( diff , 1)
        return Kvec

    def __str__(self):
        return "KTriangle(w=%.3f)"%self.width

class KIMQ(Kernel):
    """
    The inverse multiquadric (IMQ) kernel studied in

    Measure Sample Quality with Kernels
    Jackson Gorham, Lester Mackey

    k(x,y) = (c^2 + ||x-y||^2)^b
    where c > 0 and b < 0. Following a theorem in the paper, this kernel is
    convergence-determining only when -1 < b < 0. In the experiments,
    the paper sets b = -1/2 and c = 1.
    """

    def __init__(self, b=-0.5, c=1.0):
        if not b < 0:
            raise ValueError('b has to be negative. Was {}'.format(b))
        if not c > 0:
            raise ValueError('c has to be positive. Was {}'.format(c))
        self.b = b
        self.c = c

    def eval(self, X, Y):
        """Evalute the kernel on data X and Y """
        b = self.b
        c = self.c
        D2 = util.dist2_matrix(X, Y)
        K = (c**2 + D2)**b
        return K

    def pair_eval(self, X, Y):
        """Evaluate k(x1, y1), k(x2, y2), ...
        """
        assert X.shape[0] == Y.shape[0]
        b = self.b
        c = self.c
        return (c**2 + np.sum((X-Y)**2, 1))**b

    def gradX_Y(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of X in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        D2 = util.dist2_matrix(X, Y)
        # 1d array of length nx
        Xi = X[:, dim]
        # 1d array of length ny
        Yi = Y[:, dim]
        # nx x ny
        dim_diff = Xi[:, np.newaxis] - Yi[np.newaxis, :]

        b = self.b
        c = self.c
        Gdim = ( 2.0*b*(c**2 + D2)**(b-1) )*dim_diff
        assert Gdim.shape[0] == X.shape[0]
        assert Gdim.shape[1] == Y.shape[0]
        return Gdim

    def gradY_X(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of Y in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        return -self.gradX_Y(X, Y, dim)

    def gradXY_sum(self, X, Y):
        """
        Compute
        \sum_{i=1}^d \frac{\partial^2 k(X, Y)}{\partial x_i \partial y_i}
        evaluated at each x_i in X, and y_i in Y.

        X: nx x d numpy array.
        Y: ny x d numpy array.

        Return a nx x ny numpy array of the derivatives.
        """
        b = self.b
        c = self.c
        D2 = util.dist2_matrix(X, Y)

        # d = input dimension
        d = X.shape[1]
        c2D2 = c**2 + D2
        T1 = -4.0*b*(b-1)*D2*(c2D2**(b-2) )
        T2 = -2.0*b*d*c2D2**(b-1)
        return T1 + T2

