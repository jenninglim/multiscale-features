import numpy as np

from math import floor
from itertools import permutations
from scipy.special import binom, perm, comb

from mskernel import util
from mskernel import kernel

def hsic(X, Y, k, l):
    """
    From: https://github.com/wittawatj/fsic-test/blob/master/fsic/indtest.py
    Compute the biased estimator of HSIC as in Gretton et al., 2005.
    :param k: a Kernel on X
    :param l: a Kernel on Y
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError('X and Y must have the same number of rows (sample size')

    n = X.shape[0]
    K = k.eval(X, X)
    L = l.eval(Y, Y)
    Kmean = np.mean(K, 0)
    Lmean = np.mean(L, 0)
    HK = K - Kmean
    HL = L - Lmean
    # t = trace(KHLH)
    HKf = HK.flatten()/(n-1)
    HLf = HL.T.flatten()/(n-1)
    hsic = HKf.dot(HLf)
    #t = HK.flatten().dot(HL.T.flatten())
    #hsic = t/(n-1)**2.0
    return hsic

class HSIC_U():
    def __init__(self, k, l):
        self.k = k
        self.l = l
    
    def compute(self, x, y):
        """
        Compute Unbiased HSIC
        Code from: https://www.cc.gatech.edu/~lsong/papers/SonSmoGreBedetal12.pdf
        """
        nx = x.shape
        ny = y.shape
        assert nx[0] == ny[0], \
               "Argument 1 and 2 have different number of data points"

        K = self.k.eval(x,x)
        L = self.l.eval(y,y)
        kMat, lMat = K - np.diag(K.diagonal()), \
                    L - np.diag(L.diagonal())

        sK = kMat.sum(axis=1)
        ssK = sK.sum()
        sL = lMat.sum(axis=1)
        ssL = sL.sum()

        return ( kMat.__imul__(lMat).sum() + \
                 (ssK*ssL)/((nx[0]-1)*(nx[0]-2)) - \
                 2 * sK.__imul__(sL).sum() / (nx[0]-2) \
                 ) / (nx[0]*(nx[0]-3))

class HSIC_Inc():
    def __init__(self, k, l, ratio =1):
        self.k = k
        self.l = l
        self.ratio = ratio
    
    def estimates(self, x, y, seed=2):
        """
        Compute Unbiased HSIC
        Code from: https://www.cc.gatech.edu/~lsong/papers/SonSmoGreBedetal12.pdf
        """
        m = int(x.shape[0] * self.ratio)
        n_samples = x.shape[0]
        n_comb = comb(n_samples, 4)

        with util.NumpySeedContext(seed=seed):
            S = np.random.randint(0, n_comb,size=m)

        def mapping(S, n_samples,l):
            for index in S:
                res = index
                coord = []
                for power in range(1,5):
                    norm = np.math.factorial(n_samples-power) / (np.math.factorial(n_samples-4) *np.math.factorial(4))
                    i =int(np.floor(res/norm))
                    res = res - i * norm
                    coord.append(i)

                i, j, q, r = coord

                # Non diagonal elements
                j = j if i != j else j + 1

                q = q if q != i else q + 1
                q = q if q != j else q + 1

                r = r if r != i else r + 1
                r = r if r != j else r + 1
                r = r if r != q else r + 1
                yield i, j, q, r

        nx = x.shape
        ny = y.shape
        assert nx[0] == ny[0], \
               "Argument 1 and 2 have different number of data points"

        K = self.k.eval(x,x)
        L = self.l.eval(y,y)
        kMat, lMat = K - np.diag(K.diagonal()), \
                    L - np.diag(L.diagonal())
        
        estimates =  np.zeros(m)
        for i, indices in enumerate(mapping(S, nx[0], m)):
            acc = 0
            for s, t, u, v in permutations(indices):
                acc += kMat[s,t] * (lMat[s,t] + lMat[u,v] - 2 * lMat[s,u]) / 24
            estimates[i] =  acc
        return estimates.flatten()

    def compute(self, x, y, seed=5, dim=True):
        m = int(x.shape[0] * self.ratio)
        if dim:
            n, d = x.shape
            hsic = np.zeros(d)
            for i in range(d):
                hsic[i] = np.sum(self.estimates(x[:, i, np.newaxis],
                    y,
                    seed=i+seed)) / m
            return hsic
        return np.sum(self.estimates(x, y, seed=seed)) / m

    def compute_var(self, x, y, seed=5):
        m = int(x.shape[0] * self.ratio)
        return np.var(self.estimates(x, y, seed=seed))
    
    def compute_cov(self, x, y, seed=5):
        n, d = x.shape
        m = int(x.shape[0] * self.ratio)

        estimates = np.zeros((d, m))

        for i in range(d):
            estimates[i] = self.estimates(x[:,i, np.newaxis], y, seed=seed+i).flatten()
        return np.cov(estimates)

    def n_estimates(self, n):
        m = int(n * self.ratio)
        return m

class HSIC_Block():
    def __init__(self, k, l, bsize):
        self.hsic = HSIC_U(k, l)
        self.bsize = int(bsize)

    def compute(self,X,Y, dim=True):
        if dim:
            n, d = X.shape
            hsic = np.zeros(d)
            for i in range(d):
                hsic[i] = np.mean(self.estimates(X[:,i, np.newaxis], Y))
            return hsic
        return np.mean(self.estimates(X,Y))
    
    def estimates(self, X, Y):
        n = X.shape[0]
        blocksize = self.bsize

        n_blocks = int(np.floor(n / blocksize))
        samples = np.zeros((n_blocks,1))
        assert(n_blocks > 0)

        acc = 0
        for i, k in enumerate(range(n_blocks)):
            i_start = int(k * blocksize)
            i_end   = i_start + blocksize

            samples[i] = self.hsic.compute(X[i_start:i_end, :], \
                    Y[i_start:i_end, :])
        return samples

    def compute_var(self,X,Y):
        n = X.shape[0]
        blocksize = self.bsize

        n_blocks = int(np.floor(n / blocksize))

        return np.var(self.estimates(X,Y))

    def n_estimates(self, n):
        blocksize = self.bsize
        m = int(np.floor(n / blocksize))
        return m

    def compute_cov(self, x, y, seed=5):
        n, d = x.shape
        blocksize = self.bsize
        m = int(np.floor(n / blocksize))

        estimates = np.zeros((d, m))

        for i in range(d):
            estimates[i] = self.estimates(x[:,i,np.newaxis], y).flatten()
        return np.cov(estimates) 
