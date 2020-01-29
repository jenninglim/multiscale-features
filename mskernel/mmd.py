import numpy as np

from mskernel import util
from mskernel.util import NumpySeedContext
from mskernel.estimator import Estimator

class MMD_Inc(Estimator):
    def __init__(self, kernel, ratio=1, seed=2):
        self.kernel = kernel
        self.ratio = ratio
        self.seed = seed

    def compute(self, x,y, dim=True):
        d = x.shape[1]
        n = x.shape[0]
        if dim:
            h_b = np.zeros((d,self.n_estimates(n)))
            for i in range(d):
                x_dim = x[:,i,np.newaxis] if x.ndim == 2 else x[:,i]
                y_dim = y[:,i,np.newaxis] if y.ndim == 2 else y[:,i]
                h_b[i,:] = self.estimates(x_dim, y_dim).flatten()
            mmd = np.mean(h_b,axis=1)
        else:
            samples = self.estimates(x,y) 
            mmd = np.mean(samples)
        return mmd

    def compute_cov(self, x, y):
        d = x.shape[1]
        n = x.shape[0]
        l = self.n_estimates(n)
        H_b = np.zeros((d,self.n_estimates(n)))
        for i in range(d):
            x_dim = x[:,i,np.newaxis] if x.ndim == 2 else x[:,i]
            y_dim = y[:,i,np.newaxis] if y.ndim == 2 else y[:,i]
            H_b[i,:] = self.estimates(x_dim, y_dim).flatten()
        sigma = np.cov(H_b)
        return sigma

    def compute_var(self, x,y):
        n = x.shape[1]
        m = x.shape[0]
        l = self.n_estimates(m)
        h_b = self.estimates(x, y).flatten()
        return np.var(h_b)

    def n_estimates(self, m):
        return int(self.ratio * m)

    def estimates(self, X, Y, dim=False):
        if dim:
            d = Y.shape[1]
            n = Y.shape[0]
            h_b = np.zeros((d,self.n_estimates(n)))
            for i in range(d):
                h_b[i,:] = self.estimates(X[:,i,np.newaxis], Y[:,i,np.newaxis], dim=False).flatten()
            return h_b
        Kxx = self.kernel.eval(X,X)
        Kyy = self.kernel.eval(Y,Y)
        Kxy = self.kernel.eval(X,Y)
        def mapping(S, n_samples,l):
            for j in range(l):
                i = S[j]
                level = int(np.floor(i/(n_samples-1)))
                remainder = i - level * (n_samples-1)
                if remainder >= level:
                    yield level, remainder+1, j
                else:
                    yield level, remainder-1, j

        n_samples = np.shape(X)[0]
        l = self.n_estimates(n_samples)
        with NumpySeedContext(seed=self.seed):
            S = np.random.randint(0,n_samples*(n_samples-1),size=l)
        samples = np.zeros(l)
        for i,j,ind in mapping(S, n_samples,l):
            assert(i != j)
            samples[ind] = Kxx[i][j]+ Kyy[i][j] - Kxy[j][i]  - Kxy[i][j]
        return samples

class MMD_Linear(Estimator):
    """
    Linear time estimator.
    """
    def __init__(self, kernel):
        self.kernel = kernel

    def estimates(self, X, Y, dim=False):
        """Compute linear mmd estimator and a linear estimate of
            the uncentred 2nd moment of h(z, z'). Total cost: O(n).
            Code from https://github.com/wittawatj/interpretable-test/
        """
        if dim:
            d = Y.shape[1]
            n = Y.shape[0]
            h_b = np.zeros((d,self.n_estimates(n)))
            for i in range(d):
                h_b[i,:] = self.estimates(X[:,i,np.newaxis], Y[:,i,np.newaxis], dim=False).flatten()
            return h_b
        kernel = self.kernel
        if X.shape[0] != Y.shape[0]:
            raise ValueError('Require sample size of X = size of Y')
        n = X.shape[0]
        if n%2 == 1:
            # make it even by removing the last row
            X = np.delete(X, -1, axis=0)
            Y = np.delete(Y, -1, axis=0)

        Xodd = X[::2, :]
        Xeven = X[1::2, :]
        assert Xodd.shape[0] == Xeven.shape[0]
        Yodd = Y[::2, :]
        Yeven = Y[1::2, :]
        assert Yodd.shape[0] == Yeven.shape[0]
        # linear mmd. O(n)
        xx = kernel.pair_eval(Xodd, Xeven)
        yy = kernel.pair_eval(Yodd, Yeven)
        xo_ye = kernel.pair_eval(Xodd, Yeven)
        xe_yo = kernel.pair_eval(Xeven, Yodd)
        h = xx + yy - xo_ye - xe_yo
        return h

    def compute(self, x,y, dim=True):
        d = x.shape[1]
        n = x.shape[0]
        if dim:
            mmd = np.zeros(d)
            for i in range(d):
                x_dim = x[:,i,np.newaxis] if x.ndim == 2 else x[:,i]
                y_dim = y[:,i,np.newaxis] if y.ndim == 2 else y[:,i]
                mmd[i] = np.sum(self.estimates(x_dim, y_dim))/self.n_estimates(n)
        else:
            h = self.estimates(x,y)
            mmd = np.sum(h)/self.n_estimates(n)
        return mmd

    def compute_var(self, X, Y):
        m = X.shape[0]
        H_b = self.estimates(X, Y).flatten()
        sigma = np.var(H_b)
        return sigma
    
    def compute_cov(self, x,y):
        d = x.shape[1]
        n = x.shape[0]
        H_b = np.zeros((d,self.n_estimates(n)))
        for i in range(d):
            x_dim = x[:,i,np.newaxis] if x.ndim == 2 else x[:,i]
            y_dim = y[:,i,np.newaxis] if y.ndim == 2 else y[:,i]
            H_b[i,:] = self.estimates(x_dim, y_dim).flatten()
        sigma = np.cov(H_b)
        return sigma

    def n_estimates(self, m):
        return int(m/2)

def get_cross_covariance(X, Y, Z, k):
    """
    Code from: https://github.com/wittawatj/kernel-mod
    Compute the covariance of the U-statistics for two MMDs
    (Bounliphone, et al. 2016, ICLR)
    Args:
        X: numpy array of shape (nx, d), sample from the model 1
        Y: numpy array of shape (ny, d), sample from the model 2
        Z: numpy array of shape (nz, d), sample from the reference
        k: a kernel object
    Returns:
        cov: covariance of two U stats
    """
    Kzz = k.eval(Z, Z)
    # Kxx
    Kzx = k.eval(Z, X)
    # Kxy
    Kzy = k.eval(Z, Y)
    # Kxz
    Kzznd = Kzz - np.diag(np.diag(Kzz))
    # Kxxnd = Kxx-diag(diag(Kxx));

    nz = Kzz.shape[0]
    nx = Kzx.shape[1]
    ny = Kzy.shape[1]
    # m = size(Kxx,1);
    # n = size(Kxy,2);
    # r = size(Kxz,2);

    u_zz = (1./(nz*(nz-1))) * np.sum(Kzznd)
    u_zx = np.sum(Kzx) / (nz*nx)
    u_zy = np.sum(Kzy) / (nz*ny)
    # u_xx=sum(sum(Kxxnd))*( 1/(m*(m-1)) );
    # u_xy=sum(sum(Kxy))/(m*n);
    # u_xz=sum(sum(Kxz))/(m*r);

    ct1 = 1./(nz*(nz-1)**2) * np.sum(np.dot(Kzznd,Kzznd))
    # ct1 = (1/(m*(m-1)*(m-1)))   * sum(sum(Kzznd*Kzznd));
    ct2 = u_zz**2
    # ct2 =  u_xx^2;
    ct3 = 1./(nz*(nz-1)*ny) * np.sum(np.dot(Kzznd,Kzy))
    # ct3 = (1/(m*(m-1)*r))       * sum(sum(Kzznd*Kxz));
    ct4 = u_zz * u_zy
    # ct4 =  u_xx*u_xz;
    ct5 = (1./(nz*(nz-1)*nx)) * np.sum(np.dot(Kzznd, Kzx))
    # ct5 = (1/(m*(m-1)*n))       * sum(sum(Kzznd*Kxy));
    ct6 = u_zz * u_zx
    # ct6 = u_xx*u_xy;
    ct7 = (1./(nx*nz*ny)) * np.sum(np.dot(Kzx.T, Kzy))
    # ct7 = (1/(n*m*r))           * sum(sum(Kzx'*Kxz));
    ct8 = u_zx * u_zy
    # ct8 = u_xy*u_xz;

    zeta_1 = (ct1-ct2)-(ct3-ct4)-(ct5-ct6)+(ct7-ct8)
    # zeta_1 = (ct1-ct2)-(ct3-ct4)-(ct5-ct6)+(ct7-ct8);
    cov = (4.0*(nz-2))/(nz*(nz-1)) * zeta_1
    # theCov = (4*(m-2))/(m*(m-1)) * zeta_1;

    return cov

def mmd_med_heuristic(models, ref, subsample=1000, seed=100):
    # subsample first
    n = ref.shape[0]
    assert subsample > 0
    sub_models = []
    with util.NumpySeedContext(seed=seed):
        ind = np.random.choice(n, min(subsample, n), replace=False)
        for i in range(len(models)):
            sub_models.append(models[i][ind,:])
        sub_ref = ref[ind,:]

    med_mz = np.zeros(len(sub_models))
    for i, model in enumerate(sub_models):
        sq_pdist_mz = util.dist_matrix(model, sub_ref)**2
        med_mz[i] = np.median(sq_pdist_mz)**0.5

    sigma2 = 0.5*np.mean(med_mz)**2
    return sigma2
