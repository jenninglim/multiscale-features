import numpy as np
import logging
from scipy.stats import norm
from operator import itemgetter

from mskernel.model import LinearModel, PolyModel
from mskernel import util

class Multiscale():
    """
    Multiscale Bootstrap procedure.
    """
    def __init__(self, l_samples, region, n_scalings=3, params=None):
        """
        l_samples: list of samples, each of shape n x d.
        region: f: X-> {0,1}
        n_scalings: number of different scaling (changing the number of samples
                    for the estimator.)
        """
        self.region = region
        self.n_samples = l_samples[0].shape[0]
        self.n_scalings = (np.exp(np.linspace(np.log(1/10),np.log(10),n_scalings)) \
                * l_samples[0].shape[0]).astype(int)
        self.l_samples = l_samples
        if params == None:
            self.bs_proc = lambda x,y,z: np_bs_n_samples(l_samples,
                                            x,
                                            y,
                                            z)
        else:
            self.bs_proc = lambda x, y, z: p_bs_n_samples(params,
                                            x,
                                            y,
                                            z)
    
    def fit_model(self, phis, finite_scalings, n_degrees=3):
        """
            Get Parameters of a polynomial function to extrapolate to \sigma^2=-1.
        """
        sigma2s=self.n_samples/np.array(finite_scalings)[:,np.newaxis]
        linear = LinearModel(sigma2s, phis)
        models = [linear]
        for i in range(n_degrees):
            models.append(PolyModel(sigma2s, phis, degree=2+i))
        aics = [model.aic for model in models]
        ind = min(enumerate(aics), key=itemgetter(1))[0]
        return models[ind]

    def phi_model(self, region, n_bootstrap=10000, seed=10, plot=False):
        ## Bootstrap from data X'
        phi_probs = []
        finite_scalings =[]

        # Fit a parametric model for the bootstrap region of "region"
        ## For each scale caluclate the "normalised bootstrap probability"
        for i, n_scaling in enumerate(self.n_scalings):
            bs_p = bs_prob(self.bs_proc,
                    n_scaling,
                    n_bootstrap,
                    region,
                    seed=len(self.n_scalings)*seed+i)
            if not bs_p == 0 and not bs_p == 1:
                phi_probs.append(phi(bs_p, n_scaling, self.n_samples))
                finite_scalings.append(n_scaling)
        
        if len(finite_scalings) <=2:
            return None, False

        ## Fit model
        model= self.fit_model(phi_probs, finite_scalings)

        ## Basic plotting
        if plot:
            sigma2s=self.n_samples/np.array(finite_scalings)[:,np.newaxis]
            import matplotlib.pyplot as plt
            x_lin = np.linspace(-1, np.max(sigma2s))
            plt.plot(x_lin, model.f(x_lin))
            plt.scatter(sigma2s, phi_probs)
            plt.show()
        return model, True

    def pvalue(self,seed=1, n_bootstrap=10000, plot=False, stat=None):
        """
            Compute p-value
        """
        ## Create test region model
        if stat !=None:
            return norm.sf(stat)
        model, flag = self.phi_model(self.region, seed=seed, plot=plot)

        ## Extrapolate to -1
        if flag:
            xtrop= model.f(-1)
            return norm.sf(xtrop)
        else:
            return 1

def phi(bs_prob, n_s, n):
    """
        Function Phi is defined in Shimodaira et. al.
    """
    return np.sqrt(n/n_s) *norm.ppf(1- bs_prob)

def np_bs_n_samples(l_samples, n_samples, n_bs, seed):
    """
        Nonparametric bootstrap samples from l_samples.
        n_samples: Number of samples to bootstrap from l_samples.
        n_bs     : Number of bootstrap iterations.
    """
    resamples = []
    for i, samples in enumerate(l_samples):
        b_samples = []
        with util.NumpySeedContext(seed=(i+1)*seed+i):
            resamples.append(samples[np.random.choice(samples.shape[0],size=(n_bs,n_samples)),:])
    return resamples

def p_bs_n_samples(params, n_samples, n_bs, seed):
    """
        Parametric bootstrap samples
        params: a tuple (mu, sigma) describing the parameters of a normal.
        n_samples: number of samples to sample from the normal distribution.
        seed:seed number for reproducability. 
        
        Note that Sigma does NOT carries (1/n)
    """
    mu, sigma = params[0], params[1]
    with util.NumpySeedContext(seed=seed):
        norm_rvs = np.sqrt(sigma/n_samples)* np.random.randn(n_bs,1) + mu
    return norm_rvs

def bs_prob(bs_proc, n_samples, n_bootstrap, region, seed=5):
    """
        Calculate bootstrap probability of X in region.
    """
    bs_samples = bs_proc(n_samples,n_bootstrap, seed)
    c = np.sum(region(bs_samples))
    bs_p = c/n_bootstrap
    """
        TODO: Better fix?
        When bs_p = 0 or 1, since we are using inverse CDF we will deal with
        NaNs and Infs. (VERY BAD).
    """
    return bs_p
