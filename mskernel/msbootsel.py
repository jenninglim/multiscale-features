import numpy as np
import logging
from operator import itemgetter
from scipy.stats import norm

from mskernel.model import LinearModel, PolyModel, TaylModel
from mskernel import util

class Multiscale():
    """
    Multiscale Bootstrap procedure.
    """
    def __init__(self, l_samples,
            region,
            select,
            n_scalings=3,
            params=None,
            linear_model=True):
        """
        l_samples: list of samples, each of shape n x d.
        region: f: X-> {0,1}
        select: f: X->ind
        n_scalings: number of different scaling OR 1d array of scalings.
        """
        self.region = region
        self.select = select
        self.n_samples = l_samples[0].shape[0]
        if type(n_scalings) == int:
            self.scalings = (np.exp(np.linspace(np.log(2),np.log(1/2),n_scalings)) \
                    * l_samples[0].shape[0]).astype(int)
        else:
            self.scalings = n_scalings
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
        self.linear_model = linear_model
    
    def fit_model(self, phis, finite_scalings, n_degrees=5):
        """
            Get Parameters of a polynomial function to extrapolate to \sigma^2=-1.
        """
        sigma2s=self.n_samples/np.array(finite_scalings)[:,np.newaxis]
        linear = LinearModel(sigma2s, phis)
        if self.linear_model:
            return linear
        models = [linear]
        for i in range(n_degrees-1):
            models.append(TaylModel(sigma2s, phis, degree=i+1))
        aics = [model.aic for model in models]
        ind = min(enumerate(aics), key=itemgetter(1))[0]
        return models[ind] #TaylModel(sigma2s, phis, degree=4)

    def phi_model(self, region, n_bootstrap=10000, seed=10, plot=False):

        ## Bootstrap from data X'
        phi_probs = []
        finite_scalings =[]

        # Fit a parametric model for the bootstrap region of "region"
        ## For each scale caluclate the "normalised bootstrap probability"
        for i, n_scaling in enumerate(self.scalings):
            bs_p = bs_prob(self.bs_proc,
                    n_scaling,
                    n_bootstrap,
                    region,
                    seed=len(self.scalings)*seed+i)
            if not bs_p == 0 and not bs_p == 1:

                phi_probs.append(phi(bs_p, n_scaling, self.n_samples))
                finite_scalings.append(n_scaling)

        ## Fit model
        if len(finite_scalings) > 0:
            model= self.fit_model(phi_probs, finite_scalings)
        elif bs_p == 0:
            return lambda x: 10**4
        elif bs_p == 1:
            return lambda x: -10**4

        ## Basic plotting
        if plot:
            sigma2s=self.n_samples/np.array(finite_scalings)[:,np.newaxis]
            import matplotlib.pyplot as plt
            x_lin = np.linspace(0, np.max(sigma2s))
            plt.plot(x_lin, model.f(x_lin))
            plt.scatter(sigma2s, phi_probs)
            plt.ylim(np.min(model.f(x_lin)), 0)
            plt.show()
        return model.f

    def pvalue(self, params, seed=1, n_bootstrap=10000, plot=False, stat=None):
        """
            Compute p-value
        """
        ## Create test region model
        if stat ==None:
            test_model = self.phi_model(lambda y, x: self.region(y, x, *params), seed=seed, plot=plot)
            xtrop= test_model(-1)
        else:
            xtrop = stat

        sel_model  = self.phi_model(lambda y,x: self.select(y, x, *params), seed=seed, plot=plot)

        other = sel_model(0)
        pval=norm.sf(xtrop)/norm.sf(xtrop+other)
        return 1 if np.isnan(pval) or pval > 1 else pval

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
    b_samples = []
    for i, samples in enumerate(l_samples):
        with util.NumpySeedContext(seed=(i+1)*seed+i):
            b_samples.append(samples[np.random.choice(samples.shape[0],size=(n_bs, n_samples))])
    return b_samples

def p_bs_n_samples(params, n_samples, n_bs, seed):
    """
        Parametric bootstrap samples
        params: a tuple (mu, sigma) describing the parameters of a normal.
        n_samples: number of samples to sample from the normal distribution.
        seed:seed number for reproducability. 
        
        Note that Sigma carries (1/n)
    """
    mu, sigma, f, n = params
    with util.NumpySeedContext(seed=seed):
        norm_rvs = np.random.multivariate_normal(np.sqrt(f(n_samples))*mu,
                sigma,
                check_valid='raise',
                size=(n_bs))
    return norm_rvs

def bs_prob(bs_proc, n_samples, n_bootstrap, region, seed=5):
    """
        Calculate bootstrap probability of X in region.
    """
    bs_samples = bs_proc(n_samples,n_bootstrap, seed)
    with util.NumpySeedContext(seed=seed+40):
        c = region(bs_samples, n_samples)
    return c
