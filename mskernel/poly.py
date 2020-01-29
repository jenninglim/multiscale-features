import numpy as np
from scipy.stats import truncnorm
import logging
"""
Functions for calculating the paramters of a truncated normal of
https://arxiv.org/abs/1311.6238.
"""

def selection(Z, n_select):
    """
        Characterising selecting the top K features from vector Z with the
        highest values.
        input
            Z      : "Feature vector" with a normal distribution.
            K      :  Number of selections.
        return
            ind_sel: Selected index.
            A,b    : The linear combination of the selection event Az < b.
    """
    n = np.shape(Z)[0]
    ## Sorted list of Z. Descending order.
    ind_sorted = np.argsort(Z)[::-1]
    ## Pick top k
    ind_sel = ind_sorted[:n_select]
    ind_nsel = ind_sorted[n_select:]

    A = np.zeros(((n-n_select)*n_select,n))
    for i, sel in enumerate(ind_sel):
        for j, nsel in enumerate(ind_nsel):
            index = i * (n-n_select) + j
            A[index, nsel] = 1
            A[index, sel] = -1
    b = np.zeros((n-n_select)*n_select)
    assert(np.all(np.matmul(A,Z) <= b))
    return ind_sel, A, b

def psi_inf(A,b,eta, mu, cov, z):
    """
        Returns the p-value of the truncated normal. The mean,
        variance, and truncated points [a,b] is determined by Lee et al 2016.

    """
    l_thres, u_thres= calculate_threshold(z, A, b, eta, cov)
    sigma = np.dot(eta,np.dot(cov,eta))
    scale = np.sqrt(sigma)

    params = {"u_thres":u_thres,
              "l_thres":l_thres,
              "mean": np.matmul(eta,mu),
              "scale":scale,
              }

        # Weird numerical instability.
    return lambda x: 1-truncnorm.cdf(x,
                        l_thres/scale,
                        u_thres/scale,
                        loc=np.matmul(eta,mu),
                        scale=scale), params
    '''
    return lambda x: truncnorm_ppf(x,
                        l_thres,
                        u_thres,
                        loc=np.matmul(eta,mu),
                        scale=scale), params
    '''

def calculate_threshold(z, A, b, eta, cov):
    """
        Calculates the respective threshold for the method PSI_Inf.
    """
    etaz = eta.dot(z)
    Az = A.dot(z)
    Sigma_eta = cov.dot(eta)
    deno = Sigma_eta.dot(eta)
    alpha = A.dot(Sigma_eta)/deno
    assert(np.shape(A)[0] == np.shape(alpha)[0])
    pos_alpha_ind = np.argwhere(alpha>0).flatten()
    neg_alpha_ind = np.argwhere(alpha<0).flatten()
    acc = (b - np.matmul(A,z))/alpha+np.matmul(eta,z)
    if (np.shape(neg_alpha_ind)[0] > 0):
        l_thres = np.max(acc[neg_alpha_ind])
    else:
        l_thres = -10.0**10
    if (np.shape(pos_alpha_ind)[0] > 0):
        u_thres = np.min(acc[pos_alpha_ind])
    else:
        u_thres= 10**10
    return l_thres, u_thres

def test_significance(stat, A, b, eta, mu, cov, z, alpha):
    """
        Compute an p-value by testing a one-tail.
        Look at right tail or left tail?
        Returns "h_0 Reject
    """
    ppf, params = psi_inf(A, b, eta, mu, cov, z)
    if np.isnan(params['scale']) or not np.isreal(params['scale']):
        logging.warning("Scale is not real or negative, test reject")
        return False, params
    threshold = ppf(1.-alpha)
    return stat > threshold, params

def generateEta(ind_sel, n_models):
    """
        Generate multiple etas corresponding to testing
        within the selected indices.
    """
    etas = np.zeros((n_models-1, n_models))
    for i in range(n_models-1):
        index = i if i < ind_sel else i +1
        etas[i,ind_sel] = -1
        etas[i,index]=1
    return etas

def truncnorm_ppf(x, a, b,loc=0., scale=1.):
    thres = truncnorm.ppf(x,(a-loc)/scale,(b-loc)/scale,loc=loc, scale=scale)
    if np.any(np.isnan(thres)) or np.any(np.isinf(thres)):
        logging.info("Threshold is Nan using approximations.")
        thres = loc+scale*quantile_tn(x,(a-loc)/scale,(b-loc)/scale)
    return thres

def quantile_tn(u,a,b,threshold=0.0005):
    """ Approximate quantile function in the tail region
    https://www.iro.umontreal.ca/~lecuyer/myftp/papers/truncated-normal-book-chapter.pdf
    """
    q_a = q(a)
    q_b = q(b)
    c =q_a * (1- u) + q_b * u * np.exp((a**2 - b**2)/2)
    d_x = 100
    z = 1 - u + u *  np.exp((a**2 - b**2)/2)
    x = np.sqrt(a**2 - 2 * np.log(z))
    while d_x > threshold and not np.isnan(d_x):
        z = z - x * (z * q(x) - c)
        x_new = np.sqrt(a**2 - 2 * np.log(z))
        d_x = np.abs(x_new - x)/x
        x = x_new
    return x

def q(x, r=10):
    acc=0
    for i in range(r):
        acc = acc + (2*i-1)/((-1)**i*x**(2*i+1))
    return 1/x + acc
