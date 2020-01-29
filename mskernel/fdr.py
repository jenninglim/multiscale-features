import numpy as np
from scipy.stats import chi2
def bh(pvals, alpha):
    s_pvals = np.sort(pvals)
    mapping = [np.where(s_pvals==i)[0] for i in pvals]
    rejs = []
    for i, pval in enumerate(s_pvals):
        rejs.append(pval < i/len(s_pvals) * alpha)
    return rejs

def fisher(pvals, alpha):
    k = len(pvals)
    acc = 0
    for pval in pvals:
        acc = acc - 2*np.log(pval)
    return 1-chi2.cdf(acc, 2*k) < alpha
