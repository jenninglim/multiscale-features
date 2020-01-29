import numpy as np
import logging
import multiprocessing as mp
import os
import sys
import time
import matplotlib.pyplot as plt
import argparse
from functools import partial
import logging

from mskernel import hsic
from mskernel import kernel
from mskernel import util
from mskernel.featsel import MultiSel, PolySel

### Problems
from problem import *

os.system("taskset -p 0xff %d" % os.getpid())
nsamples_lin = [500, 1000, 1500, 2000]

def one_trial(i, n_samples, algsel, problem, n_select, hsic_e, params):
    p,r = problem.sample(n_samples, i)

    p_bw = util.meddistance(p, subsample=1000) **2
    r_bw = util.meddistance(r, subsample=1000) **2
    hsic_e = hsic_e(kernel.KGauss(p_bw),kernel.KGauss(r_bw), params(n_samples))

    feat_select = algsel(hsic_e, params=True)

    results = feat_select.test(p,r, args=n_select, seed=i)

    ## True selected features.
    if results['sel_vars'].shape[0] > 1:
        true = problem.is_true(results['sel_vars'])
        n_true = np.sum(true)
        fpr = np.sum(results['h0_rejs'][np.logical_not(true)])/max(n_select-n_true,1)
        tpr = np.sum(results['h0_rejs'][true])/max(n_true,1) 
    else:
        tpr, fpr = 0, 0 
    logging.debug("TPR is :{0:.3f}  FPR is :{1:.3f}".format(tpr, fpr))
    return tpr, fpr

def independent_job(problem, n_samples, n_select, algsel, hsic_e, pool, n_repeats,
        params):
    result = pool.map_async(partial(one_trial,
        n_samples=n_samples,
        algsel=algsel,
        problem=problem,
        n_select=n_select,
        hsic_e=hsic_e,
        params=params,
        ),
        [i for i in range(n_repeats)])
    res = result.get()

    tpr, fpr = zip(*res)

    tpr = { 'mean': np.sum(tpr)/n_repeats,
            'sd' : np.sqrt(np.var(tpr)/n_repeats) }
    fpr = { 'mean': np.sum(fpr)/n_repeats,
            'sd' : np.sqrt(np.var(fpr)/n_repeats)}

    logging.debug("Overall TPR is :{0:.3f} ".format(tpr['mean']) + " " + \
            "Its variance is :{0:.3f} ".format(tpr['sd']))

    logging.debug("Overall FPR is :{0:.3f} ".format(fpr['mean']) + " " + \
            "Its variance is :{0:.3f} ".format(fpr['sd']))
    return tpr, fpr 

def params_block(n):
    return 5#int(np.floor(np.sqrt(n)))

def params_inc(n):
    return 1

def runExperiments(problem, n_select, n_dim, algorithm, dist, n_repeats, threads):
    assert(n_select <= n_dim)

    if algorithm == 'MultiSel':
        algsel = MultiSel
    elif algorithm == 'PolySel':
        algsel = PolySel
    else:
        assert(0==1)

    if 'Logit' in problem:
        from math import floor
        n_true = 10
        sampler = Logit(n_true, n_dim)
    else:
        assert(0==1)

    if 'Inc' in dist:
        hsic_e = hsic.HSIC_Inc
        params = params_inc
    elif 'Block' in dist:
        hsic_e = hsic.HSIC_Block
        params = params_block
    else:
        assert(0==1)

    ## Distributed trials.
    with mp.Pool(threads) as pool:

        results = []

        for n_samples in nsamples_lin:

            results.append(independent_job(sampler,
                n_samples,
                n_select,
                algsel,
                hsic_e,
                pool,
                n_repeats,
                params))

    tpr, fpr = zip(*results)

    setting = { 'dim': n_dim,
                'nsample_lin' : nsamples_lin,
                'problem': problem,
                'algorithm': algorithm,
                'HSIC' : dist,
                'n_select': n_select}

    results = { 'setting':setting,
                'x_lin' : nsamples_lin,
                'fpr'   : fpr,
                'tpr'   : tpr,}
    
    logging.info("Setting : {0}".format(setting))
    logging.info("Results : {0}".format(results))

    np.save("temp/Ex4_{0}_{1}_{2}_{3}_{4}_{5}".format(n_repeats,
            problem,
            dist,
            algorithm,
            n_select,
            n_dim),
        results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p','--problem',
            default = 'Logit',
            type = str,
            help = 'Problem setting: MS, VS.')

    parser.add_argument('-s','--n_select',
            default = 30,
            type = int,
            help = 'Number of features to select')

    parser.add_argument('-d','--n_dim',
            default = 50,
            type = int,
            help = 'Number of dimension of the problem')

    parser.add_argument('-a','--algorithm',
            default = 'PolySel',
            type = str,
            help = 'Algorithm to use: PolySel, MultiSel.')

    parser.add_argument('-r','--n_repeats',
            default = 100,
            type = int,
            help = 'Number of trials or "r"epeats.')
    
    parser.add_argument('-t','--threads',
            default = 5,
            type = int,
            help = 'Number of threads.')

    parser.add_argument('-e','--estimator',
            default = 'Block',
            type = str,
            help = 'MMD Estimator: Inc, Block')

    parser.add_argument('-v','--verbose',
            default = 0,
            type = int,
            help = 'Verbose level: 0, 1, 2.')

    args = parser.parse_args()

    if args.verbose == 0:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    elif args.verbose == 2:
        logging.getLogger().setLevel(logging.DEBUG)

    runExperiments(
            args.problem,
            args.n_select,
            args.n_dim,
            args.algorithm,
            args.estimator,
            args.n_repeats,
            args.threads)


