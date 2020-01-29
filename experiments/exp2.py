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

from mskernel import mmd, hsic
from mskernel import kernel
from mskernel import util
from mskernel.featsel import MultiSel, PolySel

### Problems
from problem import *

Benchmark = None #Benchmark_HSIC
mmd_or_hsic = None
os.system("taskset -p 0xff %d" % os.getpid())
nsamples_lin = [500, 1000, 1500, 2000]

def one_trial(i, n_samples, problem, n_select, algo):
    p,r = problem.sample(n_samples, i)

    if 'MMD' in mmd_or_hsic:
        bw = util.meddistance(np.vstack((p, r)), subsample=1000) **2
        metric = mmd.MMD_Inc(kernel.KGauss(bw))
    else:
        p_bw = util.meddistance(p, subsample=1000) **2
        r_bw = util.meddistance(r, subsample=1000) **2
        metric = hsic.HSIC_Inc(kernel.KGauss(p_bw), kernel.KGauss(r_bw), 5)

    feat_select = algo(metric)

    results = feat_select.test(p,r, args=n_select, seed=i,)

    ## True selected features.
    if results['sel_vars'].shape[0] > 1:
        true = problem.is_true(results['sel_vars'])
        n_true = np.sum(true)
        fpr = np.sum(results['h0_rejs'][np.logical_not(true)])/max(n_select-n_true,1)
        tpr = np.sum(results['h0_rejs'][true])/max(n_true,1) 
    else:
        tpr, fpr = 0, 0 
    logging.debug("TPR is :{0:.3f}  FPR is :{1:.3f}".format(tpr, fpr, ))
    return tpr, fpr 

def independent_job(problem, n_samples, n_select, pool, n_repeats,
        algo,):
    result = pool.map_async(partial(one_trial,
        n_samples=n_samples,
        problem=problem,
        n_select=n_select,
        algo=algo,
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

def runExperiments(problem, n_select, algorithm, n_repeats, threads, benchmark):

    if 'MMD' in benchmark:
        Benchmark = Benchmark_MMD
    else:
        Benchmark = Benchmark_HSIC

    if 'Pulsar' in problem:
        nsamples_lin = [100]
        sampler = Benchmark(n_fakes = 30,
            path = './dataset/pulsar/pulsar_stars.csv',
            target = 'target_class',
            classes = [0,1])
    elif 'Heart' in problem:
        nsamples_lin = [138]
        sampler = Benchmark(n_fakes=30,
                path='./dataset/heart/heart.csv',
                target='target',
                classes=[0,1])
    elif 'Wine' in problem:
        nsamples_lin = [100]
        sampler = Benchmark(n_fakes=30,
                path='./dataset/wine/winequality.csv',
                target='type',
                classes=['white','red'])
    else:
        assert(0==1)

    if 'MultiSel' in algorithm:
        algo = MultiSel
    elif 'PolySel' in algorithm:
        algo = PolySel
    else:
        assert(0==1)

    ## Distributed trials.
    with mp.Pool(threads) as pool:

        results = []

        for n_samples in nsamples_lin:

            results.append(independent_job(sampler,
                n_samples,
                n_select,
                pool,
                n_repeats,
                algo))

    tpr, fpr = zip(*results)

    setting = { 
                'nsample_lin' : nsamples_lin,
                'problem': problem,
                'algorithm': algorithm,
                'n_select': n_select}

    results = { 'setting':setting,

                'x_lin' : nsamples_lin,
                'fpr'   : fpr,
                'tpr'   : tpr,}
    
    logging.info("Setting : {0}".format(setting))
    logging.info("Results : {0}".format(results))

    np.save("temp/Ex5_{0}_{1}_{2}".format(n_repeats,
            problem,
            algorithm,
            n_select,
            ),
        results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('-p','--problem',
            default = 'Pulsar',
            type = str,
            help = 'Problem setting: MS, VS.')

    parser.add_argument('-s','--n_select',
            default = 30,
            type = int,
            help = 'Number of features to select')

    parser.add_argument('-r','--n_repeats',
            default = 100,
            type = int,
            help = 'Number of trials or "r"epeats.')
    
    parser.add_argument('-t','--threads',
            default = 5,
            type = int,
            help = 'Number of threads.')

    parser.add_argument('-a','--algo',
            default = 'PolySel',
            type = str,
            help = 'PolySel or MultiSel')

    parser.add_argument('-v','--verbose',
            default = 0,
            type = int,
            help = 'Verbose level: 0, 1, 2.')
    
    parser.add_argument('-b','--benchmark',
            default = "MMD",
            type = str,
            help = 'Either MMD or HSIC.')

    args = parser.parse_args()

    if args.verbose == 0:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    elif args.verbose == 2:
        logging.getLogger().setLevel(logging.DEBUG)
    
    mmd_or_hsic= args.benchmark

    runExperiments(
            args.problem,
            args.n_select,
            args.algo,
            args.n_repeats,
            args.threads,
            args.benchmark)


