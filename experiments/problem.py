import numpy as np
import os
import wget
import zipfile
import logging
import pandas as pd

from scipy.stats import bernoulli
from sklearn.preprocessing import OneHotEncoder

from mskernel import util

class MeanShift():
    def __init__(self, n_dim, n_change=10):
        self.n_dim = n_dim
        self.n_change = n_change
        mu_change = np.zeros(n_dim)
        for i in range(n_change):
            mu_change[i] = 0.5
        self.mu_change = mu_change


    def sample(self, n_samples, seed):
        n_dim = self.n_dim
        with util.NumpySeedContext(seed=seed+2):
            p = np.random.multivariate_normal(self.mu_change, np.eye(n_dim),size=(n_samples))

        with util.NumpySeedContext(seed=seed+5):
            r = np.random.multivariate_normal(np.zeros(n_dim), np.eye(n_dim),size=(n_samples))
        return p, r

    def is_true(self, s_feats):
        true = s_feats < self.n_change
        return true

class VarianceShift():
    def __init__(self, n_dim, n_change=10):
        self.n_dim = n_dim
        self.n_change = n_change
        v_change = np.ones(n_dim)
        for i in range(n_change):
            v_change[i] = 1.5
        self.v_change = v_change


    def sample(self, n_samples, seed):
        n_dim = self.n_dim
        with util.NumpySeedContext(seed=seed+2):
            p = np.random.multivariate_normal(np.zeros(n_dim), np.eye(n_dim),size=(n_samples))

        with util.NumpySeedContext(seed=seed+5):
            r = np.random.multivariate_normal(np.zeros(n_dim), np.diag(self.v_change),size=(n_samples))
        return p, r

    def is_true(self, s_feats):
        true = s_feats < self.n_change
        return true

class Benchmark_MMD():
    def __init__(self,
            n_fakes=30,
            path=None,
            target=None,
            classes=[0,1]):
        # If needed download and unzip dataset.
        if not os.path.isdir(path):
            '''
            url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00326/TV_News_Channel_Commercial_Detection_Dataset.zip' 
            location = path + '/dataset.zip'
            os.makedirs(path)
            wget.download(url, location)
            zf = zipfile.ZipFile(location)
            zf.extractall(path)
            '''
        self.df = pd.read_csv(path)
        self.n_fakes = n_fakes
        self.n_true = self.df.shape[1]-1
        self.target = target
        self.classes = classes
        self.df.replace('', np.nan, inplace=True)
        self.df = self.df.dropna()

    def sample(self, n_samples, seed=5):
        l_samples = []
        for i, c in enumerate(self.classes):
            df_samples = self.df[self.df[self.target]==c].drop(self.target,axis=1).sample(n_samples,
                random_state=seed+i).values
            with util.NumpySeedContext(seed=seed * (i+1)):
                fakes = np.random.randn(n_samples, self.n_fakes)
            l_samples.append(np.hstack((df_samples,fakes)))
        return l_samples

    def is_true(self, s_feats):
        true = s_feats < self.n_true
        return true

class Benchmark_HSIC():
    def __init__(self,
            n_fakes=30,
            path=None,
            target=None,
            classes=[0,1],
            **args):
        # If needed download and unzip dataset.
        if not os.path.isdir(path):
            '''
            url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00326/TV_News_Channel_Commercial_Detection_Dataset.zip' 
            location = path + '/dataset.zip'
            os.makedirs(path)
            wget.download(url, location)
            zf = zipfile.ZipFile(location)
            zf.extractall(path)
            '''
        self.df = pd.read_csv(path, **args)
        self.n_fakes = n_fakes
        self.n_true = self.df.shape[1]-1
        self.target = target
        self.classes = classes
        enc = OneHotEncoder()
        y = np.expand_dims(self.df[self.target].values,
                axis=1)
        enc.fit(y)
        self.enc = enc
        self.df.replace('', np.nan, inplace=True)
        self.df.replace('?', np.nan, inplace=True)
        self.df = self.df.dropna()

    def sample(self, n_samples, seed=5):
        l_samples = []
        target_indices = np.isin(self.df[self.target].values, self.classes)
        with util.NumpySeedContext(seed=seed):
            df_samples = self.df[target_indices].sample(n_samples,
                    random_state=seed)
            fakes = np.random.randn(n_samples, self.n_fakes)
            y = np.expand_dims(df_samples[self.target].values,
                    axis=1)
            y_enc = self.enc.transform(y).toarray()
            x = df_samples.drop(self.target,axis=1).values
            x = np.hstack((x,fakes)).astype(float)
        return x, y_enc

    def is_true(self, s_feats):
        true = s_feats < self.n_true
        return true

class CelebA():
    def __init__(self, model_classes_mix, ref_classes_mix):
        # Download dataset
        feature_url = "http://ftp.tuebingen.mpg.de/pub/is/wittawat/kmod_share/problems/celeba/inception_features/"
        feature_dir = "./dataset/celeba_features"
        celeba_classes = ['gen_smile', 'gen_nonsmile', 'ref_smile', 'ref_nonsmile']

        if not os.path.isdir(feature_dir):
            logging.warning("Downloading dataset can take a long time")
            os.makedirs(feature_dir)
            for celeba_class in  celeba_classes:
                filename= '{}.npy'.format(celeba_class)
                npy_path = os.path.join(feature_dir, filename)
                url_path = os.path.join(feature_url, filename)
                download_to(url_path,npy_path)

        celeba_features = []
        for celeba_class in  celeba_classes:
            filename= '{}.npy'.format(celeba_class)
            npy_path = os.path.join(feature_dir, filename)
            celeba_features.append(np.load(npy_path))
        self.celeba_features =  dict(zip(celeba_classes, celeba_features))
        self.model_classes_mix = model_classes_mix
        self.ref_classes_mix = ref_classes_mix
    
    def is_true(self, s_feats):
        true = s_feats < 0
        return true

    def sample(self, n_samples, seed=5):
        ## DISJOINT SET
        model_features = {}
        ref_samples = []
        with util.NumpySeedContext(seed=seed):
            ## FOR EACH CELEBA CLASS
            for key, features in self.celeba_features.items():
                # CALCULATE HOW MUCH SHOULD BE IN THE REFERENCE POOL
                n_ref_samples = int(np.round(self.ref_classes_mix[key] * n_samples))
                random_features = np.random.permutation(features)

                ## FOR THE CANDIDATE MODELS
                model_features[key] = random_features[n_ref_samples:]
                ## FOR THE REFERENCE
                ref_samples.append(random_features[:n_ref_samples])

        ## samples for models
        model_samples = []
        for j,class_ratios in enumerate(self.model_classes_mix):
            model_class_samples = []
            for i, data_class in enumerate(class_ratios.keys()):
                n_class_samples = int(np.round(class_ratios[data_class] * n_samples))
                seed_class = i*n_samples+seed*j
                with util.NumpySeedContext(seed=seed_class):
                    indices = np.random.choice(model_features[data_class].shape[0], n_class_samples)
                model_class_samples.append(model_features[data_class][indices])
            class_samples = dict(zip(class_ratios.keys(),model_class_samples))
            model_class_stack = np.vstack(list(class_samples.values()))
            model_samples.append(model_class_stack)
            #assert model_class_stack.shape[0] == n_samples, "Sample size mismatch: {0} instead of {1}".format(samples.shape[0],n)
        with util.NumpySeedContext(seed=seed+5):
            ref_samples = np.random.permutation(np.vstack(ref_samples))
            model_samples = [np.random.permutation(samples) for samples in model_samples]
        assert ref_samples.shape[0] == n_samples, \
                "Sample size mismatch: {0} instead of {1}".format(samples.shape[0],n)
        return np.stack(model_samples,axis=1),\
                np.repeat(ref_samples[:,np.newaxis] ,axis=1, repeats=len(model_samples))

class Logit():
    def __init__(self, n_true, n_dim):
        assert(n_true <= n_dim)
        self.n_true = n_true
        self.n_dim = n_dim

    def sample(self, n_samples, seed=5):
        with util.NumpySeedContext(seed=seed):
            x = np.random.randn(n_samples, self.n_dim)
        x_true = x[:,:self.n_true]
        p = np.expand_dims(np.exp(np.sum(x_true, axis=1))/(1 + np.exp(np.sum(x_true, axis=1))),1)
        with util.NumpySeedContext(seed=seed+1):
            y_bern = bernoulli.rvs(p=p,size=(p.shape[0],1))
        return x, y_bern

    def is_true(self, select):
        return select < self.n_true

def download_to(url, file_path):
    """
    Download the file specified by the URL and save it to the file specified
    by the file_path. Overwrite the file if exist.
    """
    # see https://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3
    import urllib.request
    import shutil
    # Download the file from `url` and save it locally under `file_name`:
    with urllib.request.urlopen(url) as response, \
            open(file_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)


