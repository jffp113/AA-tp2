# -*- coding: utf-8 -*-

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE,Isomap
import numpy as np

FEATURES_TO_EXTRACT = 6

"""
     Extract Features using PCA Extracting method
"""
def pca_feature_extract(data):
    pca = PCA(n_components=FEATURES_TO_EXTRACT)
    return pca.fit_transform(data)

"""
     Extract Features using TSNE Extracting method
"""
def tsne_feature_extract(data):
    tsne = TSNE(n_components=FEATURES_TO_EXTRACT,method='exact')
    return tsne.fit_transform(data)

"""
     Extract Features using ISOMAP Extracting method
"""
def isomap_feature_extract(data):
    isomap = Isomap(n_components=FEATURES_TO_EXTRACT)
    return isomap.fit_transform(data)

def extract_feature(data):
    pca = pca_feature_extract(data)
    tsne = tsne_feature_extract(data)
    isomap = isomap_feature_extract(data)
    return np.concatenate((pca, tsne, isomap), axis=1, out=None)