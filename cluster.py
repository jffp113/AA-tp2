# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans,DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn import mixture, cluster
import numpy as np

def gmm(data):
    return mixture.GaussianMixture(
        n_components=4, covariance_type='full').fit_predict(data)

def birch(data):
    return cluster.Birch(n_clusters=4).fit_predict(data)

def kmean_cluster(data, k=4,max_iter=300):
    return KMeans(n_clusters=k,max_iter=max_iter).fit_predict(data)

def dbsan_cluster(data,min_samples=5,eps = 450):
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)
     
def tunning_eps(data):
    zeros = [0] * 563
    neigh = KNeighborsClassifier(n_neighbors=5).fit(X=data,y=zeros)
    distances, indices = neigh.kneighbors(data)
    
    sort = np.sort(distances, axis=0)
    print(distances)
    e = 0
    for i in range(0,562):
        e = e + sort[i][4]
    
    e = e / 562
    print(f"distance:{e}")
    return e