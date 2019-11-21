# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans,DBSCAN
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def kmean_cluster(data):
    return KMeans(n_clusters=3,max_iter=300).fit_predict(data)

def dbsan_cluster(data):
    e = tunning_eps(data)
    return DBSCAN(eps=e, min_samples=2).fit_predict(data)


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
    