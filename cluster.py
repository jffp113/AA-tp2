# -*- coding: utf-8 -*-

def kmean_cluster(data):
    return cluster.KMeans(n_clusters=3).fit_predict(data)