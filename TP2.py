# -*- coding: utf-8 -*-

from extraction import extract_feature
from tp2_aux import images_as_matrix,report_clusters
from cluster import kmean_cluster,dbsan_cluster
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

FEATURES = 18
SIZE = 564
N_IMAGES = 563
EXTRACT = False


def extract_features_or_read_file(data):    
    if EXTRACT:
            ft = extract_feature(data)
            np.savez(f"ft/extracted", ft)
            return ft
    else:
            return np.load("ft/extracted.npz")['arr_0']

    
selected_features = [0,1,2,3,4,5,11,12,13,14,15,16,17]
def main():
    data = images_as_matrix()
    featuresPerImage = extract_features_or_read_file(data)[:,selected_features]
    report_clusters(np.linspace(0, N_IMAGES, num=N_IMAGES), kmean_cluster(featuresPerImage), "kmean.html")
    report_clusters(np.linspace(0, N_IMAGES, num=N_IMAGES), dbsan_cluster(featuresPerImage), "dbsan.html")

    return featuresPerImage

t = main()

#from sklearn.feature_selection import f_classif


#l = np.loadtxt("labels.txt",skiprows=1,delimiter=",")
#X = t[l[:,1] > 0]
#y = l[l[:,1] > 0][:,1]
#f,prob = f_classif(X,y)


import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity


l = np.loadtxt("labels.txt",skiprows=1,delimiter=",")

#0,2,8,12,14,9,
#0 tudo
#2 tudo mas só um bocadinho de verde


def fill_histogram(t):
    for i in range(0,18):
        x = t[:,[i]]
        x_plot1 = np.linspace(min(t[:,i][l[:,1] == 1]), max(t[:,i][l[:,1] == 1]), (l[:,1] == 1).shape[0])[:, np.newaxis]
        x_plot2 = np.linspace(min(t[:,i][l[:,1] == 2]), max(t[:,i][l[:,1] == 2]), (l[:,1] == 2).shape[0])[:, np.newaxis]
        x_plot3 = np.linspace(min(t[:,i][l[:,1] == 3]), max(t[:,i][l[:,1] == 3]), (l[:,1] == 3).shape[0])[:, np.newaxis]
        
        
        bd1 = (max(t[:,i][l[:,1] == 1])- min(t[:,i][l[:,1] == 1])) / 10
        bd2 = (max(t[:,i][l[:,1] == 2]) - min(t[:,i][l[:,1] == 2])) / 10
        bd3 = (max(t[:,i][l[:,1] == 3])- min(t[:,i][l[:,1] ==3])) / 10
        
        kde1 = KernelDensity(kernel='gaussian',bandwidth=bd1).fit(x[l[:,1] == 1])
        kde2 = KernelDensity(kernel='gaussian',bandwidth=bd2).fit(x[l[:,1] == 2])
        kde3 = KernelDensity(kernel='gaussian',bandwidth=bd3).fit(x[l[:,1] == 3])
        kde1 = kde1.score_samples(x_plot1)
        kde2 = kde2.score_samples(x_plot2)
        kde3 = kde3.score_samples(x_plot3)
        fig,ax = plt.subplots(figsize=(8,6))
        ax.set_xlabel(f"{i}")
        ax.fill_between(x_plot1[:,0],0, np.exp(kde1),fc='#AAAAFF',alpha=0.5)
        ax.fill_between(x_plot2[:,0],0, np.exp(kde2),fc='#AAFFAA',alpha=0.5)
        ax.fill_between(x_plot3[:,0],0, np.exp(kde3),fc='#FFAAAA',alpha=0.5)
        plt.savefig(f"figft/{i}.png",dpi=300)
        plt.close()

#fill_histogram(t)


def pointsPlot(t):
    for i in range(0,18):
        for j in range(0,18):
            x1 = t[:,i][l[:,1] == 1]
            x2 = t[:,i][l[:,1] == 2]
            x3 = t[:,i][l[:,1] == 3]
            y1 = t[:,j][l[:,1] == 1]
            y2 = t[:,j][l[:,1] == 2]
            y3 = t[:,j][l[:,1] == 3]
            fig,ax = plt.subplots(figsize=(8,6))
            ax.plot(x1,y1,'ro')
            ax.plot(x2,y2,'go')
            ax.plot(x3,y3,'bo')
            ax.plot(t[:,i],t[:,j],'ro',alpha=0.1)
            plt.savefig(f"figft/{i}_{j}.png",dpi=300)
            plt.close()
#pointsPlot(t)
    
def tunning_eps(data):
    zeros = [0] * 563
    neigh = KNeighborsClassifier(n_neighbors=5).fit(X=data,y=zeros)
    distances, indices = neigh.kneighbors(data)
    fig,ax = plt.subplots(figsize=(8,6))
    sort = np.sort(distances, axis=0)
    ax.plot(np.linspace(0,N_IMAGES,num=N_IMAGES),sort[:,4])
    
tunning_eps(t)  

from sklearn.feature_selection import f_classif


l = np.loadtxt("labels.txt",skiprows=1,delimiter=",")
X = t[l[:,1] > 0]
y = l[l[:,1] > 0][:,1]
f,prob = f_classif(X,y)
a = np.sort(f)