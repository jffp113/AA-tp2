# -*- coding: utf-8 -*-

from extraction import extract_feature
from tp2_aux import images_as_matrix,report_clusters
from cluster import kmean_cluster,dbsan_cluster
import numpy as np

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


def check_cluster(labels):
    l = np.loadtxt("labels.txt",skiprows=1,delimiter=",")
    und = 0;
    right = 0;
    wrong = 0;
    for i in range(N_IMAGES - 1):
        if l[i][1] == 0:
            und = und + 1
        else:
            if l[i][1] == labels[i] + 1:
                right = right + 1
            else:
                wrong = wrong + 1
    print(f"und:{und}")
    print(f"right:{right}")
    print(f"wrong:{wrong}")
    

def main():
    data = images_as_matrix()
    featuresPerImage = extract_features_or_read_file(data)#[:,[15]]
    report_clusters(np.linspace(0, N_IMAGES, num=N_IMAGES), kmean_cluster(featuresPerImage), "kmean.html")
    report_clusters(np.linspace(0, N_IMAGES, num=N_IMAGES), dbsan_cluster(featuresPerImage), "dbsan.html")

    return featuresPerImage

t = main()

#from sklearn.feature_selection import f_classif


#l = np.loadtxt("labels.txt",skiprows=1,delimiter=",")
#X = t[l[:,1] > 0]
#y = l[l[:,1] > 0][:,1]
#f,prob = f_classif(X,y)


from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity


l = np.loadtxt("labels.txt",skiprows=1,delimiter=",")

#0,2,8,12,14,9,
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
   