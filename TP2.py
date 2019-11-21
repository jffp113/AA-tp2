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
    featuresPerImage = extract_features_or_read_file(data)[:,[15,11]]
    check_cluster( kmean_cluster(featuresPerImage))
    check_cluster( dbsan_cluster(featuresPerImage))
    report_clusters(np.linspace(0, N_IMAGES, num=N_IMAGES), kmean_cluster(featuresPerImage), "kmean.html")
    report_clusters(np.linspace(0, N_IMAGES, num=N_IMAGES), dbsan_cluster(featuresPerImage), "dbsan.html")

    return featuresPerImage

t = main()


#from sklearn.feature_selection import f_classif


#l = np.loadtxt("labels.txt",skiprows=1,delimiter=",")
#X = t[l[:,1] > 0]
#y = l[l[:,1] > 0][:,1]
#f,prob = f_classif(X,y)