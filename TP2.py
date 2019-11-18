# -*- coding: utf-8 -*-

from extraction import extract_feature
from tp2_aux import images_as_matrix
import cluster
import numpy as np

FEATURES = 18
SIZE = 564
N_IMAGES = 563
EXTRACT = True

def extract_features_or_read_file(data):
    featuresPerImage = []
    
    if EXTRACT:
        for i in range(0,N_IMAGES):
            ft = extract_feature(data[i].reshape(50,50))
            np.savez(f"ft/{i}",ft)
            featuresPerImage.append(ft)
    else:
        for i in range(0,N_IMAGES):
            file = np.load(f"ft/{i}.npz")
            featuresPerImage.append(file['arr_0'])
    
    return featuresPerImage

def main():
    data = images_as_matrix()
    featuresPerImage = extract_features_or_read_file(data)


    return featuresPerImage

test = main()
