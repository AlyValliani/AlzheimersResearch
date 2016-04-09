'''cutBrains

This script takes all of the MRIs in a given repository, takes the
middle most image of each, and places it into a single array pickle
which is placed into that same repository

Written by Aly Valliani Andrew Gilchrist-Scott

'''

import numpy as np
from os.path import isfile, join
from os import listdir
import nibabel
import re
import sys
import pickle
from random import shuffle
from cutBrains import getLabel
import cv2

def downsampleBrains(path_to_nii, factor):
    '''
    Takes the path to all of the .nii files that we want to put in the
    final dataset and the factor by which we want to downsample
    '''

    all_in_dir = listdir(path_to_nii)
    niis = [f for f in all_in_dir if isfile(join(path_to_nii, f)) and f[-4:] == '.nii']

    ## take the first nii as an example (assumes all same shape)
    img = nibabel.load(join(path_to_nii, niis[0]))
    data = img.get_data()
    dims = data.shape ## put all dims into same space
    down_dims = (int(dims[0] / factor), int(dims[1] / factor), int(dims[2] / factor))

    good_slices = [] ## allows us to shuffle the order easily
    for i, nii in enumerate(niis):
        print("Processing brain %d of %d"%(i+1,len(niis)))
        img = nibabel.load(join(path_to_nii, nii))
        data = np.array(img.get_data())
        down_data = downsample3D(data, down_dims, factor)
        good_slices.append((down_data, getLabel(nii)))
    
    X_dims = (len(good_slices), down_dims[0], down_dims[1], down_dims[2])
    y_dims = (len(good_slices), 1)
    
    X = np.zeros(X_dims, dtype='float32')
    y = np.zeros(y_dims, dtype='uint8')

    ## shuffle all the examples so we can get a good number of each in the test set
    shuffle(good_slices)

    for i, (sl, label) in enumerate(good_slices):
        X[i, :, :, :] = sl
        y[i] = label

    with open(join(path_to_nii, 'ADNI_X_down'), 'w+') as X_file:
        pickle.dump(X, X_file)

    with open(join(path_to_nii, 'ADNI_y_down'), 'w+') as y_file:
        pickle.dump(y, y_file)

def downsample3D(data, new_dims, factor):
    xy_down = cv2.resize(data, (new_dims[0], new_dims[1])) #(0,0), fx=1.0/factor, fy=1.0/factor)
    final = np.zeros(new_dims, dtype='float32')
    for x in range(xy_down.shape[0]):
        sdf = cv2.resize(xy_down[x,:,:], (new_dims[2], new_dims[1]))#(0,0), fx=.99/factor, fy=1.0)
        final[x,:,:] = sdf
    return final
                
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python cutBrains.py filepath down_factor')

    downsampleBrains(sys.argv[1], float(sys.argv[2]))
