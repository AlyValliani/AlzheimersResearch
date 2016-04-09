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
#import sPickle
from random import shuffle
from cutBrains import getLabel

def enlargeBrains(path_to_nii):
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

    max_brains = 40
    num_seen = [0, 0, 0]
    good_brains = [] ## allows us to shuffle the order easily
    for i, nii in enumerate(niis):
        label = getLabel(nii)
        if num_seen[label] >= max_brains:
            continue
        else:
            num_seen[label] += 1
        print("Processing brain %d of %d"%(sum(num_seen),3*max_brains))
        img = nibabel.load(join(path_to_nii, nii))
        data = np.array(img.get_data())
        if data.shape != dims:
            num_seen[label] -= 1
            continue
        good_brains.append((data, getLabel(nii)))

        if sum(num_seen) >= max_brains*3: break

    ## make equal ammounts from each class
    # num_labels = [0, 0, 0]
    # for (data, label) in good_brains:
    #     num_labels[label] += 1

    # max_brain_num = 20 #min(num_labels)

    # num_seen = [0,0,0]
    # tmp = []
    # for (data, label) in good_brains:
    #     if num_seen[label] < max_brain_num:
    #         num_seen[label] += 1
    #         tmp.append((data, label))
    #     else:
    #         del data

    # good_brains = tmp
    
    X_dims = (len(good_brains), dims[0], dims[1], dims[2])
    y_dims = (len(good_brains), 1)
    
    X = np.zeros(X_dims, dtype='float32')
    y = np.zeros(y_dims, dtype='uint8')

    ## shuffle all the examples so we can get a good number of each in the test set
    shuffle(good_brains)

    for i, (sl, label) in enumerate(good_brains):
        X[i, :, :, :] = sl
        y[i] = label

    # with open(join(path_to_nii, 'ADNI_X_full'), 'w+') as X_file:
    #     pickle.dump(X, X_file)

    # with open(join(path_to_nii, 'ADNI_y_full'), 'w+') as y_file:
    #     pickle.dump(y, y_file)

    with open(join(path_to_nii, 'ADNI_X_full'), 'w+') as X_file:
        pickle.dump(X, X_file)

    with open(join(path_to_nii, 'ADNI_y_full'), 'w+') as y_file:
        pickle.dump(y, y_file)

                
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python enlargeBrains.py filepath')

    enlargeBrains(sys.argv[1])
