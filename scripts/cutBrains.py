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

def cutBrains(paths_to_nii, file_selector=None):
    '''
    Takes the path to all of the .nii files that we want to put in the
    final dataset
    '''
    
    good_slices = []
    im_dims = None
    
    for path_to_nii in paths_to_nii:
        print("For directory %s ..."%(path_to_nii))
        all_in_dir = listdir(path_to_nii)
        niis = [f for f in all_in_dir if isfile(join(path_to_nii, f)) and f[-4:] == '.nii']

        if file_selector:
            niis = [nii for nii in niis if re.search(file_selector, nii)]

        if im_dims == None:
            ## take the first nii as an example (assumes all same shape)
            img = nibabel.load(join(path_to_nii, niis[0]))
            data = img.get_data()
            im_dims = data.shape[1:] ## take last two dims as images we'll want

        for i, nii in enumerate(niis):
            print("Processing brain %d of %d"%(i+1,len(niis)))
            img = nibabel.load(join(path_to_nii, nii))
            data = img.get_data()
            if data.shape[1:] != im_dims:
                print("Skipping this brain due to inconsistent size")
                continue
            ## chose to slice at the middle of the x dimension
            cut_loc = data.shape[0] / 2
            cut = data[cut_loc, :, :]

        
            ## convert memmap to array and drop it into the list with label
            ## and normalize
            cut = np.array(cut)
            cut /= np.max(cut)
            good_slices.append((cut, getLabel(path_to_nii)))

    
    X_dims = (len(good_slices), im_dims[0], im_dims[1])
    y_dims = (len(good_slices), 1)
    
    X = np.zeros(X_dims, dtype='float32')
    y = np.zeros(y_dims, dtype='uint8')

    ## shuffle all the examples so we can get a good number of each in the test set
    shuffle(good_slices)

    for i, (sl, label) in enumerate(good_slices):
        X[i, :, :] = sl
        y[i] = label

    with open(join(path_to_nii, 'ADNI_X_normcuts_big'), 'w+') as X_file:
        pickle.dump(X, X_file)

    with open(join(path_to_nii, 'ADNI_y_normcuts_big'), 'w+') as y_file:
        pickle.dump(y, y_file)
                
def getLabel(nii_name):
    '''
    converts name of nii to int for classification
     0: Cognitively normal (CN)
     1: Moderate cognitive impairment (MCI)
     2: Alzheimer's Disease (AD)

     -1: Some other unknown state
    '''

    if re.search('CN', nii_name):
        return 0
    elif re.search('MCI', nii_name):
        return 1
    elif re.search('AD', nii_name):
        return 2
    else:
        return -1


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python cutBrains.py filepath [filepath2] ...')

    cutBrains(sys.argv[1:], 'ssr')
