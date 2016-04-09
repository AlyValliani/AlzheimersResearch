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
from random import shuffle, randrange
from cutBrains import getLabel

def makePatches(path_to_nii, path_to_dest):
    '''
    Takes the path to all of the .nii files that we want to put in the
    final dataset, cuts them, and puts them into a pickle in the dest
    '''

    num_patches = 100000 ## 10,000 patches to start
    
    all_in_dir = listdir(path_to_nii)
    niis = [f for f in all_in_dir if isfile(join(path_to_nii, f)) and f[-4:] == '.nii']

    patches = []
    ## we have to import all of the niis first
    print('Importing niis')
    data_arr = [nibabel.load(join(path_to_nii, nii)).get_data() for nii in niis]
    
    print('Making patch')
    for i in range(num_patches):
        print('\t%d/%d'%(i+1, num_patches))
        data = data_arr[i%len(niis)]

        ## pick dimension to choose plane
        plane = randrange(3)

        patch = np.zeros((8,8))

        while not patch.all(): ## repeat if patch all zeros
            cut_loc = randrange(data.shape[plane])
            if plane == 0:
                cut = data[cut_loc, :, :]
            elif plane == 1:
                cut = data[:, cut_loc, :]
            else:
                cut = data[:, :, cut_loc]

            ## upper left (ul) corner of 8x8 patch
            ul_x, ul_y = (randrange(cut.shape[0] - 8),randrange(cut.shape[1] - 8))
            patch = cut[ul_x:ul_x+8, ul_y:ul_y+8]
                        
        patches.append(patch)
    
    dims = (len(patches), 8, 8)
    
    patch_array = np.zeros(dims, dtype='float32')

    ## shuffle all the examples so we can get a good number of each in the test set
    shuffle(patches)

    for i, patch in enumerate(patches):
        patch_array[i, :, :] = patch

    with open(join(path_to_dest, 'patches.pkl'), 'w') as f:
        pickle.dump(patch_array, f)
                
if __name__ == '__main__':
    if len(sys.argv) > 3:
        print('Usage: python makePatches.py [brain_filepath] [dest_filepath]')

    if len(sys.argv) < 2:
        brain_path = '/sonigroup/fmri/ADNI_registered_and_stripped/'
    else:
        brain_path = sys.argv[1]

    if len(sys.argv) < 3:
        dest_path = '/sonigroup/fmri/ADNI_patches/'
    else:
        dest_path = sys.argv[2]

    makePatches(brain_path, dest_path)
