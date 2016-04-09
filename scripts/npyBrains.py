'''npzBrains

This script takes all of the MRIs in a given repository, takes the
file and saves it in a .npy file in that same repository

Written by Aly Valliani Andrew Gilchrist-Scott

'''

import numpy as np
from os.path import isfile, join
from os import listdir
import nibabel
import re
import sys
from random import shuffle

def cutBrains(paths_to_nii, file_selector=None):
    '''
    Takes the path to all of the .nii files that we want to put in the
    final dataset
    '''
    
    for path_to_nii in paths_to_nii:
        print("For directory %s ..."%(path_to_nii))
        all_in_dir = listdir(path_to_nii)
        niis = [f for f in all_in_dir if isfile(join(path_to_nii, f)) and f[-4:] == '.nii']

        if file_selector:
            niis = [nii for nii in niis if re.search(file_selector, nii)]

        for i, nii in enumerate(niis):
            print("Processing brain %d of %d"%(i+1,len(niis)))
            img = nibabel.load(join(path_to_nii, nii))
            data = img.get_data()
            ## add in the below line for normalization
            # data /= np.max(data)
            new_filename = nii[:-4] + '.npy'
            np.save(join(path_to_nii, new_filename), data)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python cutBrains.py filepath [filepath2] ...')

    cutBrains(sys.argv[1:], 'ssr')
