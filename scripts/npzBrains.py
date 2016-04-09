'''npzBrains

This script takes all of the MRIs in a given repository, takes the
file and saves it in a compressed npz format

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

    brains = []
    names = []
    
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
            ## chose to slice at the middle of the x dimension
            cut_loc = data.shape[0] / 2
            ## cut the memmap and turn it into an array
            cut = np.array(data[cut_loc, :, :])
            cut /= np.max(cut) ## normalize
            brains.append(data)
            names.append(getLabel(nii))

    print('Shuffling brains')
    names_and_brains = zip(names, brains)
    shuffle(names_and_brains)
    names, brains = zip(*names_and_brains) # unzip

    filename = 'ADNI_brains.npz'
    print("Saving in %s"%(filename))
    np.savez_compressed(filename, names, brains)
    #np.savez(filename, names, brains)
                
def getLabel(nii_name):
    '''
    Converts large name into brain state label CN vs MCI vs AD
    Also labels as unknown if no patter can be found
    '''

    if re.search('CN', nii_name):
        return 'CN'
    elif re.search('MCI', nii_name):
        return 'MCI'
    elif re.search('AD', nii_name):
        return 'AD'
    else:
        return '<Unknown>'


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python cutBrains.py filepath [filepath2] ...')

    cutBrains(sys.argv[1:], 'ssr')
