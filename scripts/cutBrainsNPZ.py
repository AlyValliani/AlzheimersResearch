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
import sys
import re

def cutBrains(paths_to_nii, file_selector=None):
    '''
    Takes the path to all of the .nii files that we want to put in the
    final dataset
    '''
    
    good_slices = []
    im_dims = None
    names = []
    
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
            #good_slices.append((cut, getLabel(path_to_nii)))
            new_filename = nii[:-4] + '_cut'
            good_slices.append(cut)
            names.append(new_filename)

    final_filename = 'all_brains_cut'
    np.savez_compressed(final_filename, names, good_slices)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python cutBrains.py filepath [filepath2] ...')

    cutBrains(sys.argv[1:], 'ssr')

    
