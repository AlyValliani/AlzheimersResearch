import Image
import nibabel
import numpy as np
from os.path import isfile, join
import sys

def createImages(list_of_ims):

    for im in list_of_ims:
        nii = nibabel.load(im)
        data = nii.get_data()
        x = data.shape[1]
        data_slice = data[x/2,:,:]
        new_name = im.strip().split('/')[-1][:-3] + 'png'

        ## we have to normalize the slice to be in the range 0-255
        max_val = float(np.max(data_slice[:]))
        data_slice /= max_val
        data_slice *= 255
        data_slice = np.array(data_slice, dtype='uint8')
        
        image = Image.fromarray(data_slice, 'L')
        image.save(new_name)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python createImages.py nii_file_1 [nii_file_2] [nii_file_3] ...')
        exit
    
    createImages(sys.argv[1:])
