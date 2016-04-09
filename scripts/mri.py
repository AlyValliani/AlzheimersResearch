'''
mri.py 

This script loads pickled MRI's into a design matrix to be utilized
with various models.

Aly Valliani and Andrew Gilchrist-Scott
'''

import numpy as N
import struct
import pickle
from theano.compat.six.moves import xrange
from pylearn2.datasets import cache, control, dense_design_matrix
from pylearn2.utils import serial
from pylearn2.utils.rng import make_np_rng
from pylearn2.utils.mnist_ubyte import open_if_filename

class MRI(dense_design_matrix.DenseDesignMatrix):

    def __init__(self, which_set, start=None, stop=None, axes=['b', 0, 1, 'c']):
        self.args = locals()

        def dimshuffle(b01c):
            default = ('b', 0, 1, 'c')
            return b01c.transpose(*[default.index(axis) for axis in axes])

        if control.get_load_data():
            path = "/local/Pylearn2/data/ADNI/"

            im_path = path + 'ADNI_X_cuts'
            label_path = path + 'ADNI_y'
            #im_path = serial.preprocess(im_path)
            #label_path = serial.preprocess(label_path)

            #Locally cache the files before reading them
            datasetCache = cache.datasetCache
            im_path = datasetCache.cache_file(im_path)
            label_path = datasetCache.cache_file(label_path)

            with open_if_filename(im_path, 'rb') as f:
                #magic, number, rows, cols = struct.unpack('>iiii', f.read(16))
                #im_array = N.fromfile(f, dtype='float64')
                im_array = pickle.load(f)

            with open_if_filename(label_path, 'rb') as f:
                #label_array = N.fromfile(f, dtype='uint8')
                label_array = pickle.load(f)

            topo_view = im_array
            y = label_array
        else:

            if which_set == 'train':
                size = 396
            else:
                size = 50
            topo_view = np.random.rand(size, 256, 166)
            y = np.random.randint(0, 3, (size, 1))

        y_labels = 3

        m, r, c = topo_view.shape 
        assert r == 256
        assert c == 166
        topo_view = topo_view.reshape(m, r, c, 1)

        if which_set == 'train':
            assert m == 396
        else:
            assert m == 50

        super(MRI, self).__init__(topo_view=dimshuffle(topo_view), y=y,
                axes=axes, y_labels=y_labels)

        assert not N.any(N.isnan(self.X))

        if start is not None:
            assert start >= 0
            #if stop > self.X.shape[0]:
                #raise ValueError('stop=' + str(stop) + '>' + 'm=' + str(self.X.shape=[0]))

            assert stop > start
            self.X = self.X[start:stop, :]
            if self.X.shape[0] != stop - start:
                raise ValueError("X.shape[0]: %d. start: %d stop: %d"
                            % (self.X.shape[0], start, stop))
            if len(self.y.shape) > 1:
                self.y = self.y[start:stop, :]
            else:
                self.y = self.y[start:stop]
            assert self.y.shape[0] == stop - start
