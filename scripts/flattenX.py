import numpy as np
import pickle
import sys


def flattenX(src_path, dst_path):

    with open(src_path, 'rb') as src:
        X = pickle.load(src)

    new_shape = (X.shape[0], X.shape[1], X.shape[2]*X.shape[3])
    new_X = np.zeros(new_shape)
    for dim_i in range(X.shape[3]):
        new_X[:, :, dim_i*X.shape[2]:(dim_i+1)*X.shape[2]] = X[:, :, :, dim_i]

    with open(dst_path, 'w') as dst:
        pickle.dump(new_X, dst)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python flattenX src_path dst_path')

    flattenX(sys.argv[1], sys.argv[2])
