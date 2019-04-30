from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

def parse_sparse3d(data):
    """
    A function to retrieve sparse tensor from larcv::EventSparseTensor3D object
    Args:
        length 1 array of larcv::EventSparseTensor3D
    Return:
        a numpy array with the shape (N,4) where 4=3+1 represents (x,y,z) coordinate and stored pixel value.
    """
    from larcv import larcv
    event_tensor3d = data[0]
    num_point = event_tensor3d.as_vector().size()
    np_data   = np.zeros(shape=(num_point,4),dtype=np.float32)
    larcv.fill_3d_pcloud(event_tensor3d, np_data)
    return np_data

def parse_tensor3d(data):
    """
    A function to retrieve larcv::EventSparseTensor3D as a numpy array
    Args:
        length 1 array of larcv::EventSparseTensor3D
    Return:
        a numpy array of a dense 3d tensor object
    """
    return np.array(larcv.as_ndarray(event_tensor3d))

