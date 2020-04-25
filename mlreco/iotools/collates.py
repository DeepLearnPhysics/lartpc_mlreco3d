from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

def CollateSparse(batch):
    """
    INPUTS:
      batch - a tuple of dictionary. Each tuple element (single dictionary) is a minibatch data = key-value pairs where a value is a parser function return.
    OUTPUT:
      return - a dictionary of key-value pair where key is same as keys in the input batch, and the value is a list of data elements in the input.
    ASSUMES:
      - The input batch is a tuple of length >=1. Length 0 tuple will fail (IndexError).
      - The dictionaries in the input batch tuple are assumed to have identical list of keys.
  WARNINGS:
    TBD
  EXAMPLES:
    TBD
    """
    concat = np.concatenate
    result = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], tuple) and isinstance(batch[0][key][0], np.ndarray) and len(batch[0][key][0].shape)==2:
            # handle SCN input batch
            voxels = concat( [ concat( [sample[key][0],
                                        np.full(shape=[len(sample[key][0]),1], fill_value=batch_id, dtype=np.int32)],
                                       axis=1 ) for batch_id, sample in enumerate(batch) ],
                             axis = 0)
            data = concat([sample[key][1] for sample in batch], axis=0)
            result[key] = concat([voxels, data], axis=1)
        elif isinstance(batch[0][key],np.ndarray) and len(batch[0][key].shape)==1:
            result[key] = concat( [ concat( [np.expand_dims(sample[key],1),
                                             np.full(shape=[len(sample[key]),1],fill_value=batch_id,dtype=np.float32)],
                                            axis=1 ) for batch_id,sample in enumerate(batch) ],
                                  axis=0)
        elif isinstance(batch[0][key],np.ndarray) and len(batch[0][key].shape)==2:
            result[key] =  concat( [ concat( [sample[key],
                                              np.full(shape=[len(sample[key]),1],fill_value=batch_id,dtype=np.float32)],
                                             axis=1 ) for batch_id,sample in enumerate(batch) ],
                                   axis=0)
        elif isinstance(batch[0][key], list) and isinstance(batch[0][key][0], tuple):
            result[key] = [
                concat([
                    concat( [ concat( [sample[key][depth][0],
                                                np.full(shape=[len(sample[key][depth][0]),1], fill_value=batch_id, dtype=np.int32)],
                                               axis=1 ) for batch_id, sample in enumerate(batch) ],
                                     axis = 0),
                    concat([sample[key][depth][1] for sample in batch], axis=0)
                ], axis=1) for depth in range(len(batch[0][key]))
            ]
        else:
            result[key] = [sample[key] for sample in batch]
            
    return result

#def CollateSparse(batch):
#    concat = np.concatenate
#    result = []
#    for i in range(len(batch[0])):
#        if isinstance(batch[0][i], tuple) and isinstance(batch[0][i][0], np.ndarray) and len(batch[0][i][0].shape)==2:
#            # handle SCN input batch
#            voxels = concat( [ concat( [sample[i][0],
#                                        np.full(shape=[len(sample[i][0]),1], fill_value=batch_id, dtype=np.int32)],
#                                      axis=1 ) for batch_id, sample in enumerate(batch) ],
#                            axis = 0)
#            data = concat([sample[i][1] for sample in batch], axis=0)
#            result.append( concat([voxels, data], axis=1) )
#
#        elif isinstance(batch[0][i],np.ndarray) and len(batch[0][i].shape)==1:
#            result.append( concat( [ concat( [np.expand_dims(sample[i],1),
#                                              np.full(shape=[len(sample[i]),1],fill_value=batch_id,dtype=np.float32)],
#                                             axis=1 ) for batch_id,sample in enumerate(batch) ],
#                                   axis=0)
#                           )
#        elif isinstance(batch[0][i],np.ndarray) and len(batch[0][i].shape)==2:
#            result.append( concat( [ concat( [sample[i],
#                                              np.full(shape=[len(sample[i]),1],fill_value=batch_id,dtype=np.float32)],
#                                             axis=1 ) for batch_id,sample in enumerate(batch) ],
#                                   axis=0)
#                           )
#        else:
#            result.append([sample[i] for sample in batch])
#    return resut#



def CollateDense(batch):
    result  = {}
    for key in batch[0].keys():
        result[key] = np.array([sample[key] for sample in batch])
    return result

#def CollateDense(batch):
#    result  = []
#    for i in range(len(batch[0])):
#        result.append(np.array([sample[i] for sample in batch]))
#    return result
