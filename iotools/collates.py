from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

def CollateSparse(batch):
    concat = np.concatenate
    result  = []
    for i in range(len(batch[0])):
        if isinstance(batch[0][i],np.ndarray) and len(batch[0][i].shape)==1:
            result.append( concat( [ concat( [np.expand_dims(sample[i],1),
                                              np.full(shape=[len(sample[i]),1],fill_value=batch_id,dtype=np.float32)],
                                             axis=1 ) for batch_id,sample in enumerate(batch) ],
                                   axis=0)
                           )
        elif isinstance(batch[0][i],np.ndarray) and len(batch[0][i].shape)==2:
            result.append( concat( [ concat( [sample[i],
                                              np.full(shape=[len(sample[i]),1],fill_value=batch_id,dtype=np.float32)],
                                             axis=1 ) for batch_id,sample in enumerate(batch) ],
                                   axis=0)
                           )
        else:
            result.append([sample[i] for sample in batch])
    return result

def CollateDense(batch):
    result  = []
    for i in range(len(batch[0])):
        result.append(np.array([sample[i] for sample in batch]))
    return result

