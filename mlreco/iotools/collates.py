import numpy as np
import torch
from pprint import pprint

# def CollateSparse(batch):
#     """
#     INPUTS:
#       batch - a tuple of dictionary. Each tuple element (single dictionary) is a minibatch data = key-value pairs where a value is a parser function return.
#     OUTPUT:
#       return - a dictionary of key-value pair where key is same as keys in the input batch, and the value is a list of data elements in the input.
#     ASSUMES:
#       - The input batch is a tuple of length >=1. Length 0 tuple will fail (IndexError).
#       - The dictionaries in the input batch tuple are assumed to have identical list of keys.
#   WARNINGS:
#     TBD
#   EXAMPLES:
#     TBD
#     """
#     concat = np.concatenate
#     result = {}
#     for key in batch[0].keys():
#         if isinstance(batch[0][key], tuple) and isinstance(batch[0][key][0], np.ndarray) and len(batch[0][key][0].shape)==2:
#             # handle SCN input batch
#             voxels = concat( [ concat( [sample[key][0],
#                                         np.full(shape=[len(sample[key][0]),1], fill_value=batch_id, dtype=np.int32)],
#                                        axis=1 ) for batch_id, sample in enumerate(batch) ],
#                              axis = 0)
#             data = concat([sample[key][1] for sample in batch], axis=0)
#             result[key] = concat([voxels, data], axis=1)
#         elif isinstance(batch[0][key],np.ndarray) and len(batch[0][key].shape)==1:
#             result[key] = concat( [ concat( [np.expand_dims(sample[key],1),
#                                              np.full(shape=[len(sample[key]),1],fill_value=batch_id,dtype=np.float32)],
#                                             axis=1 ) for batch_id,sample in enumerate(batch) ],
#                                   axis=0)
#         elif isinstance(batch[0][key],np.ndarray) and len(batch[0][key].shape)==2:
#             result[key] =  concat( [ concat( [sample[key],
#                                               np.full(shape=[len(sample[key]),1],fill_value=batch_id,dtype=np.float32)],
#                                              axis=1 ) for batch_id,sample in enumerate(batch) ],
#                                    axis=0)
#         elif isinstance(batch[0][key], list) and isinstance(batch[0][key][0], tuple):
#             result[key] = [
#                 concat([
#                     concat( [ concat( [sample[key][depth][0],
#                                                 np.full(shape=[len(sample[key][depth][0]),1], fill_value=batch_id, dtype=np.int32)],
#                                                axis=1 ) for batch_id, sample in enumerate(batch) ],
#                                      axis = 0),
#                     concat([sample[key][depth][1] for sample in batch], axis=0)
#                 ], axis=1) for depth in range(len(batch[0][key]))
#             ]
#         else:
#             result[key] = [sample[key] for sample in batch]
#
#     return result


def CollateSparse(batch):
    '''
    INPUTS:
        - batch: list of dict
    '''
    import MinkowskiEngine as ME
    result = {}
    concat = np.concatenate
    for key in batch[0].keys():
        if key == 'particles_label':

            coords = [sample[key][0] for sample in batch]
            features = [sample[key][1] for sample in batch]

            batch_index = np.full(shape=(coords[0].shape[0], 1),
                                  fill_value=0,
                                  dtype=np.float32)

            coords_minibatch = []
            #feats_minibatch = []

            for bidx, sample in enumerate(batch):
                batch_index = np.full(shape=(coords[bidx].shape[0], 1),
                                      fill_value=bidx, dtype=np.float32)
                batched_coords = concat([batch_index,
                                         coords[bidx],
                                         features[bidx]], axis=1)

                coords_minibatch.append(batched_coords)

            #coords = torch.Tensor(concat(coords_minibatch, axis=0))
            coords = concat(coords_minibatch, axis=0)

            result[key] = coords
        else:
            if isinstance(batch[0][key], tuple) and \
               isinstance(batch[0][key][0], np.ndarray) and \
               len(batch[0][key][0].shape) == 2:
                # For pairs (coordinate tensor, feature tensor)

                # Previously using ME.utils.sparse_collate which is the "official" way,
                # and an argument can be made that
                # > when something gets updated with regards to coordinate batching
                # > (in MinkowskiEngine), any necessary changes will also be made
                # > to ME.utils.sparse_collate
                #
                # However that forces us to return a torch.Tensor (or convert that back
                # to a numpy array) + such changes to coordinate batching would
                # have a wider impact on our code anyway.
                # Returning a torch.Tensor is inconsistent (other options return np.array)
                # + forces us to convert input data to .numpy() in visualization,
                # event if we do not run any network.
                # Hence keeping the homemade collate for now.

                # coords = [sample[key][0] for sample in batch]
                # features = [sample[key][1] for sample in batch]
                # print(coords, features)
                # coords, features = ME.utils.sparse_collate(coords, features)
                # print('after', coords, features)
                # result[key] = torch.cat([coords.float(),
                #                          features.float()], dim=1)
                voxels = concat( [ concat( [np.full(shape=[len(sample[key][0]),1], fill_value=batch_id, dtype=np.int32),
                                            sample[key][0]],
                                           axis=1 ) for batch_id, sample in enumerate(batch) ],
                                 axis = 0)
                data = concat([sample[key][1] for sample in batch], axis=0)
                result[key] = concat([voxels, data], axis=1)
            elif isinstance(batch[0][key],np.ndarray) and \
                 len(batch[0][key].shape) == 1:
                 #
                result[key] = concat( [ concat( [np.expand_dims(sample[key],1),
                                                 np.full(shape=[len(sample[key]),1],
                                                 fill_value=batch_id,
                                                 dtype=np.float32)],
                                                 axis=1 ) \
                    for batch_id,sample in enumerate(batch) ], axis=0)

            elif isinstance(batch[0][key],np.ndarray) and len(batch[0][key].shape)==2:
                # for tensors that does not come with a coordinate tensor
                # ex. particle_graph

                result[key] =  concat( [ concat( [np.full(shape=[len(sample[key]),1],
                                                          fill_value=batch_id,
                                                          dtype=np.float32),
                                                  sample[key]],
                                                axis=1 ) for batch_id,sample in enumerate(batch) ],
                                    axis=0)

            elif isinstance(batch[0][key], list) and isinstance(batch[0][key][0], tuple):
                # For multi-scale labels (probably deprecated)
                result[key] = [
                    concat([
                        concat( [ concat( [np.full(shape=[len(sample[key][depth][0]),1],
                                                   fill_value=batch_id,
                                                   dtype=np.int32),
                                           sample[key][depth][0]], axis=1 ) for batch_id, sample in enumerate(batch) ],
                                        axis = 0),
                        concat([sample[key][depth][1] for sample in batch], axis=0)
                    ], axis=1) for depth in range(len(batch[0][key]))
                ]
            else:

                result[key] = [sample[key] for sample in batch]
    return result



def CollateDense(batch):
    result  = {}
    for key in batch[0].keys():
        result[key] = np.array([sample[key] for sample in batch])
    return result
