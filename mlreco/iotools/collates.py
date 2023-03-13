"""
Collate classes are a middleware between parsers and datasets.
They are given to `torch.utils.data.DataLoader` as `collate_fn` argument.
We have two different collate functions: one for sparse and one for dense
input data.
"""
import numpy as np

from mlreco.utils.volumes import VolumeBoundaries


def CollateSparse(batch, boundaries=None):
    '''
    Collate sparse input.

    Parameters
    ----------
    batch : a list of dictionary
        Each list element (single dictionary) is a minibatch data = key-value pairs where a value is a parser function return.
    boundaries: list, optional, default is None
        This contains a list of volume boundaries if you want to process distinct volumes independently. See VolumeBoundaries
        documentation for more details and explanations.

    Returns
    -------
    dict
        a dictionary of key-value pair where key is same as keys in the input batch, and the value is a list of data elements in the input.

    Notes
    -----
    Assumptions:

    - The input batch is a tuple of length >=1. Length 0 tuple will fail (IndexError).
    - The dictionaries in the input batch tuple are assumed to have identical list of keys.
    '''
    import MinkowskiEngine as ME

    split_boundaries = boundaries is not None
    vb = VolumeBoundaries(boundaries) if split_boundaries else None

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
            dim = coords[0].shape[1]
            coords = concat(coords_minibatch, axis=0)
            if split_boundaries:
                coords[:, :dim+1], perm = vb.split(coords[:, :dim+1])
                coords = coords[perm]

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

                if split_boundaries:
                    voxels, perm = vb.split(voxels)
                    voxels = voxels[perm]
                    data = data[perm]

                result[key] = concat([voxels, data], axis=1)

            elif isinstance(batch[0][key],np.ndarray) and \
                 len(batch[0][key].shape) == 1:
                #
                result[key] = concat( [ concat( [np.full(shape=[len(sample[key]),1],
                                                 fill_value=batch_id,
                                                 dtype=np.float32),
                                                 np.expand_dims(sample[key],1)],
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

            elif isinstance(batch[0][key], list) and len(batch[0][key]) and isinstance(batch[0][key][0], tuple):
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
    """
    Collate dense input.

    Very basic collate function that makes a numpy.ndarray for each key.

    Parameters
    ----------
    batch : list
    """
    result  = {}
    for key in batch[0].keys():
        result[key] = np.array([sample[key] for sample in batch])
    return result
