import numpy as np
import torch

def list_concat(data_blob, outputs, avoid_keys=[]):
    result_data = {}
    for key,data in data_blob.items():
        if key in avoid_keys:
            result_data[key]=data
            continue
        if isinstance(data[0],list):
            result_data[key] = []
            for d in data: result_data[key] += d
        elif isinstance(data[0],np.ndarray):
            result_data[key] = np.concatenate(data)
        else:
            print('Unexpected data type',type(data))
            raise TypeError

    result_outputs = {}
    for key,data in outputs.items():
        if key in avoid_keys:
            result_outputs[key]=data
            continue
        if len(data) == 1:
            result_outputs[key]=data[0]
            continue
        # remove the outer-list
        if isinstance(data[0],list):
            result_outputs[key] = []
            for d in data:
                result_outputs[key] += d
        elif isinstance(data[0],np.ndarray):
            result_outputs[key] = np.concatenate(data)
        elif isinstance(data[0],torch.Tensor):
            result_outputs[key] = torch.concatenate(data,axis=0)
        else:
            print('Unexpected data type',type(data))
            raise TypeError

    return result_data, result_outputs


def unwrap(data_blob, outputs, batch_id_col=0, avoid_keys=[], input_key='input_data'):
    '''
    Break down the data_blob and outputs dictionary into events
    for sparseconvnet formatted tensors.

    Need to account for: multi-gpu, minibatching, multiple outputs, batches.
    INPUTS:
        data_blob: a dictionary of array of array of
            minibatch data [key][num_minibatch][num_device]
        outputs: results dictionary, output of trainval.forward,
            [key][num_minibatch*num_device]
        batch_id_col: 2 for 2D, 3 for 3D,,, and indicate
            the location of "batch id". For MinkowskiEngine, batch indices
            are always located at the 0th column of the N x C coordinate
            array
    OUTPUT:
        two un-wrapped arrays of dictionaries where
            array length = num_minibatch*num_device*minibatch_size
    ASSUMES:
        the shape of data_blob and outputs as explained above
    '''

    batch_idx_max = 0

    # Handle data
    result_data = {}
    unwrap_map  = {} # dict of [#pts][batch_id] = where
    # a-0) Find the target keys
    target_array_keys = []
    target_list_keys  = []
    for key,data in data_blob.items():
        if key in avoid_keys:
            result_data[key]=data
            continue
        if not key in result_data: result_data[key]=[]
        if isinstance(data[0],np.ndarray) and len(data[0].shape) == 2:
            target_array_keys.append(key)
        elif isinstance(data[0],torch.Tensor) and len(data[0].shape) == 2:
            target_array_keys.append(key)
        elif isinstance(data[0],list) and \
             isinstance(data[0][0],np.ndarray) and \
             len(data[0][0].shape) == 2:
            target_list_keys.append(key)
        elif isinstance(data[0],list):
            for d in data: result_data[key].extend(d)
        else:
            print('Un-interpretable input data...')
            print('key:',key)
            print('data:',data)
            raise TypeError
    # a-1) Handle the list of ndarrays

    for target in target_array_keys:
        data = data_blob[target]
        for d in data:
            # print(target, d, d.shape)
            # check if batch map is available, and create if not
            if not d.shape[0] in unwrap_map:
                batch_map = {}
                batch_id_loc = batch_id_col if d.shape[1] > batch_id_col else -1
                batch_idx = np.unique(d[:,batch_id_loc])
                if len(batch_idx):
                    batch_idx_max = max(batch_idx_max, int(batch_idx.max()))
                for b in batch_idx:
                    batch_map[b] = d[:,batch_id_loc] == b
                unwrap_map[d.shape[0]]=batch_map

            batch_map = unwrap_map[d.shape[0]]
            for where in batch_map.values():
                result_data[target].append(d[where])

    # a-2) Handle the list of list of ndarrays
    for target in target_list_keys:
        data = data_blob[target]
        for dlist in data:
            # construct a list of batch ids
            batch_ids = []
            batch_id_loc = batch_id_col if d.shape[1] > batch_id_col else -1
            for d in dlist:
                batch_ids.extend([n for n in np.unique(d[:,batch_id_loc]) if not n in batch_ids])
            batch_ids.sort()
            for b in batch_ids:
                result_data[target].append([ d[d[:,batch_id_loc] == b] for d in dlist ])

    # Handle output
    result_outputs = {}

    # Fix unwrap map
    output_unwrap_map  = {}
    for key in unwrap_map:
        if (np.array([d.shape[0] for d in data_blob[input_key]]) == key).any():
            output_unwrap_map[key] = unwrap_map[key]
    unwrap_map = output_unwrap_map

    # b-0) Find the target keys
    target_array_keys = []
    target_list_keys  = []
    # print(len(result_outputs['points']))
    for key, data in outputs.items():
        if key in avoid_keys:
            if not isinstance(data, list):
                result_outputs[key] = [data]    # Temporary Fix
            else:
                result_outputs[key] = data
            continue
        if not key in result_outputs: result_outputs[key]=[]
        if not isinstance(data,list): result_outputs[key].append(data)
        elif isinstance(data[0],np.ndarray) and len(data[0].shape)==2:
            target_array_keys.append(key)
        elif isinstance(data[0],list) and isinstance(data[0][0],np.ndarray) and len(data[0][0].shape)==2:
            target_list_keys.append(key)
        elif isinstance(data[0],list):
            for d in data: result_outputs[key].extend(d)
        else:
            result_outputs[key].extend(data)
            #print('Un-interpretable output data...')
            #print('key:',key)
            #print('data:',data)
            #raise TypeError

    # b-1) Handle the list of ndarrays
    if target_array_keys is not None:
        target_array_keys.sort(reverse=True)

    for target in target_array_keys:
        data = outputs[target]
        for d in data:
            # check if batch map is available, and create if not
            if not d.shape[0] in unwrap_map:
                batch_map = {}
                batch_id_loc = batch_id_col if d.shape[1] > batch_id_col else -1
                batch_idx = np.unique(d[:,batch_id_loc])
                # ensure these are integer values
                # if target == 'points':
                #     print(target)
                #     print(d)
                #     print("--------------Batch IDX----------------")
                #     print(batch_idx)
                #     assert False
                # print(target, len(batch_idx), len(np.unique(batch_idx.astype(np.int32))))
                assert(len(batch_idx) == len(np.unique(batch_idx.astype(np.int32))))
                if len(batch_idx):
                    batch_idx_max = max(batch_idx_max, int(batch_idx.max()))
                # We are going to assume **consecutive numbering of batch idx** starting from 0
                # b/c problems arise if one of the targets is missing an entry (eg all voxels predicted ghost,
                # which means a batch idx is missing from batch_idx for target = input_rescaled)
                # then alignment across targets is lost (output[target][entry] may not correspond to batch id `entry`)
                batch_idx = np.arange(0, int(batch_idx_max)+1, 1)
                # print(target, batch_idx, [np.count_nonzero(d[:,batch_id_loc] == b) for b in batch_idx])
                for b in batch_idx:
                    batch_map[b] = d[:,batch_id_loc] == b
                unwrap_map[d.shape[0]]=batch_map

            batch_map = unwrap_map[d.shape[0]]
            for where in batch_map.values():
                result_outputs[target].append(d[where])

    # b-2) Handle the list of list of ndarrays
    #for target in target_list_keys:
    #    data = outputs[target]
    #    num_elements = len(data[0])
    #    for list_idx in range(num_elements):
    #        combined_list = []
    #        for d in data:
    #            target_data = d[list_idx]
    #            if not target_data.shape[0] in unwrap_map:
    #                batch_map = {}
    #                batch_idx = np.unique(target_data[:,data_dim])
    #                for b in batch_idx:
    #                    batch_map[b] = target_data[:,data_dim] == b
    #                unwrap_map[target_data.shape[0]]=batch_map

    #            batch_map = unwrap_map[target_data.shape[0]]
    #            combined_list.extend([ target_data[where] for where in batch_map.values() ])
    #            #combined_list.extend([ target_data[target_data[:,data_dim] == b] for b in batch_idx])
    #        result_outputs[target].append(combined_list)

    # b-2) Handle the list of list of ndarrays

    # ensure outputs[key] length is same for all key in target_list_keys
    # for target in target_list_keys:
    #     print(target,len(outputs[target]))
    num_elements = np.unique([len(outputs[target]) for target in target_list_keys])
    assert len(num_elements)<1 or len(num_elements) == 1
    num_elements = 0 if len(num_elements) < 1 else int(num_elements[0])
    # construct unwrap mapping
    list_unwrap_map = []
    list_batch_ctrs = []
    for data_index in range(num_elements):
        element_map = {}
        batch_ctrs  = []
        for target in target_list_keys:
            dlist = outputs[target][data_index]
            for d in dlist:
                # print(d)
                if not d.shape[0] in element_map:
                    if len(d.shape) < 2:
                        print(target, d.shape)
                    batch_id_loc = batch_id_col if d.shape[1] > batch_id_col else -1
                    batch_idx = np.unique(d[:,batch_id_loc])
                    if len(batch_idx):
                        batch_idx_max = max(batch_idx_max, int(batch_idx.max()))
                    batch_ctrs.append(int(batch_idx_max+1))
                    try:
                        assert(len(batch_idx) == len(np.unique(batch_idx.astype(np.int32))))
                    except AssertionError:
                        raise AssertionError("Result key {} is not included in concat_result".format(target))
                    where = [d[:,batch_id_loc] == b for b in range(batch_ctrs[-1])]
                    element_map[d.shape[0]] = where
        # print(batch_ctrs)
        # if len(np.unique(batch_ctrs)) != 1:
        #     print(element_map)
        #     for i, d in enumerate(dlist):
        #         print(i, d, np.unique(d[:, batch_id_loc].astype(int)))
        # assert len(np.unique(batch_ctrs)) == 1
        list_unwrap_map.append(element_map)
        list_batch_ctrs.append(min(batch_ctrs))
    for target in target_list_keys:
        data = outputs[target]
        for data_index, dlist in enumerate(data):
            batch_ctrs  = list_batch_ctrs[data_index]
            element_map = list_unwrap_map[data_index]
            for b in range(batch_ctrs):
                result_outputs[target].append([ d[element_map[d.shape[0]][b]] for d in dlist])
    return result_data, result_outputs
