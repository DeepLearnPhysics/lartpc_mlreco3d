import numpy as np

def unwrap_2d_scn(data_blob, outputs, main_key=None, data_keys=None, output_keys=None):
    """
    See unwrap_scn
    """
    return unwrap_scn(data_blob, outputs, 2, main_key, data_keys, output_keys)


def unwrap_3d_scn(data_blob, outputs, main_key=None, data_keys=None, output_keys=None):
    """
    See unwrap_scn
    """
    return unwrap_scn(data_blob, outputs, 3, main_key, data_keys, output_keys)


def unwrap_scn(data_blob, outputs, data_dim, main_key=None, data_keys=None, output_keys=None):
    """
    Break down the data_blob and outputs dictionary into events for sparseconvnet formatted tensors.
    Need to account for: multi-gpu, minibatching, multiple outputs, batches.
    INPUTS:
        data_blob: a dictionary of array of array of minibatch data [key][num_minibatch][num_device]
        outputs: results dictionary, output of trainval.forward, [key][num_minibatch*num_device]
        data_dim: 2 for 2D, 3 for 3D,,, and indicate the location of "batch id"
        main_key: used to identify a unique set of batch ids to be parsed
        data_keys: a list of string keys to specify, if needed, a subset of data to be returned
        output_keys: a list of string keys to specify, if needed, a subset of output to be returned
    OUTPUT:
        two un-wrapped arrays of dictionaries where array length = num_minibatch*num_device*minibatch_size
    ASSUMES:
        the shape of data_blob and outputs as explained above
    """

    # Handle data
    result_data = {}
    unwrap_map  = {} # dict of [#pts][batch_id] = where
    # a-0) Find the target keys
    target_array_keys = []
    target_list_keys  = []
    for key,data in data_blob.items():
        if not key in result_data: result_data[key]=[]
        if isinstance(data[0],np.ndarray) and len(data[0].shape) == 2: 
            target_array_keys.append(key)
        elif isinstance(data[0],list) and isinstance(data[0][0],np.ndarray) and len(data[0][0].shape) == 2: 
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
            # check if batch map is available, and create if not
            if not d.shape[0] in unwrap_map:
                batch_map = {}
                batch_idx = np.unique(d[:,data_dim])
                for b in batch_idx:
                    batch_map[b] = d[:,data_dim] == b
                unwrap_map[d.shape[0]]=batch_map

            batch_map = unwrap_map[d.shape[0]]
            for where in batch_map.values():
                result_data[target].append(d[where])
                
    # a-2) Handle the list of list of ndarrays
    #for target in target_list_keys:
    #    data = data_blob[target]
    #    num_elements = len(data[0])
    #    for list_idx in range(num_elements):
    #        combined_list = []
    #        for d in data:
    #            target_data = d[list_idx]
    #
    #            if not target_data.shape[0] in unwrap_map:
    #                batch_map = {}
    #                batch_idx = np.unique(target_data[:,data_dim])
    #                for b in batch_idx:
    #                    batch_map[b] = target_data[:,data_dim] == b
    #                unwrap_map[target_data.shape[0]]=batch_map
    #
    #            batch_map = unwrap_map[target_data.shape[0]]
    #            combined_list.extend([ target_data[where] for where in batch_map.values() ])
    #        result_data[target].append(combined_list)

    # a-2) Handle the list of list of ndarrays
    for target in target_list_keys:
        data = data_blob[target]
        for dlist in data:
            # construct a list of batch ids
            batch_ids = []
            for d in dlist:
                batch_ids.extend([n for n in np.unique(d[:,data_dim]) if not n in batch_ids])
            batch_ids.sort()
            for b in batch_ids:
                result_data[target].append([ d[d[:,data_dim] == b] for d in dlist ])
            
    # Handle output
    result_outputs = {}
    # b-0) Find the target keys
    target_array_keys = []
    target_list_keys  = []
    for key, data in outputs.items():
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
    for target in target_array_keys:
        data = outputs[target]
        for d in data:
            # check if batch map is available, and create if not
            if not d.shape[0] in unwrap_map:
                batch_map = {}
                batch_idx = np.unique(d[:,data_dim])
                # ensure these are integer values
                assert(len(batch_idx) == len(np.unique(batch_idx.astype(np.int32))))
                for b in batch_idx:
                    batch_map[b] = d[:,data_dim] == b
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
    for target in target_list_keys:
        print(target,len(outputs[target]))
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
                if not d.shape[0] in element_map:
                    batch_idx = np.unique(d[:,data_dim])
                    batch_ctrs.append(int(np.max(batch_idx)+1))
                    assert(len(batch_idx) == len(np.unique(batch_idx.astype(np.int32))))
                    where = [d[:,data_dim] == b for b in range(batch_ctrs[-1])]
                    element_map[d.shape[0]] = where
        assert len(np.unique(batch_ctrs)) == 1
        list_unwrap_map.append(element_map)
        list_batch_ctrs.append(batch_ctrs[0])
        
    for target in target_list_keys:
        data = outputs[target]
        for data_index, dlist in enumerate(data):
            batch_ctrs  = list_batch_ctrs[data_index]
            element_map = list_unwrap_map[data_index]
            for b in range(batch_ctrs):
                result_outputs[target].append([ d[element_map[d.shape[0]][b]] for d in dlist])

    return result_data, result_outputs


def unwrap_scn2(data_blob, outputs, data_dim, main_key=None, data_keys=None, output_keys=None):
    """
    Break down the data_blob and outputs dictionary into events for sparseconvnet formatted tensors.
    Need to account for: multi-gpu, minibatching, multiple outputs, batches.
    INPUTS:
        data_blob: a dictionary of array of array of minibatch data [key][num_minibatch][num_device]
        outputs: results dictionary, output of trainval.forward, [key][num_minibatch*num_device]
        data_dim: 2 for 2D, 3 for 3D,,, and indicate the location of "batch id"
        main_key: used to identify a unique set of batch ids to be parsed
        data_keys: a list of string keys to specify, if needed, a subset of data to be returned
        output_keys: a list of string keys to specify, if needed, a subset of output to be returned
    OUTPUT:
        two un-wrapped arrays of dictionaries where array length = num_minibatch*num_device*minibatch_size
    ASSUMES:
        the shape of data_blob and outputs as explained above
    """
    if data_keys   is None: data_keys   = list(data_blob.keys())
    if output_keys is None: output_keys = list(outputs.keys()  )
    if main_key    is None: main_key    = data_keys[0]
    
    parsed_data_blob = []
    parsed_outputs   = []

    #for key in data_blob.keys(): parsed_data_blob[key]=[]
    #for key in outputs.keys()  : parsed_outputs[key]=[]

    num_forward = len(data_blob[main_key])
    # Only unwrap those elements of data_blob that have the correct dimension: [key][num_forward][num_gpu]
    tmp=[]
    for key in data_keys:
        if not len(data_blob[key]) == num_forward: continue
        if not isinstance(data_blob[key],list): continue
        if not isinstance(data_blob[key][0],list): continue
        if not len(data_blob[key][0]) == len(data_blob[main_key][0]): continue
        tmp.append(key)
    data_keys = tmp

    # Only unwrap those elements of outputs that have the correct dimension: [key][num_forwards*num_gpu]
    tmp=[]
    num_total_element = 0
    for i in range(num_forward):
        num_total_element += len(data_blob[main_key][i])
    for key in output_keys:
        if len(outputs[key]) == num_total_element:
            tmp.append(key)
    output_keys = tmp
    output_index = 0
    for i in range(num_forward):
        num_gpus = len(data_blob[main_key][i])
        for j in range(num_gpus):
            batch_idx = np.unique(data_blob[main_key][i][j][:, data_dim])
            for b in batch_idx:
                data_index = data_blob[main_key][i][j][:, data_dim] == b
                data_blob_element = {}
                output_element    = {}
                for key in data_keys:
                    #print('Unwrapping input',key)
                    if isinstance(data_blob[key][i][j], np.ndarray) and len(data_blob[key][i][j].shape) == 2:
                        data_blob_element[key] = data_blob[key][i][j][data_blob[key][i][j][:, data_dim] == b]
                    elif isinstance(data_blob[key][i][j], list):
                        data_blob_element[key] = data_blob[key][i][j][int(b)]
                        #print('skipping',key)
                #print(outputs['segmentation'])
                #print('---')
                #print(outputs['segmentation'][output_index])
                for key in output_keys:
                    #print('Unwrapping output',key)
                    target = outputs[key][output_index]
                    if isinstance(target, np.ndarray):
                        if target.shape[0] == data_index.shape[0]:
                            output_element[key] = target[data_index]
                        else:
                            output_element[key] = target[target[:,data_dim] == b]
                    elif isinstance(target, list) and isinstance(target[0], np.ndarray) and len(target[0].shape) == 2:
                        output_element[key] = [ list_element[list_element[:,data_dim] == b] for list_element in target ]
                parsed_data_blob.append(data_blob_element)
                parsed_outputs.append(output_element)
            output_index += 1
                
    return parsed_data_blob, parsed_outputs

