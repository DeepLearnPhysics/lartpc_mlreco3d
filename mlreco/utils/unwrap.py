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
                    if isinstance(data_blob[key][i][j], np.ndarray) and len(data_blob[key][i][j].shape) == 2:
                        data_blob_element[key] = data_blob[key][i][j][data_blob[key][i][j][:, data_dim] == b]
                    elif isinstance(data_blob[key][i][j], list):
                        data_blob_element[key] = data_blob[key][i][j][int(b)]
                #print(outputs['segmentation'])
                #print('---')
                #print(outputs['segmentation'][output_index])
                for key in output_keys:
                    target = outputs[key][output_index]
                    if isinstance(target, np.ndarray):
                        if target.shape[0] == data_index.shape[0]:
                            output_element[key] = target[data_index]
                        else:
                            output_element[key] = target[target[:,data_dim] == b]
                    elif isinstance(target, list):
                        output_element[key] = target[int(b)]

                parsed_data_blob.append(data_blob_element)
                parsed_outputs.append(output_element)
            output_index += 1
                
    return parsed_data_blob, parsed_outputs

