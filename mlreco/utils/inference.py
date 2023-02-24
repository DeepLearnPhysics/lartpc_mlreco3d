import yaml

def get_inference_cfg(cfg_path, dataset_path=None, weights_path=None, batch_size=None, cpu=False):
    '''
    Turns a training configuration into an inference configuration:
    - Turn `train` to `False`
    - Set sequential sampling
    - Load the specified validation dataset_path, if requested
    - Load the specified set of weights_path, if requested
    - Reset the batch_size to a different value, if requested
    - Make the model run in CPU mode, if requested

    Parameters
    ----------
    cfg_path : str
        Path to the configuration file
    dataset_path : str
        Path to the dataset to use for inference
    weights_path : str
        Path to the weigths to use for inference
    batch_size: int
        Number of data samples per batch
    cpu: bool
        Whether or not to execute the inference on CPU

    Returns
    ------
    dict
        Dictionary of parameters to initialize handlers
    '''
    # Get the config file from the train file
    cfg = open(cfg_path)

    # Convert the string to a dictionary
    cfg = yaml.load(cfg, Loader=yaml.Loader)

    # Turn train to False
    cfg['trainval']['train'] = False

    # Delete the random sampler
    if 'sampler' in cfg['iotool']:
        del cfg['iotool']['sampler']

    # Load weights_path, if requested
    if weights_path is not None:
        cfg['trainval']['model_path'] = weights_path

    # Change the batch_size, if requested
    cfg['iotool']['batch_size'] = batch_size

    # Put the network in CPU mode, if requested
    if cpu:
        cfg['trainval']['gpus'] = ''
    
    return cfg
