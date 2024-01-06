import yaml

def get_inference_cfg(cfg_path, dataset_path=None, weights_path=None, batch_size=None, num_workers=None, cpu=False):
    '''
    Turns a training configuration into an inference configuration:
    - Turn `train` to `False`
    - Set sequential sampling
    - Load the specified validation dataset_path, if requested
    - Load the specified set of weights_path, if requested
    - Reset the batch_size to a different value, if requested
    - Sets num_workers to a different value, if requested
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
    num_workers:
        Number of workers that load data
    cpu: bool
        Whether or not to execute the inference on CPU

    Returns
    -------
    dict
        Dictionary of parameters to initialize handlers
    '''
    # Get the config file from the train file
    cfg = open(cfg_path)

    # Convert the string to a dictionary
    cfg = yaml.load(cfg, Loader=yaml.Loader)

    # Turn train to False
    cfg['trainval']['train'] = False

    # Turn on unwrapper
    cfg['trainval']['unwrap'] = True

    # Delete the random sampler
    if 'sampler' in cfg['iotool']:
        del cfg['iotool']['sampler']

    # Change dataset, if requested
    if dataset_path is not None:
        cfg['iotool']['dataset']['data_keys'] = [dataset_path]

    # Change weights, if requested
    if weights_path is not None:
        cfg['trainval']['model_path'] = weights_path

    # Change the batch_size, if requested
    cfg['iotool']['batch_size'] = batch_size

    # Set the number of workers, if requested
    if num_workers is not None:
        cfg['iotool']['num_workers'] = num_workers

    # Put the network in CPU mode, if requested
    if cpu:
        cfg['trainval']['gpus'] = ''
    
    return cfg
