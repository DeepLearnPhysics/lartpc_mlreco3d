#!/usr/bin/env python
import os
import sys
import yaml
from os import environ
import argparse

current_directory = os.path.dirname(os.path.abspath(__file__))
current_directory = os.path.dirname(current_directory)
sys.path.insert(0, current_directory)
from mlreco.main_funcs import process_config, train_loop, inference_loop, prepare


def load(filename, limit=None):
    import glob
    logs = []
    files = sorted(glob.glob(filename))
    print(filename)
    for f in files:
        #print(f)
        try:
            x = np.genfromtxt(f, delimiter=',', names=True)
        except e:
            print(e)
            continue
        logs.append(x)
    if limit is not None:
        print(len(logs), limit)
        logs = logs[:limit]
    #print(len(logs), [len(x) for x in logs], len(x[0]), len(x[-1]))
    #print(logs[0].dtype, logs[-1].dtype)
    logs = np.concatenate(logs, axis=0)
    print("Loaded %d events for %s" % (logs.shape[0], filename))
    return logs


def main(config, chain_config):
    cfg_file = config
    if not os.path.isfile(cfg_file):
        cfg_file = os.path.join(current_directory, 'config', config)
    if not os.path.isfile(cfg_file):
        print(config, 'not found...')
        sys.exit(1)

    chain_config_file = chain_config
    if not os.path.isfile(chain_config_file):
        chain_config_file = os.path.join(current_directory, 'config', chain_config)
    if not os.path.isfile(chain_config_file):
        print(chain_config, 'not found...')
        sys.exit(1)

    cfg = yaml.load(open(cfg_file, 'r'), Loader=yaml.Loader)
    chain_cfg = yaml.load(open(chain_config_file, 'r'), Loader=yaml.Loader)

    if environ.get('CUDA_VISIBLE_DEVICES') is not None and cfg['trainval']['gpus'] == '-1':
        cfg['trainval']['gpus'] = os.getenv('CUDA_VISIBLE_DEVICES')

    # Create validation cfg
    import copy
    validation_cfg = copy.deepcopy(cfg)
    validation_cfg['iotool']['dataset']['data_keys'] = chain_cfg['validation_data_keys']
    validation_cfg['iotool']['batch_size'] = chain_cfg['validation_batch_size']
    validation_cfg['iotool']['sampler'] = {
        'name': 'SequentialBatchSampler',
        'batch_size': chain_cfg['validation_batch_size']
    }
    validation_cfg['trainval']['iterations'] = chain_cfg['validation_num_iterations']
    # set log and weight dir
    validation_cfg['trainval']['log_dir'] = chain_cfg['validation_log_dir']
    validation_cfg['trainval']['weight_prefix'] = os.path.join(chain_cfg['validation_log_dir'], 'snapshot')
    validation_cfg['trainval']['train'] = False

    # Add postprocessing
    keys_list = []
    for threshold_list in chain_cfg['thresholds'].values():
        keys_list.extend(threshold_list.keys())
    keys_list = set(keys_list) # remove duplicates

    validation_cfg['post_processing'] = {
        'store_output': { 'keys_list': list(keys_list) }
    }

    process_config(cfg)
    print(validation_cfg['iotool']['dataset']['data_keys'])
    print(cfg['iotool']['dataset']['data_keys'])
    process_config(validation_cfg)

    if not cfg['trainval']['train']:
        print("This is meant to be a training script. Abort.")
        sys.exit(1)

    # Enable all stages to allow them to be constructed in __init__
    previous_chain = {}
    for key in cfg['model']['modules']['chain']:
        if key == 'enable_ghost' or key == 'enable_gnn_particle':
            continue
        if 'enable' in key:
            previous_chain[key] = cfg['model']['modules']['chain'][key]
            cfg['model']['modules']['chain'][key] = True
    # print(cfg['model']['modules']['chain'])

    event_list = None
    handlers = prepare(cfg, event_list=event_list)
    validation_handlers = prepare(validation_cfg, event_list=event_list)

    # Now revert enable_* config of network
    for key, value in previous_chain.items():
        setattr(handlers.trainer._net, key, value)

    # Set max iterations
    original_iterations = handlers.cfg['trainval']['iterations']
    handlers.cfg['trainval']['iterations'] = handlers.cfg['trainval']['checkpoint_step']

    while handlers.iteration < original_iterations:
        train_loop(handlers)
        handlers.cfg['trainval']['iterations'] += handlers.cfg['trainval']['checkpoint_step']

        # Run validation
        validation_handlers.cfg['trainval']['model_path'] = '%s-%d.ckpt' % (handlers.cfg['trainval']['weight_prefix'], handlers.iteration-1)
        if not os.path.isfile(validation_handlers.cfg['trainval']['model_path']):
            print('No weight found at ', validation_handlers.cfg['trainval']['model_path'])
            sys.exit(1)
        inference_loop(validation_handlers)

        # Read saved output and compare with thresholds
        logs = load(os.path.join(chain_cfg['validation_log_dir'], 'save-output-*.csv'))
        for enable, params in chain_cfg['enable_thresholds'].items():
            if getattr(handlers.trainer._net, enable, None) is None:
                print(enable + ' is not a correct name. Please double check for a typo.')
                sys.exit(1)
            are_thresholds_met = True
            # Loop over all threshold groups necessary to enable this step
            for threshold in params['thresholds']:
                if threshold not in chain_cfg['thresholds']:
                    print(threshold, ' is not defined in config.')
                    sys.exit(1)
                is_threshold_met = True
                for criteria, acc in chain_cfg['thresholds'][threshold].items():
                    if not (criteria in logs and logs[criteria] >= acc):
                        is_threshold_met = False
                are_thresholds_met = are_thresholds_met and is_threshold_met

            if are_thresholds_met:
                print('\nEnabling new step: ', enable, '\n')
                # Now we can enable this step
                # TODO also enable on validation handler
                setattr(handlers.trainer._net, enable, True)
                # Potentially freeze previous stages
                # (it is a cumulative parameter, we only freeze more parameters with time
                # and never un-freeze parameters)
                for module_name in params.get('freeze_below', []):
                    print('Freezing weights for a sub-module',module_name)
                    for name, param in handlers.trainer._net.named_parameters():
                        if module_name in name:
                            param.requires_grad = False
                # Adjust batch size
                batch_size_changed = params.get('batch_size', cfg['iotool']['batch_size']) != cfg['iotool']['batch_size']
                cfg['iotool']['batch_size'] = params.get('batch_size', cfg['iotool']['batch_size'])
                cfg['iotool']['sampler']['batch_size'] = params.get('batch_size', cfg['iotool']['batch_size'])
                # Changing batch size forces us to reload I/O
                if batch_size_changed:
                    print('\nBatch size changed, reloading I/O...\n')
                    # Instantiate DataLoader
                    handlers.data_io = loader_factory(cfg, event_list=event_list)
                    # IO iterator
                    handlers.data_io_iter = iter(cycle(handlers.data_io))

if __name__ == '__main__':
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument("--detect_anomaly",
                        help="Turns on autograd.detect_anomaly for debugging",
                        action='store_true')
    parser.add_argument('chain_config')
    args = parser.parse_args()
    if args.detect_anomaly:
        with torch.autograd.detect_anomaly():
            main(args.config, args.chain_config)
    else:
        main(args.config, args.chain_config)
