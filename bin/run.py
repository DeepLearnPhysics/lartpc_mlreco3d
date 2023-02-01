#!/usr/bin/env python
import os
import sys
import yaml
from os import environ
import argparse

current_directory = os.path.dirname(os.path.abspath(__file__))
current_directory = os.path.dirname(current_directory)
sys.path.insert(0, current_directory)
from mlreco.main_funcs import process_config, train, inference


def main(config):
    cfg_file = config
    if not os.path.isfile(cfg_file):
        cfg_file = os.path.join(current_directory, 'config', config)
    if not os.path.isfile(cfg_file):
        print(config, 'not found...')
        sys.exit(1)

    cfg = yaml.load(open(cfg_file, 'r'), Loader=yaml.Loader)

    if environ.get('CUDA_VISIBLE_DEVICES') is not None and cfg['trainval']['gpus'] == '-1':
        cfg['trainval']['gpus'] = os.getenv('CUDA_VISIBLE_DEVICES')

    process_config(cfg)

    if cfg['trainval']['train']:
        train(cfg)
    else:
        inference(cfg)

if __name__ == '__main__':
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument("--detect_anomaly",
                        help="Turns on autograd.detect_anomaly for debugging",
                        action='store_true')
    args = parser.parse_args()
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    main(args.config)
