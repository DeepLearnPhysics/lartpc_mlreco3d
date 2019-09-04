#!/usr/bin/env python
import os
import sys
import yaml
from os import environ

current_directory = os.path.dirname(os.path.abspath(__file__))
current_directory = os.path.dirname(current_directory)
sys.path.insert(0, current_directory)
from mlreco.main_funcs import process_config, train, inference


def main():
    cfg_file = sys.argv[1]
    if not os.path.isfile(cfg_file):
        cfg_file = os.path.join(current_directory, 'config', sys.argv[1])
    if not os.path.isfile(cfg_file):
        print(sys.argv[1], 'not found...')
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
    main()
