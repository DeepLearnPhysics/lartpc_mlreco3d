import numpy as np
import pandas as pd
import sys
import os, re
import torch
import yaml
import time
from pathlib import Path
import argparse

current_directory = os.path.dirname(os.path.abspath(__file__))
current_directory = os.path.dirname(current_directory)
sys.path.insert(0, current_directory)

from mlreco.main_funcs import process_config, train, inference
from mlreco.utils.dense_cluster import *
from pprint import pprint

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-test_cfg', '--test_config', help='config_path', type=str)
    #parser.add_argument('-ckpt', '--checkpoint_number', type=int)
    args = parser.parse_args()
    args = vars(args)
    cfg = yaml.load(open(args['test_config'], 'r'), Loader=yaml.Loader)

    train_cfg = cfg['config_path']
    print("-------------__CFG__---------------")
    pprint(cfg)
    mode = cfg.get('mode', False)
    print(mode)

    start = time.time()
    if mode == 'optimize':
        output = main_loop_parameter_search(train_cfg, **cfg)
    elif mode == 'voxel_cut':
        output = main_loop_voxel_cut(train_cfg, **cfg)
    else:
        output = main_loop(train_cfg, **cfg)
    end = time.time()
    print("Time = {}".format(end - start))
    name = '{}.csv'.format(cfg['name'])
    if not os.path.exists(cfg['target']):
        os.mkdir(cfg['target'])
    target = os.path.join(cfg['target'], name)
    output.to_csv(target, index=False, mode='a', chunksize=50)
