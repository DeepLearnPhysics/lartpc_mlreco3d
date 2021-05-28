#!/usr/bin/env python
import os
import sys
import yaml
import numpy as np
import argparse

current_directory = os.path.dirname(os.path.abspath(__file__))
current_directory = os.path.dirname(current_directory)
sys.path.insert(0, current_directory)
from mlreco.main_funcs import process_config, inference

# nohup python3 bin/run.py val_cfg_file.cfg wts/folder/path >> log_val_file.txt &

def main(cfg_file, ckpt_dir):

    if not os.path.isfile(cfg_file):
        cfg_file = os.path.join(current_directory, 'config', sys.argv[1])
    if not os.path.isfile(cfg_file):
        print(cfg_file, 'not found...')
        sys.exit(1)
        
        
    if not os.path.isdir(ckpt_dir):
        print(ckpt_dir, ' not a valid directory!')
        sys.exit(1)
        

    cfg = yaml.load(open(cfg_file, 'r'), Loader=yaml.Loader)

    process_config(cfg)

    log_dir = cfg['trainval']['log_dir']
    
    # loop over configuration files
    wfiles = np.array([w for w in os.listdir(ckpt_dir) if w.endswith('.ckpt')])
    wfiles = np.sort(wfiles)
    for wfile in wfiles:
        print(wfile)
        filename, file_extension = os.path.splitext(wfile)
        # set weight file
        cfg['trainval']['model_path'] = ckpt_dir + '/' + wfile
        # set output log
        cfg['trainval']['log_dir'] = log_dir + '/' + filename
        
        # run inference
        inference(cfg)

if __name__ == '__main__':
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_file')
    parser.add_argument('ckpt_dir')
    parser.add_argument("--detect_anomaly",
                        help="Turns on autograd.detect_anomaly for debugging",
                        action='store_true')
    args = parser.parse_args()
    if args.detect_anomaly:
        with torch.autograd.detect_anomaly():
            main(args.cfg_file, args.ckpt_dir)
    else:
        main(args.cfg_file, args.ckpt_dir)
