#!/usr/bin/env python
import os
import sys
import yaml
import numpy as np

current_directory = os.path.dirname(os.path.abspath(__file__))
current_directory = os.path.dirname(current_directory)
sys.path.insert(0, current_directory)
from mlreco.main_funcs import process_config, inference

# nohup python3 bin/run.py val_cfg_file.cfg wts/folder/path >> log_val_file.txt &

def main():
    cfg_file = sys.argv[1]
    if not os.path.isfile(cfg_file):
        cfg_file = os.path.join(current_directory, 'config', sys.argv[1])
    if not os.path.isfile(cfg_file):
        print(sys.argv[1], 'not found...')
        sys.exit(1)
        
        
    ckpt_dir = sys.argv[2]
    if not os.path.isdir(ckpt_dir):
        print(sys.argv[2], ' not a valid directory!')
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
    main()
