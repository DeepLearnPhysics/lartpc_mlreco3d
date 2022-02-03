import yaml
import argparse
import os, sys

current_directory = os.path.dirname(os.path.abspath(__file__))
current_directory = os.path.dirname(current_directory)
sys.path.insert(0, current_directory)

from mlreco.main_funcs import process_config
from analysis.algorithms.selection import *

def main(analysis_cfg_path, model_cfg_path):

    analysis_config = yaml.load(open(analysis_cfg_path, 'r'), 
                                Loader=yaml.Loader)
    config = yaml.load(open(model_cfg_path, 'r'), Loader=yaml.Loader)
    process_config(config, verbose=False)

    print(analysis_config)
    process_func = eval(analysis_config['analysis']['name'])

    # Run Algorithm
    process_func(config, analysis_config)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('analysis_config')
    args = parser.parse_args()
    main(args.analysis_config, args.config)